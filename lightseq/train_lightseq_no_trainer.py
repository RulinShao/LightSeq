import os
from typing import Optional
from dataclasses import dataclass, field
import time
from tqdm import tqdm
import functools

# import fastckpt before transformers
from lightseq_ckpt_monkey_patch import replace_hf_ckpt_with_new_ckpt, clear_all_buffers_at_the_end_of_training
replace_hf_ckpt_with_new_ckpt()

import numpy as np
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name    
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP 
import torch.distributed as dist

from async_communication import reset_global_memory_buffer


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    rope_scaling_mode: Optional[str] = field(default=None)
    rope_scaling_factor: Optional[int] = field(default=1.0)


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def train():
    global local_rank
    global model_path

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_config(config).to(device="cuda", dtype=torch.float16)
    model.model.gradient_checkpointing = True

    def initialize_distributed():
        if dist.is_initialized():
            if dist.get_rank() == 0:
                print(
                    "torch distributed is already initialized, "
                    "skipping initialization ...",
                    flush=True,
                )
        else:
            if int(os.environ["RANK"]) == 0:
                print("Initializing Torch distributed.")
            dist.init_process_group(backend="nccl")
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            global_world_size = dist.get_world_size()
            torch.cuda.set_device(dist.get_rank() % local_world_size)

    initialize_distributed()
    transformer_cls_to_wrap = set()
    transformer_cls = get_module_class_from_name(model, "LlamaDecoderLayer")
    if transformer_cls is None:
        raise Exception("Could not find the transformer layer class to wrap in the model.")
    else:
        transformer_cls_to_wrap.add(transformer_cls)

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls=transformer_cls_to_wrap,
        )
    print(config)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"trainable params {params}")
    # Wrap the base model with an outer FSDP wrapper
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD
        #sharding_strategy=ShardingStrategy.SHARD_GRAD_OP
        )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    #optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=0.01)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=0.01)

    for length in [4096, 8192, 16384, 32768]:
        time_list = []
        data = {"input_ids": torch.randint(0, 1000, (1, length,), device="cuda"),
                "labels": torch.randint(0, 1000, (1, length,), device="cuda"), 
                "attention_mask": torch.ones(1, length, device="cuda")}
        for i in tqdm(range(100)):
            if i > 0:
                time_s = time.time()
            outputs = model(**data, use_cache=True)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i > 0:
                time_e = time.time()
                time_list.append(time_e - time_s)
                print(f"Cumulated time list {time_list}")
        time_arr = np.asarray(time_list) * 1000.0
        print(f"Per GPU shape: {data['input_ids'].shape}, {np.mean(time_arr)}, {np.std(time_arr)}")
        reset_global_memory_buffer()
        clear_all_buffers_at_the_end_of_training()


if __name__ == "__main__":
    train()
