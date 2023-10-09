# LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers
Official repository for LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers. LightSeq achieves up to 2x faster, 2-8x longer sequences vs Megatron-LM on 16 80GB A100s.

Paper: https://arxiv.org/pdf/2310.03294.pdf

### WARNING: This repo has not been fully tested for end-to-end training. We are actively working on that, with additional feature such as allowing padding in inputs. The current code is only for research preview and reproducing the paper results. We will make public announcement when more support are ready.

## News
- [2023/08] 🔥 Our paper is on! We provide a code preview of LightSeq. Stay tuned for future releases!

## LightSeq Implementation
* `lightseq_async_attn.py` contains codes for [DistAttn](https://github.com/RulinShao/LightSeq/blob/main/lightseq/lightseq_async_attn.py#L436) adapted from flash attention kernel.
* `async_communication.py` contains codes for communication-computation overlapped and workload-balanced communication.

## Usage
We provide an example to use lightseq in training. For example, to run LightSeq on 8 nodes, replace the `data_path` with your own dataset and run

```bash
python -m torch.distributed.run --nproc_per_node=8 \
         lightseq/train_lightseq_no_trainer.py \
        --model_name_or_path Llama-2-7b-chat-hf \
        --data_path <your_dataset>.pkl \
        --bf16 \
        --output_dir outputs \
        --num_train_epochs 3    \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4  \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 1000  \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True  \
        --model_max_length 16384  \
        --gradient_checkpointing True  \
        --lazy_preprocess True
```


## Rematerialization-aware gradient checkpointing
Rematerialization-aware gradient checkpointing can save your training time in one line! 
We release it as a Python package, [`fastckpt`](https://github.com/RulinShao/FastCkpt), so you install it by
```bash
pip install fastckpt
```

To replace **both** HF checkpointing with FashCkpt and HF LlamaAttention with FlashAttention, run 

```python
# import fastckpt before importing transformers
from fastckpt.llama_flash_attn_ckpt_monkey_patch import replace_hf_ckpt_with_fast_ckpt
replace_hf_ckpt_with_fast_ckpt()

# import transformers and other packages
import transformers
...
```

Alternatively, if you only want to replace the attention module with FlashAttention, simply run

```python
# import fastckpt before importing transformers
from fastckpt.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

# import transformers and other packages
import transformers
...
```

Support for one-line importing of LightSeq will come soon.
