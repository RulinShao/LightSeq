# LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers
Official repository for LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers

Paper: https://arxiv.org/pdf/2310.03294.pdf

## News
- [2023/08] 🔥 Our paper is on! We provide a code preview of LightSeq. Stay tuned for future releases!

## Code Preview
We show a preview of LightSeq in the current release. We will add instructions and examples of training models with LightSeq soon! 
* `lightseq_async_attn.py` contains codes for [DistAttn](https://github.com/RulinShao/LightSeq/blob/main/lightseq/lightseq_async_attn.py#L436) adapted from flash attention kernel.
* `async_communication.py` contains codes for communication-computation overlapped and workload-balanced communication.

## Rematerialization-aware gradient checkpointing
Rematerialization-aware gradient checkpointing can save your training time in one line! 
We plan to release it as a Python package, [`fastckpt`](https://github.com/RulinShao/fast-ckpt), so you can use it by simply running

```python
# import fastckpt before importing transformers
from fastckpt.llama_flash_attn_monkey_patch import replace_hf_ckpt_with_fast_ckpt
replace_llama_attn_with_flash_attn()

# import transformers and other packages
import transformers
...
```

The package will be ready to use soon. Stay tuned for a formal release!
