import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384 # batchsize * seq_len, max_num_seqs：是并发数上限
    max_num_seqs: int = 512 # 同一批次内，最多并行处理多少条独立请求（并发队列上限）
    max_model_len: int = 4096 # 单条请求允许的最大 token 总数（上下文 + 生成内容加起来）
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)


# ● 三个参数的理解都是正确的。对照代码验证一下：
#
#   max_num_batched_tokens — 正确。在 scheduler.py 的 prefill 阶段作为 token 预算上限：
#
#   remaining = self.max_num_batched_tokens - num_batched_tokens
#
#   prefill 每个 seq 的 num_scheduled_tokens 受此约束，累计不能超过它。decode 阶段不受此限制（每个 seq 只处理 1 个 token）。
#
#   max_num_seqs — 正确。在 scheduler.py 的 prefill 和 decode 阶段都用作 len(scheduled_seqs) 的上限。同时在 capture_cudagraph() 中也用它作为
#   CUDA graph 的最大 batch size：
#
#   max_bs = min(self.config.max_num_seqs, 512)
#
#   max_model_len — 正确。在 config.py 的 __post_init__ 中会和模型的 max_position_embeddings 取较小值：
#
#   self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
#
#   超出这个长度的序列在 KV-cache 分配阶段就会被限制（max_num_blocks 基于它计算）。
#
#   三个描述都没问题，注释可以放心写。