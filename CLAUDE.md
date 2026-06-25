# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Nano-vLLM is a lightweight vLLM implementation (~1200 lines) with comparable inference speed (1434 tok/s vs vLLM's 1362 tok/s on Qwen3-0.6B + RTX 4070).

Only Qwen3 is currently supported as the model architecture (`nanovllm/models/qwen3.py`).

## Feature inventory

### Core inference
- **Continuous batching** — prefill and decode alternate per step; multiple seqs batched within each
- **Ragged (varlen) prefill** — FlashAttention varlen mode, seqs of different lengths concatenated into one batch with `cu_seqlens` indexing
- **Chunked prefill** — long prompts split across multiple steps to avoid starving decode; only the first seq in a batch can be chunked
- **CUDA graph decode** — forward pass captured for batch sizes [1,2,4,8,16,32,...,512] (geometric then stride-16); graph pool shared across all captured graphs
- **`torch.compile`** — applied on Sampler, RMSNorm, SiluAndMul, RotaryEmbedding for fusion and reduced launch overhead

### KV-cache & memory
- **PagedAttention** — KV-cache split into fixed-size blocks (`block_size=256`), `block_table` maps logical→physical blocks, reference-counted sharing
- **Prefix caching (chain xxhash)** — `h_i = xxhash(token_ids[i], prefix=h_{i-1})` ensures same tokens in different contexts produce different hashes; released blocks retain hash + token_ids for cold-cache hits
- **Preemption** — when decode runs out of free blocks, evict tail of running queue; evicted seq releases all blocks but keeps CPU token_ids, re-prefills later (potentially hitting cold prefix cache)
- **Dynamic KV-cache sizing** — `num_kvcache_blocks` computed at init from `gpu_memory_utilization * total - current_usage`, no hardcoded block count

### Kernel & operator fusion
- **FlashAttention** — `flash_attn_varlen_func` for ragged prefill, `flash_attn_with_kvcache` for paged decode
- **Triton store_kvcache** — custom Triton kernel writes K/V into cache slots indexed by `slot_mapping`
- **QKV projection fusion** — Q/K/V linear layers fused into single `QKVParallelLinear`
- **Gate-Up fusion** — `gate_proj` + `up_proj` fused into `MergedColumnParallelLinear`
- **Fused residual + RMSNorm** — `add_rms_forward` combines residual add and RMSNorm in a single `torch.compile`-friendly kernel
- **Fused SiLU + multiply** — `SiluAndMul` gate activation in one kernel

### Parallelism
- **Tensor parallelism** — TP-aware linear layers (ColumnParallelLinear, RowParallelLinear, QKVParallelLinear), embedding (VocabParallelEmbedding), and LM head (ParallelLMHead). IPC via SharedMemory + Event with pickle serialization. NCCL handles all_reduce/gather.
- **GQA / MQA** — Qwen3 supports different `num_attention_heads` and `num_key_value_heads`; Q/K/V sharding respects GQA ratio under TP

### Qwen3-specific
- **QKNorm** — RMSNorm applied to Q and K before attention (when `qkv_bias=False`)
- **RoPE** — rotary position embedding with precomputed cos/sin cache, `torch.compile`-accelerated
- **Tied embeddings** — `lm_head` shares weights with `embed_tokens` when `tie_word_embeddings=True`

### Sampling
- **Temperature sampling** — Gumbel-max trick (`-log(-log(uniform)) / temperature + logits → argmax`), greedy mode not supported (`temperature > 1e-10` enforced)

## Build / run

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
# or editable: pip install -e .
```

No test suite exists. The project is verified by running:
- `python example.py` — basic generation (modify `model_path` to point to a local Qwen3-0.6B)
- `python bench.py` — throughput benchmark vs vLLM

## Architecture

```
LLM (nanovllm/llm.py)
  └─ inherits LLMEngine (nanovllm/engine/llm_engine.py)
       ├─ Scheduler (engine/scheduler.py) — CPU-side: decides what to run each step
       │    └─ BlockManager (engine/block_manager.py) — KV-cache physical block allocator
       └─ ModelRunner (engine/model_runner.py) — GPU-side: owns model, KV-cache tensor, CUDA graphs
```

### Data flow per step

```
schedule() → pick seqs + decide prefill/decode
  → prepare_prefill/decode() → build input_ids, positions, slot_mapping
    → run_model() → model.forward() on GPU
      → Attention.forward() → Triton store_kvcache writes K/V to cache slots
                             → FlashAttention reads from cache via block_tables
    → sampler() → pick tokens (rank 0 only)
  → postprocess() → hash_blocks (register prefix cache), append tokens, check finish
```

### Key abstractions

**Sequence** (`engine/sequence.py`): Represents one generation request. Holds `token_ids` (CPU), `block_table` (logical→physical block mapping), `num_cached_tokens` (how many tokens already in KV-cache), `num_scheduled_tokens` (tokens to process this step). Lifecycle: WAITING → RUNNING → FINISHED.

**BlockManager** (`engine/block_manager.py`): Manages fixed-size physical KV-cache blocks (`block_size=256`). Uses `ref_count` for sharing (prefix cache hits). Implements chain xxhash where `h_i = xxhash(token_ids[i], prefix=h_{i-1})` — this ensures same tokens in different contexts produce different hashes. `hash_to_block_id` is the global lookup; released blocks keep their hash/`token_ids` for cold-cache hits.

**Slot mapping**: The bridge between logical blocks and physical addresses. `slot = block_table[block_idx] * block_size + offset_in_block`. Each token gets a slot; the Triton kernel writes `k_cache[slot] = key`.

**KV-cache tensor** (`model_runner.py:allocate_kv_cache`): Shape `(2, n_layers, n_blocks, block_size, n_kv_heads, head_dim)`. Index 0=K, 1=V. Sliced and bound to each Attention layer's `k_cache`/`v_cache` attribute. Allocation size is computed dynamically from `gpu_memory_utilization * total - current_usage`.

### Scheduling policy

1. **Prefill-first**: process `waiting` queue first. Each seq may be fully or partially processed (chunked prefill when token budget exceeded). The first seq in a step can be chunked; subsequent ones cannot (prevents fragmentation).
2. **Decode**: only when no prefill possible. Round-robin via `running.popleft()` + `extendleft(reversed(...))`.
3. **Preemption**: when decode runs out of free blocks, evict the tail of `running` (least recently scheduled). Evicted seq loses all KV-cache blocks but keeps `token_ids` on CPU; re-enters `waiting` head and is re-prefilled later, potentially hitting cold prefix cache.

### Prefix cache lifecycle

```
can_allocate(seq)  → chain-hash lookup in hash_to_block_id, count matches
allocate(seq, n)   → reuse cached blocks (ref_count++) + allocate new ones
hash_blocks(seq)   → after prefill, register newly-completed blocks into hash_to_block_id
deallocate(seq)    → ref_count-- on all blocks; blocks reaching 0 return to free pool
                       (hash/token_ids preserved for cold-cache hits)
```

### Tensor parallelism

When `tensor_parallel_size > 1`, worker processes run `ModelRunner.loop()` waiting on shared memory events. Rank 0 writes method calls via `pickle` to `SharedMemory`, sets `Event` to notify workers, then executes locally. All ranks execute the same model forward; NCCL handles all_reduce/gather. Only rank 0 runs the sampler.
