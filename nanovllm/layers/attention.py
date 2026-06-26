"""
Attention 层 —— 包含 KV-cache 写入（Triton kernel）+ FlashAttention 推理。

================================================================================
一、整体数据流
================================================================================

    Qwen3Attention.forward(q, k, v)
        │
        ├─ ① store_kvcache(k, v, k_cache, v_cache, slot_mapping)
        │      Triton kernel: k_cache[slot] = k,  v_cache[slot] = v
        │      把刚算出的 K/V 按 slot_mapping 写入物理缓存位置
        │
        └─ ② FlashAttention
              ├─ prefill:  flash_attn_varlen_func
              │     - 有 prefix cache 命中 → k,v 改用 cache 中的（包含历史 context）
              │     - 无 prefix cache     → k,v 用刚算出的（只有本轮 prefill 的 token）
              │
              └─ decode:   flash_attn_with_kvcache
                    - q: 当前 1 个 token 的 query
                    - k,v: 从 paged KV-cache 读取完整历史，按 block_table 索引

================================================================================
二、slot_mapping 的角色
================================================================================

    slot_mapping[i] = 第 i 个 token 应写入 KV-cache 的物理槽位号。
    它由 ModelRunner.prepare_prefill/decode 构建，是 BlockManager 的 block_table
    和 Attention 的 KV-cache 之间的唯一桥梁。

    示例（prefill）:
      block_table = [5, 8], block_size=256
      token 0 → slot = 5*256 + 0   = 1280
      token 1 → slot = 5*256 + 1   = 1281
      ...
      token 255 → slot = 5*256 + 255 = 1535
      token 256 → slot = 8*256 + 0   = 2048

    示例（decode，每个 seq 只 1 token）:
      seq A: last token 在 block_table[-1]=3, offset=127
             → slot = 3*256 + 127 = 895

================================================================================
三、KV-cache 张量的 shape
================================================================================

    k_cache / v_cache: (n_blocks, block_size, n_kv_heads, head_dim)
    实际上是 kv_cache[0, layer_id] 和 kv_cache[1, layer_id] 的视图，
    由 ModelRunner.allocate_kv_cache 分配并绑定到每层 Attention 模块。

    k_cache[slot] 直接索引到 (block_id, offset_in_block) 位置:
      block_id    = slot // block_size
      offset      = slot % block_size
      k_cache[block_id, offset] → shape (n_kv_heads, head_dim)
"""

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# ============================================================================
# Triton Kernel: 将 K/V 向量写入 KV-cache
# ============================================================================

@triton.jit
def store_kvcache_kernel(
    key_ptr,           # key 的内存首地址（类型: fp16/bf16*）
    key_stride,        # key 张量第 0 维的 stride，即 token 之间的元素步长
    value_ptr,         # value 的内存首地址
    value_stride,      # value 张量第 0 维 stride
    k_cache_ptr,       # K cache 的内存首地址
    v_cache_ptr,       # V cache 的内存首地址
    slot_mapping_ptr,  # slot 映射表的首地址（int32 数组，长度 N）
    D: tl.constexpr,   # 编译期常量，每个 token 需要搬运的元素数
):
    """
    将 key/value 写入 KV-cache 的 Triton kernel。grid=(N,)，每个 program 处理 1 个 token。

    Triton 的内存模型：
        本 kernel 里所有 ptr 参数都是**原始指针**（不是 torch.Tensor）。
        tl.load(ptr + offset) 从 GPU 显存的 ptr+offset 地址处读取数据。
        tl.store(ptr + offset, data) 向 ptr+offset 地址处写入数据。
        没有边界检查——offset 越界会读到垃圾/写到错误位置，依赖调用方保证正确性。

    为什么 D 是 num_kv_heads * head_dim？
        KV-cache 的 shape 是 (n_blocks, block_size, n_kv_heads, head_dim)。
        我们把它当成 (n_blocks * block_size, n_kv_heads * head_dim) 的 2D 数组，
        每个 slot 对应一行 D 个元素。把 n_kv_heads 和 head_dim 合并为一维，
        一次 tl.load/tl.store 就能搬运一个 token 的全部 K（或 V），只需 1 次 IO。
    """

    # ---- (1) 我是谁？----
    # program_id(0) 返回当前 program 在 grid 第 0 维的索引，范围 0 ~ N-1。
    # 每个 program 处理第 idx 个 token 的 key/value。
    idx = tl.program_id(0)

    # ---- (2) 这个 token 要写到哪个 slot？----
    # slot_mapping 是 int32 数组，每个元素存一个物理槽位号。
    # slot_mapping[idx] 就是从 slot_mapping_ptr 首地址偏移 idx 个 int32 的位置。
    # tl.load 一次读 1 个 int32。
    slot = tl.load(slot_mapping_ptr + idx)

    # slot == -1 是 CUDA graph 的 padding 标记。
    # CUDA graph 要求 batch size 固定，当实际 seq 数 < graph 的 max_bs 时，
    # ModelRunner 把多余位置的 slot_mapping 填 -1。这些位置没有有效的 token，
    # 直接 return 跳过，不写 cache 也不报错。
    if slot == -1:
        return

    # ---- (3) 读取 key[idx, :] 和 value[idx, :] ----
    # 目标：从 key 张量中读出第 idx 个 token 的全部 D 个元素。
    #
    # key 的 shape: (N, num_kv_heads, head_dim)，内存布局可能是：
    #   key[0,0,0], key[0,0,1], ..., key[0,0,head_dim-1],  ← head 0 of token 0
    #   key[0,1,0], key[0,1,1], ..., key[0,1,head_dim-1],  ← head 1 of token 0
    #   ...
    #   key[1,0,0], ...                                      ← head 0 of token 1
    #
    # 第 idx 个 token 的起始地址 = key_ptr + idx * key_stride。
    # 其中 key_stride = num_kv_heads * head_dim = D（连续布局时）。
    # tl.arange(0, D) 生成 [0, 1, 2, ..., D-1]，
    # 所以 key_offsets = [idx*D+0, idx*D+1, ..., idx*D+D-1]，正好覆盖 D 个元素。
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # tl.load 一次读 D 个元素到寄存器，返回 shape (D,) 的向量。
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # ---- (4) 写入 k_cache[slot, :] 和 v_cache[slot, :] ----
    # KV-cache 的内存布局与 key 类似：
    #   k_cache[0, 0, :]  k_cache[0, 1, :]  ...  k_cache[0, block_size-1, :]  ← block 0
    #   k_cache[1, 0, :]  k_cache[1, 1, :]  ...                                  ← block 1
    #
    # 每个 slot 的一行也是 D 个连续元素，所以 slot * D 就是偏移量。
    # cache_offsets = [slot*D+0, slot*D+1, ..., slot*D+D-1]。
    # 这个是写入物理显存，物理显存slot对应的位置，每个slot跨越需要跨过D个元素
    cache_offsets = slot * D + tl.arange(0, D)

    # tl.store 把 D 个寄存器值一次性写回显存。
    # 这一步完成 "KV-cache 第 slot 个位置 = 第 idx 个 token 的 key/value"。
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,          # (N, num_kv_heads, head_dim)
    value: torch.Tensor,        # (N, num_kv_heads, head_dim)
    k_cache: torch.Tensor,      # (n_blocks, block_size, n_kv_heads, head_dim)
    v_cache: torch.Tensor,      # 同上
    slot_mapping: torch.Tensor, # (N,) int32，每个 token 的 cache 写入位置
):
    """
    将新计算的 K/V 写入 KV-cache 的指定槽位。

    prefill 时 N = 本轮处理的所有 prompt token 数（可能上百/上千）。
    decode 时 N = batch_size（每个 seq 只 1 token）。

    stride 断言说明:
      key.stride(-1) == 1         → head_dim 维连续（标准 torch 布局）
      key.stride(1) == head_dim   → token 内 head 之间紧邻不交错
      k_cache.stride(1) == D      → cache 中每个 slot 的 D 个元素连续存放
      这些 stride 用于 Triton kernel 中计算偏移量。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim        # 扁平化: (num_kv_heads * head_dim) 个 float

    # stride 校验：确保 Triton kernel 的指针运算正确
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # 启动 Triton kernel: grid=(N,), 每个 program 处理 1 个 token
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache, slot_mapping, D,
    )


# ============================================================================
# Attention 层
# ============================================================================

class Attention(nn.Module):
    """
    支持 GQA/MQA 的 Attention 层，集成 PagedAttention KV-cache。

    初始化时 k_cache/v_cache 只是占位空张量，真正的缓存由 ModelRunner 在
    allocate_kv_cache() 中分配并替换。
    KV cache 在模型初始化阶段被预先分配为 block pool。
    在 prefill 和 decode 的 forward 过程中，每个 token 的 K/V 向量会实时写入由 slot_mapping 映射的物理 block 位置中，
    实现 paged KV cache 的在线构建与复用。

    forward 流程:
      1. 将当前 K/V 写入物理 cache → store_kvcache(k, v, ...)
      2. 执行 FlashAttention:
         - prefill:  varlen（ragged）模式，可选的 prefix cache 读取
         - decode:   paged KV-cache 模式，从 block_table 索引的历史 cache 读取
    """

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads       # Q 的 head 数（TP 分片后的）
        self.head_dim = head_dim         # 每个 head 的维度
        self.scale = scale               # 1/sqrt(head_dim)，attention softmax 缩放因子
        self.num_kv_heads = num_kv_heads # K/V 的 head 数（GQA: <= num_heads）
        # 占位空张量，ModelRunner.allocate_kv_cache 之后被替换为实际的 GPU 张量视图
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q: query 张量
               prefill: (total_tokens, num_heads, head_dim)  ragged layout
               decode:  (batch_size, num_heads, head_dim)    每个 seq 1 token
            k: key 张量，shape 同 q（但 head 数是 num_kv_heads，GQA 下更少）
            v: value 张量，shape 同 k

        Returns:
            o: attention 输出
               prefill: (total_tokens, num_heads, head_dim)
               decode:  (batch_size, num_heads, head_dim)
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # ---- 步骤 1: 写入 KV-cache ----
        # k_cache.numel() == 0 发生在模型预热（warmup）阶段，那时还没调用
        # allocate_kv_cache，k_cache 仍是初始的空张量，跳过写入。
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # ---- 步骤 2: FlashAttention ----
        if context.is_prefill:
            # ========== Prefill 路径 ==========
            # 使用 flash_attn_varlen_func: 支持不等长序列拼接（ragged layout）。
            # 所有 seq 的 token 拼接成一个 batch，通过 cu_seqlens 标记边界。
            #
            # prefix cache 逻辑:
            #   如果 context.block_tables is not None，说明有 prefix cache 命中。
            #   cu_seqlens_k > cu_seqlens_q（K 序列比 Q 序列长，多出的是缓存部分）。
            #   此时把 k,v 替换为 k_cache/v_cache，FlashAttention 会通过
            #   block_table 从 cache 中读取历史的 K/V。
            #
            #   如果没有命中，k,v 就是刚算出的（和 q 同长度），block_table=None
            #   走标准的 varlen attention。
            if context.block_tables is not None:
                k, v = k_cache, v_cache    # 从 cache 读（包含历史 context）

            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,  # None → 普通 varlen；非 None → paged 读取
            )
        else:
            # ========== Decode 路径 ==========
            # 使用 flash_attn_with_kvcache: 专门为 decode 优化的 paged attention。
            # q: (batch_size, 1, num_heads, head_dim) — 每个 seq 只有当前 token 的 query
            # k_cache/v_cache: 预分配的 paged cache，包含所有历史 token 的 K/V
            # block_table:     每个 seq 的逻辑→物理 block 映射表
            # cache_seqlens:   每个 seq 的 K/V 长度（用于 causal mask 边界）
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),                      # 增加 seqlen 维度: (bs,) → (bs, 1, n_heads, hd)
                k_cache, v_cache,
                cache_seqlens=context.context_lens,  # 每个 seq 已缓存的 token 总数
                block_table=context.block_tables,    # 物理 block 索引表
                softmax_scale=self.scale,
                causal=True,
            )

        return o


"""
我原来以为prefill阶段，是没有kv cache利用的，只是生成kv cache，没想到prefill阶段也能用之前的prefix cache，读取kv cache
你这个误解其实非常常见，而且你现在已经踩到 **vLLM / PagedAttention 的一个关键设计点了**：prefill ≠ “纯计算”，它是“计算 + cache 复用”的混合阶段。

我们把它讲清楚。

---

# 1. 先纠正一个核心认知

## ❌ 旧理解（很多人一开始都会这样想）

> prefill = 只算 QKV → 写 KV cache
> decode = 读 KV cache + 写 KV cache

---

## ✔ 实际情况是

> **prefill = 可以读 KV cache（prefix cache） + 计算新 KV + 写 KV cache**

也就是说：

```text id="pfk8x1"
prefill ≠ 纯写入阶段
prefill = 计算 + 可选复用历史 KV
```

---

# 2. 为什么 prefill 也能用 KV cache？

关键原因是：

> LLM 请求经常不是“从零开始”，而是“共享前缀”

---

## 典型场景

### 1）system prompt 复用

```text id="x7v3qp"
System: You are a helpful assistant...
User A: ...
User B: ...
User C: ...
```

system prompt 完全相同

---

### 2）RAG / 多轮对话

```text id="c9kq2m"
history prefix = 固定
new question = 不同
```

---

### 3）多请求 batch

多个 request 共享 prefix：

```text id="h2v9ld"
Prompt A: [A B C D E F]
Prompt B: [A B C X Y Z]
```

👉 前半段完全一样

---

# 3. prefix cache 在 prefill 中是怎么工作的？

你代码里这一段是关键：

```python id="r7m2aa"
if context.block_tables is not None:
    k, v = k_cache, v_cache
```

---

## 这句话的真实含义是：

> “当前 prefill 的一部分 K/V 不需要重新计算，而是直接从 KV cache 读取”

---

# 4. prefill + prefix cache 的计算图

我们拆一个例子：

---

## 输入 prompt

```text id="v3kq8p"
prefix (cached):  A B C D
new tokens:       E F G
```

---

## Step 1：prefill attention

不是只算 EFG，而是：

```text id="k2x9lm"
Q = A B C D E F G
K = A B C D E F G（但 A B C D 来自 cache）
V = 同理
```

---

## Step 2：KV来源混合

| token   | KV来源     |
| ------- | -------- |
| A B C D | KV cache |
| E F G   | 本次计算     |

---

# 5. block_table 在这里的作用

```text id="t8qv2d"
block_table
```

本质是：

> logical token → physical KV block mapping

---

## prefill 时发生：

* 已存在 prefix → 直接映射 block
* 新 token → 分配新 block
* flash_attn_varlen 统一读取

---

# 6. 为什么 decode 必须用 cache，而 prefill 是“可选用 cache”？

## decode：

```text id="d8m2pz"
必须读 KV cache（否则无法逐 token 推理）
```

---

## prefill：

```text id="p1v7qs"
可以不用 cache，也可以复用 cache
```

因为：

> prefill 是“重算整段 prompt”的过程，本身不依赖历史状态

但系统为了省算力，会尝试复用 prefix。

---

# 7. 一句话本质区别（非常重要）

## ✔ prefill：

> “构建 + 复用 KV cache 的 attention”

---

## ✔ decode：

> “依赖 KV cache 的 incremental attention”

---

# 8. 一个更系统级的理解（建议记住）

可以这样理解整个流程：

```text id="s7qk1m"
Prefill：
    把 prompt 写进 memory（KV cache）
    + 可能 reuse 已存在 memory

Decode：
    只使用 memory 做增量推理
```

---

# 9. 为什么你会误解（很正常）

因为很多教材默认：

```text id="u2kq9v"
cache = decode 专属优化
```

但工业系统里其实是：

> cache = 全生命周期 memory（prefill + decode 共用）

---

# 10. 再给你一个更底层的直觉

你可以把 KV cache 想成：

> 一个“外置 RAM”

---

### prefill：

```text id="q9v2lm"
CPU 写 RAM + 读 RAM（可能已有内容）
```

### decode：

```text id="n3xv8a"
CPU 每步都读 RAM + 写 RAM
```

---

# 如果你想继续往下挖（这个很关键）

你现在已经到系统设计关键节点了，下一步可以深入：

### 👉 prefix cache 是怎么命中的？

* hash（xxhash / sha-like）
* block fingerprint
* trie / radix tree cache
* prompt dedup

---

### 👉 为什么 block_table 可以“拼接 cache + new tokens”？

这涉及：

* paged memory allocator
* fragmentation
* KV reuse policy

---

如果你愿意，我可以把这一整套：

> prefix cache → block_table → KV reuse → scheduling

给你串成一条完整 vLLM 推理引擎链路。


可以，用一个具体例子把这两个函数在 **prefill / decode** 下的行为讲清楚，你就会彻底通。

---

# 1. 一个真实 batch 例子（最关键）

假设我们一次 batch 进来 2 个请求：

```text id="ex1"
Request A: "Hello world"
Request B: "I love AI"
```

token 化后：

```text id="ex2"
A: [a1 a2]        (len=2)
B: [b1 b2 b3]     (len=3)
```

---

## 拼接成 FlashAttention 输入（prefill）

```text id="ex3"
q/k/v (flatten):
[a1 a2 b1 b2 b3]
```

---

## cu_seqlens（核心隔离工具）

```text id="ex4"
cu_seqlens = [0, 2, 5]
```

含义：

```text id="ex5"
seq A: [0, 2)
seq B: [2, 5)
```

---

# 2. flash_attn_varlen_func 是怎么工作的？

## ✔ 本质目标

> 在同一个大 tensor 里，**让不同 request 的 attention 完全隔离**

---

## ✔ kernel 内部做的事

对于任意 query token，比如 `b2`：

### Step 1：找所属序列

```text id="ex6"
b2 ∈ [2,5) → 属于 Request B
```

---

### Step 2：只在 B 的范围内做 attention

```text id="ex7"
b2 只能 attend:
b1 b2（以及可能的 prefix cache）
```

---

### ❗关键点（回答你问题）

> ✔ 是的，cu_seqlens 就是用来“屏蔽不同请求之间 attention 干扰”的

但机制不是 mask，而是：

> 👉 **根本不让 kernel 去算跨区间 attention**

---

## ✔ 也就是说：

不是：

```text id="ex8"
softmax(masked QK^T)
```

而是：

```text id="ex9"
Q_i 只会和同 sequence 的 K 做 matmul
```

---

# 3. 如果有 prefix cache，会发生什么？

假设：

```text id="ex10"
Request A 的 "Hello" 已经 cache 过
```

那么：

```text id="ex11"
A = [cached tokens] + [new tokens]
```

---

## block_table 介入：

```text id="ex12"
K/V = cache部分 + 当前计算部分
```

---

## 所以 prefill 其实是：

```text id="ex13"
Q = new tokens
K = cached + new
V = cached + new
```

---

# 4. decode 阶段（flash_attn_with_kvcache）

现在变成：

```text id="ex14"
Request A: 1 token
Request B: 1 token
```

---

## KV cache 已经存在：

```text id="ex15"
A cache: [a1 a2]
B cache: [b1 b2 b3]
```

---

## 每一步 decode：

### A 当前 token a3：

```text id="ex16"
Q = a3
K = [a1 a2 a3]
V = 同上
```

---

## kernel 做的事：

```text id="ex17"
从 block_table 找到 A 的 KV blocks
逐 block 读取 K/V
做 streaming attention
```

---

# 5. 两个函数的本质区别（核心总结）

## ✔ flash_attn_varlen_func（prefill）

```text id="ex18"
输入：连续 tensor + cu_seqlens
作用：在“逻辑序列边界内”做 attention
特点：不依赖 KV cache
```

👉 关键词：

> “sequence-level isolation”

---

## ✔ flash_attn_with_kvcache（decode）

```text id="ex19"
输入：KV cache + block_table
作用：跨 block 读取历史 KV 做 attention
特点：必须依赖 cache
```

👉 关键词：

> “memory-level retrieval”

---

# 6. 回答你的核心问题（非常重要）

## ❓“prefill 是不是用 cu_seqlens 隔离不同请求 attention？”

### ✔ 答案是：

是的，但更精确说法是：

> cu_seqlens 定义了 **attention 的计算边界（computation boundary）**

---

## ❗但注意：

它不是 mask，也不是 softmax 屏蔽，而是：

> 👉 **kernel 在执行时根本不会跨 segment 读取 K/V**

---

# 7. 一句话帮你彻底定性

## prefill：

> 在连续内存上，通过 cu_seqlens 划分多个独立 attention 子问题

---

## decode：

> 在分页 KV cache 上，通过 block_table 做 memory-retrieval attention

---

# 8. 一个非常关键的工程直觉

你可以这样记：

```text id="ex20"
prefill = “算图问题”（graph partition）
decode  = “存储问题”（memory access）
```

---

# 如果你下一步想继续深入（强烈建议）

你现在已经理解“计算层”了，下一步真正核心是：

### 👉 block_table vs cu_seqlens 的本质区别

以及：

* 为什么 prefill 不需要 block_table 也能跑
* 为什么 decode 必须 block paging
* KV cache 为什么必须分页而不是连续 tensor
* online softmax 在 decode 里怎么做

这个是 vLLM 核心分界线。

"""