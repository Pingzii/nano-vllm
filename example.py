"""
Nano-vLLM 功能测试脚本。

运行前请修改 MODEL_PATH 指向本地的 Qwen3-0.6B 模型目录。

测试用例一览:
  测试1 — 基础生成: 验证模型能正常推理、streaming 不报错
  测试2 — Chunked Prefill: 调小 max_num_batched_tokens，用长 prompt 触发分块
  测试3 — Prefix Cache: 两个 prompt 共享长前缀，验证缓存命中
  测试4 — 混合测试: 同时触发分块 prefill + 前缀缓存复用
"""

import os
import time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ============================================================
# 配置
# ============================================================
MODEL_PATH = os.path.expanduser("/code/qwen3_0.6B")
MAX_OUTPUT = 32  # 每个用例的输出 token 数，减少总运行时间


def chat(prompt: str) -> str:
    """将纯文本 prompt 包装为 Qwen3 chat template。"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ============================================================
# 测试1：基础生成
# ============================================================
def test_basic(llm):
    """
    测试1 — 基础生成
    ─────────────────
    最简单的用法：两个普通 prompt，验证模型能正常推理并返回文本。
    """
    print("\n" + "=" * 60)
    print("测试1 — 基础生成")
    print("=" * 60)

    prompts = [
        chat("introduce yourself"),
        chat("Explain what is machine learning in one sentence."),
    ]

    print(f"Prompt 1 tokens: {len(tokenizer.encode(prompts[0]))}")
    print(f"Prompt 2 tokens: {len(tokenizer.encode(prompts[1]))}")

    t0 = time.time()
    outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=MAX_OUTPUT))
    print(f"Elapsed: {time.time() - t0:.2f}s")

    for i, out in enumerate(outputs):
        print(f"\n--- Output {i+1} ---")
        print(out["text"][:200])


# ============================================================
# 测试2：Chunked Prefill
# ============================================================
def test_chunked_prefill(llm):
    """
    测试2 — Chunked Prefill（分块预填充）
    ─────────────────────────────────────
    max_num_batched_tokens=256，用 ~700 token 长 prompt 强制分块执行。
    """
    print("\n" + "=" * 60)
    print("测试2 — Chunked Prefill")
    print("=" * 60)

    # 构造 ~700 token 的 prompt（重复一句话）
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 70
    prompt = chat(long_prompt)
    print(f"Prompt tokens: {len(tokenizer.encode(prompt))} "
          f"(>> max_num_batched_tokens=256, will be chunked)")

    sampling_params = SamplingParams(temperature=0.6, max_tokens=MAX_OUTPUT)
    t0 = time.time()
    outputs = llm.generate([prompt], sampling_params)
    print(f"Elapsed: {time.time() - t0:.2f}s")
    print(f"Output: {outputs[0]['text'][:200]}")


# ============================================================
# 测试3：Prefix Cache（前缀缓存）
# ============================================================
def test_prefix_cache(llm):
    """
    测试3 — Prefix Cache（前缀缓存）
    ────────────────────────────────
    两个 prompt 共享 ~300 token 公共前缀（>=1 个 block），
    第一个 prefill 后前缀 block 注册到 hash_to_block_id，
    第二个命中缓存，只需 prefill 后缀部分。
    第三个 prompt 无共享前缀，作为对照组。
    """
    print("\n" + "=" * 60)
    print("测试3 — Prefix Cache")
    print("=" * 60)

    # 公共前缀：重复一句话凑够 ~300 token（> block_size=256）
    common = (
        "You are a knowledgeable AI assistant who provides accurate, "
        "well-structured answers to any question. "
    ) * 20  # ~300 tokens

    p1 = chat(common + "\nUser question: What is the speed of light?")
    p2 = chat(common + "\nUser question: Who wrote Romeo and Juliet?")
    p3 = chat("Tell me a short joke.")  # 对照组：完全不同的 prompt

    print(f"Prompt 1 tokens: {len(tokenizer.encode(p1))}")
    print(f"Prompt 2 tokens: {len(tokenizer.encode(p2))}  "
          f"(shares ~{len(tokenizer.encode(common))} tokens prefix with prompt 1)")
    print(f"Prompt 3 tokens: {len(tokenizer.encode(p3))}  (no shared prefix)")

    t0 = time.time()
    outputs = llm.generate(
        [p1, p2, p3],
        SamplingParams(temperature=0.6, max_tokens=MAX_OUTPUT),
    )
    print(f"Elapsed: {time.time() - t0:.2f}s")

    for i, out in enumerate(outputs):
        print(f"\n--- Output {i+1} ---")
        print(out["text"][:200])


# ============================================================
# 测试4：混合测试（分块 + 前缀缓存）
# ============================================================
def test_combined(llm):
    """
    测试4 — 混合测试：分块 Prefill + 前缀缓存
    ──────────────────────────────────────────
    两个 prompt 共享 ~500 token 长前缀（>max_num_batched_tokens=256），
    同时触发分块 + 缓存复用。第三个 prompt 无关，对比验证。
    """
    print("\n" + "=" * 60)
    print("测试4 — 混合测试：分块 + 前缀缓存")
    print("=" * 60)

    # 公共前缀 ~500 token（>1 block，也 >256 触发分块）
    common = (
        "System: You are a highly capable AI assistant with expertise in "
        "computer science, mathematics, physics, and engineering. "
        "Please provide detailed and accurate answers to all questions. "
    ) * 25  # ~500 tokens

    p1 = chat(common + "\nUser: Explain what a binary search tree is.")
    p2 = chat(common + "\nUser: Explain what a hash table is.")
    p3 = chat("Hi!")  # 对照组

    common_len = len(tokenizer.encode(common))
    print(f"Shared prefix: {common_len} tokens (> max_num_batched_tokens=256 → chunked)")
    print(f"  block_size=256 → shared prefix spans ~{common_len // 256} complete blocks")
    print(f"Prompt 1 total: {len(tokenizer.encode(p1))} tokens")
    print(f"Prompt 2 total: {len(tokenizer.encode(p2))} tokens  "
          f"(prefix hit expected)")
    print(f"Prompt 3: {len(tokenizer.encode(p3))} tokens  (baseline, no cache hit)")

    t0 = time.time()
    outputs = llm.generate(
        [p1, p2, p3],
        SamplingParams(temperature=0.6, max_tokens=MAX_OUTPUT),
    )
    print(f"Elapsed: {time.time() - t0:.2f}s")

    for i, out in enumerate(outputs):
        print(f"\n--- Output {i+1} ---")
        print(out["text"][:200])


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    # 所有测试共用一个 LLM 实例（dist.init_process_group 只能调用一次）
    llm = LLM(
        model=MODEL_PATH,
        enforce_eager=True,
        max_num_batched_tokens=256,   # 设小，方便触发分块 prefill
        max_num_seqs=16,
    )

    test_basic(llm)
    test_chunked_prefill(llm)
    test_prefix_cache(llm)
    test_combined(llm)
