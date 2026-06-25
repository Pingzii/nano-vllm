import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    model_path = os.path.expanduser("/code/qwen3_0.6B")

    # 加载 tokenizer（本地无需 repo_type）
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型有自定义代码
        local_files_only=True  # 强制仅使用本地文件，避免联网检查
    )

    # 加载 vLLM 模型
    llm = LLM(
        model=model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        trust_remote_code=True  # vLLM 也支持该参数
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        '''
        ### Kernel & operator fusion
- **FlashAttention** — `flash_attn_varlen_func` for ragged prefill, `flash_attn_with_kvcache` for paged decode
- **Triton store_kvcache** — custom Triton kernel writes K/V into cache slots indexed by `slot_mapping`
- **QKV projection fusion** — Q/K/V linear layers fused into single `QKVParallelLinear`
- **Gate-Up fusion** — `gate_proj` + `up_proj` fused into `MergedColumnParallelLinear`
- **Fused residual + RMSNorm** — `add_rms_forward` combines residual add and RMSNorm in a single `torch.compile`-friendly kernel
- **Fused SiLU + multiply** — `SiluAndMul` gate activation in one kernel
        ''',
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
