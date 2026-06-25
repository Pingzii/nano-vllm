import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    Nano-vLLM 的顶层推理引擎，协调 Scheduler 和 ModelRunner 完成批量生成。

    架构角色:
        引擎是用户 API (LLM 类) 和底层 GPU 执行之间的桥梁。
        - LLM 继承 LLMEngine，无需额外代码
        - LLMEngine.generate() 是唯一对外暴露的生成接口
        - 内部通过 step() 驱动 Scheduler ↔ ModelRunner 循环

    Tensor Parallelism 架构:
        当 tensor_parallel_size > 1 时，引擎会:
        1. 在主进程 (rank 0) 创建 ModelRunner
        2. 用 mp.spawn 创建 rank 1..N-1 的 worker 子进程，每个运行 ModelRunner
        3. 子进程进入 loop() 死循环，通过 SharedMemory + Event 等待主进程指令
        4. 主进程通过 ModelRunner.call() 同步所有 rank 执行相同操作

    生成循环 (generate 方法):
        ┌─────────────────────────────────────────────┐
        │  add_request() → waiting 队列               │
        │       ↓                                     │
        │  schedule() → 选出本轮 seqs, 确定模式       │
        │       ↓                                     │
        │  model_runner.run() → GPU 推理 → token_ids  │
        │       ↓                                     │
        │  postprocess() → 更新状态, 检查完成         │
        │       ↓                                     │
        │  is_finished()? ──否──→ 循环                │
        │       是                                     │
        │  tokenizer.decode() → 返回文本              │
        └─────────────────────────────────────────────┘
    """

    def __init__(self, model, **kwargs):
        # 从 kwargs 中过滤出 Config 需要的字段，其余忽略（兼容 vLLM 参数）
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 设置 Sequence 的类级别 block_size（所有 Sequence 共享）
        Sequence.block_size = config.kvcache_block_size

        # ---- 创建 Tensor Parallel 进程 ----
        self.ps = []          # 子进程对象列表
        self.events = []      # 每个子进程对应一个 Event，用于通知
        ctx = mp.get_context("spawn")                         # Windows 兼容：必须用 spawn
        for i in range(1, config.tensor_parallel_size): # 子进程编号从1开始，0是主进程
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 主进程 (rank 0) 的 ModelRunner，拥有所有 worker 的 events 用于广播
        self.model_runner = ModelRunner(config, 0, self.events)

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id             # 从 tokenizer 获取 EOS

        # 创建调度器
        self.scheduler = Scheduler(config)

        # 注册进程退出时的清理函数
        atexit.register(self.exit)

    def exit(self):
        """清理所有 TP worker 进程，向每个发送 'exit' 命令后等待结束"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加一个生成请求。

        如果 prompt 是字符串则 tokenize 为 token id 列表。
        创建 Sequence 对象并加入 Scheduler 的 waiting 队列。
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行一步推理（一次 prefill 或一次 decode）。

        流程:
            1. Scheduler.schedule() → 选出本轮处理的序列和模式
            2. ModelRunner.call("run", seqs, is_prefill) → GPU 推理
               - call() 会通过 SharedMemory 同步所有 TP rank
               - 返回采样出的 token_ids 列表
            3. Scheduler.postprocess() → 更新序列状态

        返回:
            outputs: [(seq_id, completion_token_ids), ...]  已完成的序列
            num_tokens: 处理的 token 数（正=prefill, 负=decode）
        """
        seqs, is_prefill = self.scheduler.schedule()

        # 统计 token 数：prefill 是处理的 token 总数，decode 是序列数（取负用于区分）
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

        # 跨 TP 调用 run 方法
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 后处理：更新缓存状态，追加 token，检查终止
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        """所有请求是否已完成"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成接口，调用后阻塞直到所有请求完成。

        参数:
            prompts:          字符串或 token id 列表的列表
            sampling_params:  单个 SamplingParams（会复制给所有 prompt）或列表
            use_tqdm:         是否显示进度条和吞吐量

        返回:
            [{"text": ..., "token_ids": [...]}, ...]  按请求顺序排列

        循环内部的吞吐量计算:
            - prefill_throughput: 处理的 prompt tokens / 耗时
            - decode_throughput:  生成的序列数 / 耗时
              因为 decode 每步每个 seq 只产生 1 个 token，序列数 = 新生成的 token 数
        """
        # 进度条：按完成的序列数更新
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        # 标准化 sampling_params 为列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 批量添加请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.

        # 主循环：一直 step 直到所有请求完成
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)   # prefill: tokens/s
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)  # decode: seqs/s = tokens/s

            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            # 收集已完成的序列
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)                                         # 进度条 +1

        pbar.close()

        # 按 seq_id 排序以保证输出顺序与输入顺序一致
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
                   for token_ids in outputs]
        return outputs
