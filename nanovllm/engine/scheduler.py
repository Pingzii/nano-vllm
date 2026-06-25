"""
Nano-vLLM 调度器 —— 推理引擎的控制中枢。

================================================================================
一、调度器在系统中的位置
================================================================================

    [用户请求]
        │
        ▼
    LLMEngine.generate()
        │
        ├─ add_request() → Scheduler.waiting 队列
        │
        └─ 主循环:
              step()
                ├─ schedule()          ← ① 选出本轮处理的序列
                ├─ model_runner.run()  ← ② GPU 推理
                └─ postprocess()       ← ③ 更新状态、检查终止

    Scheduler 只负责 ① 和 ③，不碰 GPU。它是纯粹的 CPU 端调度逻辑。

================================================================================
二、双队列架构
================================================================================

    waiting (deque)              running (deque)
    ┌─────────────────┐        ┌─────────────────┐
    │ 新请求           │        │ seq_A (decode中) │
    │ 被抢占的序列      │  调度   │ seq_B (decode中) │
    │ 等待重试的序列    │ ────→  │ seq_C (decode中) │
    │                  │        │ ...             │
    └─────────────────┘        └─────────────────┘
          ▲                            │
          │                            │ 序列完成 / 被抢占
          └────────────────────────────┘

    - waiting: 还没拿到 block 资源，或拿到了但被抢占后需要重试
    - running: 已分配 block_table，参与每步 decode

================================================================================
三、调度策略（schedule 的核心逻辑）
================================================================================

    优先级:  Prefill > Decode

    1. Prefill 阶段（处理 waiting 队列）:
       - 只 peek 队首（不 pop），因为可能分块处理
       - can_allocate() 检查 prefix cache 命中 + 空闲 block 是否足够
       - 分块 prefill：token 预算不够一次处理完时砍断，下次继续
       - 全部 token 处理完后才 pop + 移入 running

    2. Decode 阶段（处理 running 队列）:
       - 每个 seq 只处理 1 个 token
       - 空闲 block 不够时触发抢占（preempt）
       - 调度完逆序放回，实现 round-robin 公平轮转

    3. 抢占 (Preemption):
       - 发生在 decode 阶段 block 不够用时
       - 选择 running 队尾的 seq 驱逐（最近最少调度，牺牲它对公平性影响最小）
       - 释放其所有 block，放回 waiting 头部优先重试

================================================================================
四、分块 Prefill (Chunked Prefill) 的目的
================================================================================

    长 prompt（比如几千个 token）如果一次 prefill 完，会：
      - 撑爆 token 预算（max_num_batched_tokens）
      - 导致 waiting 队列中的其他 seq 全部等待
      - 正在 decode 的 seq 被饿死（延迟增大）

    分块 prefill 把长 prompt 切成多个 chunk，分摊到多个 step 中执行，
    这样 decode 不会被一个超长 prefill 完全阻塞。

    限制规则:
      - 只有在 scheduled_seqs 为空时（即本轮第一个 prefill）才允许切分
      - 如果已经有 seq 被调度了，遇到剩余预算不够时直接 break
      - 这保证了至少有一个 seq 被 prefill，同时避免只处理半个 seq 就放弃

================================================================================
五、Round-Robin 公平性
================================================================================

    Decode 阶段：
      running.popleft()          → 取最早进入的（或上一轮最早返回的）
      处理完后:
      running.extendleft(reversed(scheduled_seqs))  → 逆序放回左侧

    逆序机制示意:
      假设顺序调度了 [A, B, C]:
        放回时逆序: C 放最左, B 次之, A 最右
        下次 popleft(): 先取 C（上一轮最后被调度的放到了最前面）

      这其实形成了一个 LIFO 式的轮转：后调度的先被取走，经过足够多轮后趋向均匀。
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    Nano-vLLM 调度器。

    每个推理 step 调用一次 schedule()，返回本轮要处理的序列列表和模式（prefill/decode）。
    schedule() 内部保证每次调用只做一个模式：要么 prefill，要么 decode，不会混合。

    Attributes:
        max_num_seqs:          单步最多处理的序列数（prefill 和 decode 共用上限）。
        max_num_batched_tokens: 单步最多处理的 token 总数（主要约束 prefill；decode 每 seq 只 1 token）。
        eos:                   EOS token id，用于判断序列是否自然结束。
        block_size:            KV-cache block 大小（token 数），传给 BlockManager 和 Sequence。

        block_manager:         KV-cache 块管理器，负责 block 分配/释放/prefix cache。
        waiting:               等待调度的序列队列（新请求、被抢占的、分块 prefill 未完成的）。
        running:               正在参与推理的序列队列（已分配 block_table）。
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size

        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )

        # waiting: 右侧 append 入队，左侧 pop 出队（或 peek [0]）
        self.waiting: deque[Sequence] = deque()
        # running: 左侧 popleft 取 seq，逆序 extendleft 放回
        self.running: deque[Sequence] = deque()

    # ==================== 状态查询 ====================

    def is_finished(self):
        """两个队列都为空时，表示所有请求处理完毕。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """接收新请求，追加到 waiting 队列尾部。FIFO 顺序，先来的先调度。"""
        self.waiting.append(seq)

    # ==================== 核心调度 ====================

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        每步推理前调用。Prefill 优先，没有 prefill 时才做 decode。
        返回 (scheduled_seqs, is_prefill)：本轮处理的序列列表 + 模式标志。
        max_num_batched_tokens 是一次 batch 最多能处理的 token 总数。
        num_batched_tokens 累计已经放入该 batch 的 token 数。
        remaining 就是还能往 batch 里追加的 token 数。如果 remaining == 0，预算花光，直接结束调度。
        num_tokens 就是本轮还需处理的 token 数（从断点继续）。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # ==================== Prefill ====================
        # 遍历 waiting 队首（peek，不 pop）：检查 block 资源 → 分配 block_table →
        # 设置 num_scheduled_tokens（允许分块）→ 处理完所有 prompt token 后移入 running。
        # 退出条件：waiting 空 / 达 max_num_seqs / token 预算耗尽 / block 不够 / 分块限制。
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            if not seq.block_table:
                # 首次分配（新请求 or 抢占后重试）：查 prefix cache + 检查空闲 block
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 分块 prefill 续处理：block_table 已在上一轮分配，从断点继续
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # 分块限制：只有本轮第一个 seq 允许切分，否则遇到预算不够就 break
            if remaining < num_tokens and scheduled_seqs:
                break
            '''
            设计意图：每个推理 step 中，最多只允许一个序列被“切分”（即只处理它的一部分 token），
            其余序列必须完整 prefilled 或者根本不放。
            这样可以避免多个序列同时处于“半截”状态，简化 KV cache 管理和调度状态机。
            如果当前序列恰好是第一个被考虑的请求（scheduled_seqs 为空），则即使 remaining < num_tokens 也会被调度，
            成为那个被切分的序列。
            '''

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            # 如果是第一个序列，那么会分块，后续的序列都不会被chunked
            # 如果是chunked的序列，这里调度的应该是remaining
            # 取 num_tokens（真正需要的）和 remaining（预算允许的）的最小值，避免超预算。
            num_batched_tokens += seq.num_scheduled_tokens

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # 分块未完 → 留在 waiting，下次继续

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # ==================== Decode ====================
        # 每个 seq 只处理 1 token。block 不够时抢占 running 队尾（最少调度的 seq）。
        # 调度完逆序放回，配合 popleft 实现 round-robin。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()

            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs, (
            "Scheduler 死锁：prefill 和 decode 都没有调度任何 seq。"
            "可能原因：waiting 非空但 block 资源不足，且 running 为空无法释放。"
        )

        # 逆序放回：[A,B,C] → extendleft([C,B,A]) → 下轮 popleft 先取 C
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # ==================== 抢占 ====================

    def preempt(self, seq: Sequence):
        """
        抢占一个序列：释放 KV-cache → 重置状态 → 放回 waiting 头部。

        什么时候发生？
          decode 阶段 can_append(seq) 返回 False（当前 block 已满且空闲池耗尽），
          必须释放一些 block 才能继续。优先抢占 running 队尾的 seq，如果没有其他 seq
          则抢占自己。

        抢占后会发生什么？
          被抢占的 seq 丢失所有 KV-cache，下次被调度时需要重新走 prefill 路径
          重算全部 KV。但因为 prompt token_ids 还在（CPU 端），重算成本可控。

        为什么放回 waiting 头部（appendleft）而不是尾部？
          被抢占的 seq 已经等待过一轮了，让它优先重试可以减少整体延迟。

          当 KV-cache block 资源耗尽，导致 decode 无法继续分配新 block 时，调度器通过 preempt 释放已有 sequence 的 KV-cache，
          并将其放回 waiting 队列头部重新进入 prefill，从而用计算换取有限显存资源。

          decode 过程中生成的 response token 会保留在 CPU 的 sequence 里，
          但 GPU 上用于继续生成的 KV-cache 会被抢占清空，因此恢复时必须用完整 token 序列重新做 prefill 来重建计算状态，而不是重新生成文本本身。
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True                  # 重新走 prefill 路径（重算 KV-cache）
        self.block_manager.deallocate(seq)     # 释放所有 block
        self.waiting.appendleft(seq)           # 插入 waiting 头部，优先重试

    # ==================== 后处理 ====================

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ):
        """
        模型推理完成后的后处理。每个 step 调用一次（在 model_runner.run() 之后）。

        处理流程（对每个被调度的 seq）:
          ┌──────────────────────────────────────────────────────┐
          │ 1. hash_blocks(seq)                                 │
          │    将本步新写满的 block 注册到 prefix cache           │
          │    → 后续 seq 可以通过 hash_to_block_id 找到并复用   │
          │                                                      │
          │ 2. 更新 num_cached_tokens += num_scheduled_tokens    │
          │    标记这些 token 的 KV 已写入 block                 │
          │                                                      │
          │ 3. 分块 prefill 中途？                               │
          │    ├─ 是 → 跳过 append（prompt 还没处理完）           │
          │    └─ 否 → append_token(token_id)                    │
          │                                                      │
          │ 4. 检查终止条件                                       │
          │    ├─ 命中 EOS 或达到 max_tokens → FINISHED           │
          │    │     ├─ deallocate(seq) 释放 block               │
          │    │     └─ 从 running 移除                           │
          │    └─ 否则 → 继续参与下轮调度                         │
          └──────────────────────────────────────────────────────┘

        Args:
            seqs:       本轮被调度的序列列表（与 schedule() 返回值一致）
            token_ids:  model_runner 采样出的 token id，长度与 seqs 相同
            is_prefill: 本轮是 prefill 还是 decode
        """
        for seq, token_id in zip(seqs, token_ids):
            # --- 步骤 1: 注册新 block 到 prefix cache ---
            # 只注册本次 prefill/decode 中新写满的完整 block。
            # 已经在 prefix cache 中的 block（之前命中的）不会重复注册。
            self.block_manager.hash_blocks(seq)

            # --- 步骤 2: 推进缓存进度 ---
            # num_cached_tokens 记录了"KV 已写入 block 且 hash 已注册"的 token 数。
            # 每次 postprocess 后推进这个指针，下次 schedule 时以此为起点。
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0     # 重置，本步结束

            # --- 步骤 3: 分块 prefill 中途 → 跳过 append ---
            # prefill 阶段如果 num_cached_tokens < num_tokens，说明 prompt 还没
            # 完全处理完（被分块了），此时不应该追加新 token——我们还在"吃"prompt，
            # 还没到"生成"阶段。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            # --- 步骤 4: 追加采样出的 token ---
            # decode 阶段或 prefill 最后一步（prompt 刚好吃完、生成了第一个 token），
            # 把 model_runner 采样出的新 token 追加到序列。
            seq.append_token(token_id)

            # --- 步骤 5: 检查终止条件 ---
            stop_by_eos = (not seq.ignore_eos and token_id == self.eos)
            stop_by_len = (seq.num_completion_tokens == seq.max_tokens)

            if stop_by_eos or stop_by_len:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)   # 释放所有 KV-cache block
                self.running.remove(seq)             # 从 running 队列移除
