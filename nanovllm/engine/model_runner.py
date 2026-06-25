import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    GPU 端推理执行器，是 Nano-vLLM 中与 GPU 交互的唯一入口。

    每个 TP rank 运行一个 ModelRunner 实例:
    - rank 0 (主进程):  拥有模型、KV-cache、CUDA graph，负责采样并返回 token_ids
    - rank 1..N-1 (子进程): 拥有相同的模型和 KV-cache 副本，通过共享内存接收命令执行

    核心职责:
    1. 模型生命周期管理: 加载权重、预热、退出清理
    2. KV-cache 管理: 根据 GPU 显存动态分配 block，绑定到各 Attention 层
    3. 推理执行: prefill (ragged layout) 和 decode (CUDA graph 加速)
    4. Tensor Parallel 协调: 通过 SharedMemory + Event 同步所有 rank
    5. CUDA Graph: 捕获 decode 阶段的计算图，减少 kernel launch overhead

    TP 通信机制 (SharedMemory):
        ┌─────────┐  write_shm()   ┌───────────────┐  Event.set()  ┌─────────┐
        │ Rank 0  │ ──────────────→ │ Shared Memory │ ─────────────→│ Rank 1  │
        │ (main)  │                 │  (pickle)     │               │ (worker)│
        └─────────┘                 └───────────────┘               └─────────┘
             │                                                            │
             │  call() 本地执行                                     read_shm()
             │  相同的方法和参数                                  执行相同的方法
             ↓                                                            ↓
        [模型前向]                                                  [模型前向]
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager            # True = 禁用 CUDA graph
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # ---- 初始化分布式环境 ----
        # 所有 rank 连接同一个 TCP 地址，NCCL 负责 GPU 间通信（all_reduce、gather）
        dist.init_process_group("nccl", "tcp://localhost:2333",
                                world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)                          # 每个 rank 绑定一块 GPU

        # 保存默认设置，加载模型时切换到模型配置的 dtype
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        # ---- 模型与采样器 ----
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)                 # 从 safetensors 加载权重（TP 感知分片）
        self.sampler = Sampler()

        # ---- 初始化流程 ----
        self.warmup_model()                                  # 预热：触发 CUDA kernel 编译
        self.allocate_kv_cache()                             # 根据显存分配 KV-cache
        if not self.enforce_eager:
            self.capture_cudagraph()                         # 捕获 CUDA graph 用于 decode 加速

        # 恢复默认设备为 CPU（后续 Python 操作在 CPU，模型调用在 GPU）
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # ---- TP Worker 初始化 ----
        if self.world_size > 1:
            if rank == 0:
                # 主进程: 创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  # 1MB 共享内存
                dist.barrier()                               # 等待所有 rank 就绪
            else:
                # 子进程: 等待共享内存创建后打开
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()                                  # 进入无限循环等待命令

    def exit(self):
        """优雅退出：关闭共享内存，清理 CUDA graph，销毁进程组"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()                                   # 同步：确保所有 rank 已关闭 shm
            if self.rank == 0:
                self.shm.unlink()                            # 只有创建者才能删除共享内存
        if not self.enforce_eager:
            del self.graphs, self.graph_pool                 # 释放 CUDA graph 资源
        torch.cuda.synchronize()                              # 等待所有 GPU 操作完成
        dist.destroy_process_group()

    # ==================== TP 进程间通信 ====================

    def loop(self):
        """
        Worker 进程的主循环，阻塞等待主进程指令直到收到 'exit'。

        协议: 主进程通过 write_shm (pickle 序列化) → Event.set() 发送 (method_name, *args)。
              Worker 读取后本地执行同名方法。
              所有 rank 执行相同的方法调用，只是 rank 0 额外负责返回采样结果。
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)                    # worker 执行
            if method_name == "exit":
                break

    def read_shm(self):
        """Worker: 等待 Event → 读取共享内存 → 反序列化 → 清除 Event"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()                                    # 阻塞等待主进程通知
        n = int.from_bytes(self.shm.buf[0:4], "little")      # 前 4 字节: 数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化
        self.event.clear()                                   # 清除信号，准备下一次
        return method_name, args

    def write_shm(self, method_name, *args):
        """主进程: 序列化方法调用 → 写入共享内存 → 通知所有 worker"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")          # 写入长度
        self.shm.buf[4:n+4] = data                           # 写入数据
        for event in self.event:
            event.set()                                      # 通知所有 worker

    def call(self, method_name, *args):
        """
        RPC 风格的方法调用，确保所有 rank 执行相同操作。

        - rank 0:  先通过 write_shm 通知所有 worker，然后本地执行
        - rank >0:  直接本地执行（在 loop 中被调用，主进程已通过 shm 同步）
        - TP = 1:   直接本地执行，无需 shm
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # ==================== 初始化阶段 ====================

    def warmup_model(self):
        """
        模型预热：用最大可能的 batch 跑一次 prefill，触发所有 CUDA kernel 的 JIT 编译。

        为什么需要预热:
        - Triton kernel 和 PyTorch CUDA 操作在首次调用时需要编译/初始化
        - 预热后的显存统计更准确（peak vs current），方便后续 KV-cache 分配
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        # 创建虚假序列，token 全是 0
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)                                 # 跑一次 prefill
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        根据 GPU 显存动态分配 KV-cache。

        计算逻辑:
            可用显存 = total * gpu_memory_utilization - 已使用的显存
            模型静态占用 = used - peak + current（模型权重和激活占用的固定部分）
            KV-cache 可用 = total * util - used + peak - current

            每个 block 的字节数 = 2(K+V) × n_layers × block_size × n_kv_heads × head_dim × dtype_size
            num_blocks = KV-cache 可用 // 每个 block 的字节数

        分配后，将 kv_cache tensor 的各层切片绑定到对应 Attention 模块的 k_cache/v_cache 属性，
        这样 Attention.forward() 就可以直接操作预分配的缓存。
        """
        config = self.config
        hf_config = config.hf_config

        # 获取显存信息
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]    # 预热期间的峰值分配
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前分配（模型权重等）

        # 计算 TP 后的 KV head 数和每层 KV-cache 大小
        # ❗它“至少假设/启用了 TP”，但不排除系统里还有其他并行（只是这里没体现）。
        # 每个进程都运行部分的TP
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim",
                           hf_config.hidden_size // hf_config.num_attention_heads)
        # block 字节数: 2(存K+V) × n_layers × block_size × n_kv_heads × head_dim × 元素字节数
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size *
                       num_kv_heads * head_dim * hf_config.dtype.itemsize)

        # 动态计算可分配的 block 数
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // block_bytes
        assert config.num_kvcache_blocks > 0                  # 必须至少能分配 1 个 block

        # 分配 KV-cache: shape = (2, n_layers, n_blocks, block_size, n_kv_heads, head_dim)
        # 维度 0: [0]=K, [1]=V
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers,
                                     config.num_kvcache_blocks, self.block_size,
                                     num_kv_heads, head_dim)

        # 将 KV-cache 切片绑定到每层 Attention 模块
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # 第 layer_id 层的 K cache
                module.v_cache = self.kv_cache[1, layer_id]  # 第 layer_id 层的 V cache
                layer_id += 1
            # 对应分配的是attention模块那块的k_cache 还有v_cache。
    # ==================== 输入准备 ====================

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        将各序列不等长的 block_table 对齐为 GPU tensor。

        block_tables 的形状: (batch_size, max_block_table_len)
        用 -1 填充较短的序列（FlashAttention 将 -1 视为无效 block）。

        1. prepare_block_tables：KV读取索引对齐
        🧠 它做什么？

        把每个 sequence 的 block_table（不等长）
        👉 padding 成 GPU batch tensor

        📥 输入
        seqs = [
          [3, 7, 10],
          [2, 5]
        ]
        ⚙️ 处理
        pad → -1

        变成：

        [
          [3, 7, 10],
          [2, 5, -1]
        ]
        📤 输出
        Tensor(batch_size, max_blocks)

        GPU tensor

        🧠 本质作用

        ❗ 给 FlashAttention / kernel 一个“统一的 KV block lookup 表”

        🔥 关键点
        block_table = “读 KV cache 的索引”
        -1 = invalid block（忽略）
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table))
                        for seq in seqs]
        # pin_memory 加速 CPU→GPU 传输
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        构建 prefill 的 ragged (可变长度) 输入布局。

        Prefill 使用 FlashAttention 的 varlen 模式，将多个序列的 token 拼接成一个 batch:
          input_ids  = [seq0_tok0, seq0_tok1, ..., seq1_tok0, seq1_tok1, ...]
          positions  = [0, 1, ..., 0, 1, ...]（考虑缓存偏移）

        cu_seqlens (cumulative sequence lengths):
          cu_seqlens_q = [0, len_q(seq0), len_q(seq0)+len_q(seq1), ...]  未缓存部分
          cu_seqlens_k = [0, len_k(seq0), len_k(seq0)+len_k(seq1), ...]  包含缓存

        slot_mapping 将每个 token 映射到 KV-cache 中的物理位置:
          slot = block_table[block_idx] * block_size + offset_in_block

        如果有 prefix cache 命中 (cu_seqlens_k > cu_seqlens_q)，
        需要准备 block_tables 让 FlashAttention 从 cache 中读取 K/V。

        返回:
            input_ids, positions: GPU tensor
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]                                   # query 累积长度
        cu_seqlens_k = [0]                                   # key 累积长度（包含缓存）
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            start = seq.num_cached_tokens                    # 从缓存之后的位置开始
            seqlen_q = seq.num_scheduled_tokens              # 本步处理的 query token 数
            end = start + seqlen_q
            seqlen_k = end                                   # key 序列长度 = 缓存 + 新 token

            # 收集 token_ids 和 positions
            input_ids.extend(seq[start:end])                #
            positions.extend(range(start, end))

            # 累积序列长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:                          # 预热阶段，无 block_table
                continue

            # 构建 slot_mapping: token → KV-cache 物理地址
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size    # 第一个 block 可能有偏移
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        # 有 prefix cache 命中 → 需要 block_tables 从 cache 读 K/V
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # CPU pin_memory → GPU 异步传输
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 设置全局上下文，供 Attention 等层读取
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        构建 decode 的输入布局。

        Decode 每个序列只处理 1 个 token (last_token):
          input_ids  = [last_token(seq0), last_token(seq1), ...]
          positions  = [len(seq0)-1, len(seq1)-1, ...]
          context_lens = [len(seq0), len(seq1), ...]         用于 FlashAttention 的 cache_seqlens

        slot_mapping 指向每个序列最后一个 block 中该 token 应写入的位置。

        返回:
            input_ids, positions: GPU tensor
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 最后一个 token 应写入 KV-cache 的位置
            slot_mapping.append(seq.block_table[-1] * self.block_size +
                                seq.last_block_num_tokens - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """收集各序列的 temperature 为 GPU tensor，仅 rank 0 使用（只有 rank 0 做采样）"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # ==================== 模型执行 ====================

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型前向计算，返回 logits。

        三种执行路径:
        1. Prefill:        直接调用 model(input_ids, positions) → compute_logits
        2. Eager Decode:   同样直接调用（enforce_eager=True 或 batch > 512）
        3. CUDA Graph Decode: 将输入填充到预捕获的 graph_vars，replay CUDA graph

        CUDA Graph 加速原理:
        - 正常执行: 每次都要 launch 几十个 CUDA kernel，CPU 侧 launch overhead 大
        - CUDA Graph: 把整个 forward 的 kernel launch 序列录制下来，replay 时一次提交
        - 限制: graph 形状固定，只能处理 ≤ 捕获 batch size 的请求
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 直接执行
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph replay
            bs = input_ids.size(0)
            # 选择 ≥ bs 的最小捕获 batch size
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            # 从全局上下文获取当前 batch 的元数据（slot_mapping, context_lens, block_tables）
            context = get_context()

            # 填充输入到预分配的张量中
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)             # 超出 bs 的位置填充 -1
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            graph.replay()                                   # 一次提交所有 kernel
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        完整的推理步骤：准备输入 → 模型前向 → 采样 → 返回 token_ids。

        这是 model_runner 对外的核心接口，被 LLMEngine.step() 调用。
        在 TP 场景下所有 rank 同时执行，但只有 rank 0 执行采样和返回结果。
        """
        # 准备输入（所有 rank 执行相同操作）
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)

        # 采样参数只有 rank 0 需要
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 模型前向（所有 rank 执行，TP 内部通过 all_reduce/gather 通信）
        logits = self.run_model(input_ids, positions, is_prefill)

        # 采样: 只在 rank 0 上执行（logits 已被 gather 到 rank 0）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        # 清理全局上下文，避免状态泄露
        reset_context()
        return token_ids

    # ==================== CUDA Graph 捕获 ====================

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获多个 batch size 的 CUDA graph 用于 decode 加速。

        捕获的 batch sizes: [1, 2, 4, 8, 16, 32, ..., max_bs]
        (小 batch 以 2 倍递增，大 batch 以 16 递增)

        每个 graph 录制完整的 model.forward() 调用。

        graph_pool: 所有 graph 共享同一个内存池，避免重复分配。
        graph_vars: 预分配的输入/输出张量，replay 时填充实际数据。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)          # 最多捕获到 512
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 预分配所有 graph 共享的张量
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # batch size 序列: 1, 2, 4, 8, 16, 32, 48, ..., max_bs
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):                   # 从大到小捕获（大 pool 先创建）
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs],
                        context_lens=context_lens[:bs],
                        block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup（触发 kernel 编译）
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # 录制
            if self.graph_pool is None:
                self.graph_pool = graph.pool()               # 从第一个 graph 获取共享内存池
            self.graphs[bs] = graph
            torch.cuda.synchronize()                           # 确保录制完成
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
