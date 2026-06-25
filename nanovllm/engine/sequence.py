from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列的三种生命周期状态"""
    WAITING = auto()   # 等待调度，尚未获得 block 资源
    RUNNING = auto()   # 正在运行，已分配 block_table，参与 prefill 或 decode
    FINISHED = auto()  # 已完成，遇到 EOS 或达到 max_tokens，将被移出调度


class Sequence:
    """
    表示一个生成请求的完整生命周期。

    每个 Sequence 会被赋予全局唯一的 seq_id，并经历 WAITING → RUNNING → FINISHED 三态流转。

    核心属性:
        - token_ids:   完整的 token 序列（prompt + 已生成的 completion）
        - block_table: KV-cache 的物理 block 索引列表，由 BlockManager 管理
        - num_cached_tokens:  已写入 KV-cache 且被 hash 记录的 token 数
        - num_scheduled_tokens: 本步调度的 token 数（prefill 可能>1，decode 固定=1）

    跨进程传输:
        为支持 tensor parallelism 的多进程架构，自定义了 __getstate__/__setstate__。
        序列化时只传关键状态而非完整 token_ids，减小 IPC 开销：
        - prefill 阶段传完整 token_ids（worker 需要做 prefix cache 查找）
        - decode 阶段只传 last_token（单个整数）
    """

    block_size = 256      # 类变量：每个 KV-cache block 包含的 token 数，由 Config 在初始化时设置
    counter = count()     # 全局自增计数器，为每个 Sequence 分配唯一的 seq_id

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # ---- 标识与状态 ----
        self.seq_id = next(Sequence.counter)   # 全局唯一序列 ID
        self.status = SequenceStatus.WAITING   # 初始为等待状态

        # ---- token 数据 ----
        self.token_ids = copy(token_ids)       # 深拷贝，避免外部修改影响内部状态
        self.last_token = token_ids[-1]        # 缓存最后一个 token，decode 序列化时只需传这一个值

        # ---- 计数 ----
        self.num_tokens = len(self.token_ids)          # 当前总 token 数（prompt + 已生成）
        self.num_prompt_tokens = len(token_ids)        # prompt 的原始长度，用于计算 completion 长度
        self.num_cached_tokens = 0                     # 已被 prefix cache 命中的 token 数（单位: token）
        self.num_scheduled_tokens = 0                  # 本步被调度处理的 token 数

        # ---- 运行标志 ----
        self.is_prefill = True                         # 下一个 step 是否走 prefill（刚初始化或抢占后为 True）

        # ---- KV-cache 资源 ----
        self.block_table = []                          # 物理 block ID 列表，由 BlockManager 分配

        # ---- 采样参数（从 SamplingParams 提取，跨进程传输时不需要完整对象）----
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # ==================== 序列操作 ====================

    def __len__(self):
        """返回当前总 token 数"""
        return self.num_tokens

    def __getitem__(self, key):
        """按索引访问 token_ids，支持切片"""
        return self.token_ids[key]

    def append_token(self, token_id: int):
        """追加一个生成的 token（decode 每步调用一次）"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # ==================== block 操作 ====================

    @property
    def num_blocks(self):
        """当前 token_ids 占用的 block 总数（向上取整）"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个 block 中的 token 数，用于计算 slot_mapping 的偏移"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """返回第 i 个 block 对应的 token_ids 切片 [i*block_size : (i+1)*block_size)"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    # ==================== 派生属性 ====================

    @property
    def is_finished(self):
        """序列是否已完成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 token 数 = 总 token 数 - prompt 长度"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """prompt 部分的 token_ids"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """已生成部分的 token_ids"""
        return self.token_ids[self.num_prompt_tokens:]

    # ==================== 跨进程序列化 ====================

    def __getstate__(self):
        """
        序列化：多进程传输时由 pickle 调用。

        为减小 IPC 开销，不传输完整的 token_ids：
        - prefill 阶段：传完整列表（主进程需要用 block() 切片、worker 需要做 hash lookup）
        - decode 阶段：只传 last_token（单个整数），主进程自己维护完整 token_ids

        传输的字段:
            num_tokens, num_prompt_tokens, num_cached_tokens, num_scheduled_tokens,
            block_table, last_state (列表或整数)
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """
        反序列化：从主进程接收到的数据中还原 Sequence。

        根据 last_state 的类型判断来源:
        - list:  来自 prefill 阶段，直接恢复 token_ids 和 last_token
        - int:   来自 decode 阶段，token_ids 置空，只恢复 last_token
                 （主进程在调用侧已维护了完整的 token_ids）
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, \
            self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            # prefill: 完整恢复
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode: token_ids 由主进程维护，这里只恢复 last_token
            self.token_ids = []
            self.last_token = last_state
