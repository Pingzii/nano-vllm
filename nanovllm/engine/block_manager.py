"""
PagedAttention 块管理器 —— KV-cache 的物理存储与 Prefix Cache 系统。

================================================================================
一、背景：为什么需要 BlockManager？
================================================================================

Transformer 自回归推理中，每个 token 的 Key/Value 向量需要缓存下来供后续 token 做 attention，
这就是 KV-cache。在 vLLM 的 PagedAttention 架构中，KV-cache 不再是一整块连续显存，而是被
切分成固定大小的 "block"（类似操作系统的内存分页）：

  - block_size = 256，即一个 block 存储 256 个 token 的 K/V 向量
  - 每个 Sequence 持有一个 block_table（逻辑块→物理块映射表）
  - 多个 Sequence 可以通过共享物理 block 实现 prefix cache 复用

BlockManager 的职责：
  1. 管理物理 block 池（分配 / 释放 / 引用计数）
  2. 实现 prefix cache 的查找与复用（链式 xxhash）
  3. 支持 prefill 和 decode 两个阶段的 block 操作

================================================================================
二、Prefix Cache 的链式哈希设计
================================================================================

普通的按块独立 hash 有一个致命缺陷：相同的 token 序列出现在不同上下文时 hash 相同。
例如 "the cat" 在 "the cat sat" 和 "I saw the cat" 两个 prompt 中，按块 hash 会命中，
但它们的 KV-cache 实际不同（因为 attention 的因果掩码会看到不同的前缀）。

解决方案：链式哈希 —— 每个 block 的 hash 不仅包含自己的 token_ids，还混入了前一个
block 的 hash 值（即整个前缀的指纹）：

    h_0 = xxhash(block_0_tokens)
    h_1 = xxhash(block_1_tokens, prefix=h_0)
    h_2 = xxhash(block_2_tokens, prefix=h_1)
    ...

这样，"the cat" 在两个不同上下文中的 hash 完全不同，只有完整前缀匹配才会命中。
这也是 compute_hash 方法中 prefix 参数的核心作用。

================================================================================
三、内存管理流程总览
================================================================================

[主进程 Scheduler]
     │
     ├─ can_allocate(seq)        ← 检查是否有足够空闲 block + 计算 prefix cache 命中数
     │    └─ 返回 num_cached_blocks（命中数）或 -1（资源不足）
     │
     ├─ allocate(seq, num_cached) ← 分配完整的 block_table
     │    ├─ 前 num_cached 个: 复用缓存（ref_count++ 或从 free 取回）
     │    └─ 剩余: _allocate_block() 新分配
     │
     ├─ hash_blocks(seq)          ← prefill 完成后注册新 block 到 prefix cache
     │    └─ 链式计算 hash → 更新 Block 对象 → 写入 hash_to_block_id
     │
     ├─ may_append(seq)           ← decode 阶段 token 装满 block 时追加新 block
     │
     └─ deallocate(seq)           ← 序列完成/中止时释放所有 block

================================================================================
四、引用计数与共享语义
================================================================================

ref_count 表示一个物理 block 被多少个 Sequence 同时引用：

  ref_count = 0  → 空闲，在 free_block_ids 中
  ref_count = 1  → 被 1 个 seq 独占使用，在 used_block_ids 中
  ref_count ≥ 2  → 被多个 seq 共享（prefix cache 命中），在 used_block_ids 中

释放时 ref_count 减 1，减到 0 才真正归还给 free 列表。这是典型的"写时复制"之前的
引用追踪 —— 当前实现中 block 一旦写满就不可变，因此共享是安全的。
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV-cache 的物理存储单元。

    每个 Block 对应 GPU 显存中的一段连续空间，存储 block_size 个 token 的 K/V 向量。
    Block 对象本身只记录元数据（hash / token_ids / ref_count），实际 K/V 张量
    在 GPU 侧由 model runner 管理，通过 block_id 索引。

    Attributes:
        block_id:   全局唯一编号，也是 GPU 侧 KV-cache 张量的索引维度。
        ref_count:  引用计数。0=空闲，≥1=被引用数。多个 seq 共享同一前缀时 >1。
        hash:       链式 xxhash 值。初始 -1 表示未计算（新分配的 block）。
                    注意：block 释放时 hash 不会清除，因此 block 在 free 池中时
                    hash 可能是"过期"的——它记录的仍是旧数据，而 hash_to_block_id
                    可能已指向另一个存有相同 token 序列的新 block。
        token_ids:  该 block 对应的原始 token 序列。两个用途：
                    1. can_allocate 中做内容校验——hash 只是快速索引，token_ids 才是
                       判定缓存是否命中的最终依据（hash_to_block_id 指向的 block 可能
                       在 free 池中，其内容不一定和当前 seq 一致）；
                    2. 与 hash 配合完成 prefix cache 的精确匹配。
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0          # 初始空闲
        self.hash = -1              # -1 = 未注册到 prefix cache
        self.token_ids = []         # 空列表 = 无数据

    def update(self, hash: int, token_ids: list[int]):
        """
        prefill 完成后将 block 注册到 prefix cache。

        调用时机：hash_blocks() 对每个新写满/新计算的 block 调用。
        之后该 block 可被其他 seq 通过 hash_to_block_id 查找并共享。
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        从空闲池取出分配时，重置为初始占用状态。

        注意：ref_count 设为 1（被分配者独占），hash 和 token_ids 清空。
        这些值会在 prefill 完成后的 hash_blocks() 中被重新设置。
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    PagedAttention 块管理器。

    管理固定数量的物理 block，负责：
      - 分配与释放（类似 malloc/free）
      - Prefix cache 的查找与复用
      - 引用计数维护

    外部接口（按调用顺序）:
      1. can_allocate(seq)          → 调度前检查资源 + 计算缓存命中数
      2. allocate(seq, cached_num)  → 分配 block_table
      3. hash_blocks(seq)           → prefill 后注册新 block 到缓存
      4. may_append(seq)            → decode 时按需追加 block
      5. can_append(seq)            → decode 前检查是否需要新 block
      6. deallocate(seq)            → 序列结束时释放

    Attributes:
        block_size:        每个 block 的 token 容量（默认 256）。
        blocks:            所有 Block 对象的数组，用 block_id 索引。大小固定，由 GPU 显存决定。
        hash_to_block_id:  链式 hash → block_id 的全局映射。prefix cache 的核心数据结构。
        free_block_ids:    空闲 block 的双端队列，从左侧 popleft 分配，右侧 append 归还。
                           deque 保证 O(1) 的分配/释放。
        used_block_ids:    已占用 block 的集合。用 set 存储以支持 O(1) 的成员检查
                           （can_allocate 中需要判断 block 是否已被使用来决定是否消耗空闲资源）。
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # 预创建所有 Block 对象，block_id 0 ~ num_blocks-1
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 全局 hash 查找表：链式 hash → 物理 block_id
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲 block 池，初始全部空闲
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已占用 block 集合，用于 O(1) 判断 block 是否在用
        self.used_block_ids: set[int] = set()

    # ==================== 哈希计算 ====================

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算一个 block 的链式 xxhash。

        算法:
            1. 如果 prefix != -1，先将 prefix 的 8 字节小端表示写入哈希
            2. 再将当前 block 的 token_ids（转为 numpy bytes）写入哈希
            3. 返回最终的 64 位整数摘要

        链式设计的关键作用：
            假设两个 prompt:
              A: "The cat sat on the mat"  → block_0 = [The, cat, ..., sat]
              B: "I saw the cat on the mat" → block_0 = [I, saw, ..., cat]

            如果没有链式，A.block_1（"on the mat"）和 B.block_1（"on the mat"）
            会有相同的 hash，导致错误命中。但加入 prefix hash 后，A.block_0 和
            B.block_0 的 hash 不同，所以 A.block_1 和 B.block_1 的 hash 也不同。

        Args:
            token_ids: 该 block 的 token ID 序列（长度 ≤ block_size）
            prefix:    前一个 block 的链式 hash，-1 表示这是第一个 block

        Returns:
            64 位 xxhash 整数摘要
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 先将前缀的 hash 值注入——确保"相同 token、不同上下文"产生不同 hash
            h.update(prefix.to_bytes(8, "little"))
        # 再将当前 block 的 token IDs 注入
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # ==================== 底层 block 操作 ====================

    def _allocate_block(self) -> int:
        """
        从空闲池分配一个物理 block。
        流程:
          1. 从 free_block_ids 左侧取出一个空闲 ID（FIFO，利于 LRU 风格的复用）
          2. 如果该 block 之前被 hash 注册过，且全局映射仍然指向它，清除旧映射
          3. reset 该 block：ref_count=1, hash=-1, token_ids=[]
          4. 加入 used_block_ids

        Returns:
            新分配的 block_id

        Precondition:
            free_block_ids 非空（调用方应在 can_allocate 中提前检查）

            这段逻辑的本质是：`hash_to_block_id` 表示“某个 hash 当前最新归属的 block”，是会被后续写入不断覆盖的，
            而 block 从 free pool 取出时只是复用旧对象，它内部的 `hash` 可能仍残留上一次生命周期的值；
            因此当 `_allocate_block` 尝试清理旧映射时，必须判断 `hash_to_block_id.get(block.hash) == block_id`，
            只有当当前映射仍然指向自己时才允许删除，否则说明这个 hash 已经被后续 block 重新绑定（覆盖）过，
            直接删除会误伤新 block 的缓存映射。本质是在解决“对象复用 + 状态残留 + 映射被覆盖”导致的 key 归属不一致问题。

        """
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} is not free (ref_count={block.ref_count})"
        # 安全清除旧的 hash 映射。
        #
        # 只判断 block.hash != -1 是不够的——需要加后半部分来防止误删：
        #
        #   场景: Block 5 存 tokens T，hash=H，释放回 free 池（hash 保留）。
        #         之后 Block 7 被分配，恰好也存相同的 tokens T，计算出相同的 hash=H，
        #         hash_blocks 将 hash_to_block_id[H] 更新为 7（覆盖了 5）。
        #         现在 Block 5 再次被 _allocate_block 取出，block.hash 仍为旧值 H，
        #         但 hash_to_block_id.get(H) = 7 ≠ 5。
        #
        #   - 如果只判断 block.hash != -1：直接 del hash_to_block_id[H]
        #     → Block 7 的映射被误删！Block 7 仍在 used 状态，却从 prefix cache 消失了。
        #
        #   - 加上 hash_to_block_id.get(H) == block_id 后：
        #     7 ≠ 5 → 不删除 → Block 7 的映射安全保留。
        #
        # 总结：这防的不是 hash 碰撞（不同 token → 相同 hash），而是 hash 复用
        # （相同 token → 相同 hash → 不同物理 block → 旧映射被新 block 覆盖）。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """
        将物理 block 归还空闲池。

        注意：这不是公共接口。外部应调用 deallocate(seq)，由 deallocate 在
        ref_count 减到 0 时内部调用本方法。这样可以保证引用计数的正确性。

        Precondition:
            block.ref_count == 0（由调用方 deallocate 保证）
        """
        assert self.blocks[block_id].ref_count == 0, \
            f"Block {block_id} still has ref_count={self.blocks[block_id].ref_count}"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)    # 放回队列右侧

    # ==================== Sequence 级分配与释放 ====================

    def can_allocate(self, seq: Sequence) -> int:
        """
        调度前的资源检查 + prefix cache 查找。

        遍历 seq 的所有完整 block（排除最后一个可能不完整的 block），从前往后
        进行链式 hash 查找。每找到一个完整匹配的 block，缓存命中数 +1。

        缓存命中的两种情形:
          - 命中且 block 在 used_block_ids 中: 直接共享，不消耗空闲资源
            这种情况是正在运行的其他 seq 也引用了同一个前缀 block。
          - 命中但 block 不在 used_block_ids 中: 这个 block 之前被注册到 hash
            映射但已被释放回 free 池。需要从 free 中取出，消耗空闲资源。
            这相当于"冷缓存命中"。

        为什么跳过最后一个 block？
          最后一个 block 的 token 数可能 < block_size（不完整），如果缓存中
          的 block 是满的，token_ids 内容比对会失败；如果缓存中也是不满的，
          两个不满 block 的内容几乎不可能完全一致。跳过可避免无效比对。

        Args:
            seq: 待检查的序列

        Returns:
            ≥0: prefix cache 命中的 block 数量
            -1: 空闲 block 不足，无法分配
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks              # 总共需要多少个物理 block

        for i in range(seq.num_blocks - 1):          # 跳过最后一个不完整 block
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)      # 链式计算当前 block 的 hash
            block_id = self.hash_to_block_id.get(h, -1)

            # hash 只是快速索引，token_ids 才是内容真相。
            #
            # hash_to_block_id 指向的 block 可能在 free 池中（冷缓存），其 token_ids
            # 不一定和当前 seq 一致——比如该 block 之前存的是另一组 token，只是碰巧
            # 链式 hash 相同。所以必须用 token_ids 做最终裁决，不能只看 hash 匹配。
            #
            # 一旦 token_ids 不匹配，缓存链就此断裂——后续 block 的 hash 都依赖
            # 前一个 block 的 hash 值，前一个错了后面全错，直接 break。

            ## cache miss的情况
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break

            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
            # 正在使用的共享 block，不消耗空闲资源，这里面额外需要增肌的block就减小1.
            # else: block 在 free 池中，需要消耗空闲资源，num_new_blocks 不变

        if len(self.free_block_ids) < num_new_blocks:
            return -1                                 # 空闲资源不足
        return num_cached_blocks


    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """
        为 seq 分配完整的 block_table。

        必须紧跟 can_allocate 返回的非负值调用，num_cached_blocks 就是
        can_allocate 的返回值。

        分配分两段:
          第一段（block 0 ~ num_cached_blocks-1）: 复用缓存命中的 block
            - 如果 block 已在 used_block_ids 中（其他 seq 正在用），只需 ref_count++
              这是纯引用计数操作，不消耗空闲资源。
            - 如果 block 不在 used_block_ids 中（在 free 池里），设置 ref_count=1，
              从 free_block_ids 中移除并加入 used_block_ids。这相当于从空闲池"认领"
              一个已有数据的 block。

          第二段（block num_cached_blocks ~ num_blocks-1）: 新分配
            - 调用 _allocate_block() 从空闲池获取空白 block。
            - 这些 block 将在 prefill 后被 hash_blocks() 注册到缓存。

        Args:
            seq:               待分配资源的序列
            num_cached_blocks: prefix cache 命中的 block 数

        Precondition:
            seq.block_table 为空（序列是首次分配）
            num_cached_blocks ≤ seq.num_blocks
            free_block_ids 数量 ≥ seq.num_blocks - num_cached_blocks（由 can_allocate 保证）
        """
        assert not seq.block_table, "Sequence already has a block_table"
        h = -1

        # === 第一段：复用缓存命中的 block ===
        ## 缓存的block一般都在开头
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]          # can_allocate 已验证存在
            block = self.blocks[block_id]

            if block_id in self.used_block_ids:
                # 情况 A: 其他 seq 正在使用，共享之——只需引用计数 +1
                block.ref_count += 1
            else:
                # 情况 B: block 在 free 池中（冷缓存命中），从 free 中取回
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)

            seq.block_table.append(block_id)

        # === 第二段：为剩余 block 分配新空间 ===
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())

        # 记录已缓存的 token 数量，供 hash_blocks 和 scheduler 使用
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """
        释放 seq 持有的所有物理 block。

        核心逻辑：遍历 block_table，对每个 block 的 ref_count 减 1。
        减到 0 时归还给空闲池。

        为什么倒序遍历？
          通常情况下，只有最后一个 block 是独占的（其他 seq 共享前缀只到前面
          几个 block）。倒序从尾部释放可以更快地将独占 block 归还给空闲池，
          而前缀共享 block 只是 ref_count 减 1 不会归还。

        设计细节：
          - block 的 hash / token_ids 不会清除。即使 ref_count=0 回到 free 池，
            它仍然保留旧的 hash 映射。这允许后续的 can_allocate 通过 hash 找到它
            （冷缓存命中场景）。只有当该 block 被 _allocate_block 重新分配时，
            旧的 hash 映射才会被清除。
          - seq.block_table 被清空，num_cached_tokens 重置为 0。

        Args:
            seq: 需要释放资源的序列（通常状态为 FINISHED）
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        ## 在 KV cache 系统里，共享只存在于前缀对齐区域，因此越靠后的 KV block 越“不可共享”，
        # 表现上看起来像是“更独特”，但本质是共享机会递减。

    # ==================== Decode 阶段的 block 追加 ====================

    def can_append(self, seq: Sequence) -> bool:
        """
        判断 decode 阶段能否为 seq 追加一个 token。

        Decode 阶段每步生成 1 个 token。大多数情况下这个 token 写入当前 block 的
        未使用位置即可，不需要新的物理 block。只有当 token 数刚好在 block 边界上时
        （即当前 block 已满），才需要分配一个新 block。

        判断条件 len(seq) % block_size == 1:
          假设 block_size=256，seq 当前有 256 个 token。新 token 是第 257 个，
          256 % 256 = 0 ≠ 1，不需要新 block（还有空位）。
          当 seq 有 257 个 token（即刚跨过边界，用掉了新 block 的第一个位置），
          257 % 256 = 1，下一次 decode 需要新 block。

        实际上这个条件等价于"上一个 token 用掉了当前 block 的最后一个空位"，
        即当前 block 已满。写法简洁但含义需要仔细理解。

        Returns:
            True:  有足够空闲 block（或不需要新 block）
            False: 需要新 block 但空闲池已空
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        Decode 阶段，如果当前 block 已满，追加一个新 block。

        调用时机：每次 decode step 后，在写入 KV-cache 之前调用。
        如果当前 token 数刚好跨过 block 边界（len(seq) % block_size == 1），
        说明当前 block 的最后空位已被上一轮的 token 占用，本轮需要新 block。

        为什么叫 may_append 而不是 append？
          大多数 decode step 不需要新 block，本方法内部有条件判断，"may"表示可能操作。

        条件详解:
          假设 block_size=256:
            - seq.num_tokens=256: 256%256=0≠1 → 不追加（当前 block 刚满，但新 token
              写入前才需要新 block，这里是在 token 写入后调用的）
            - seq.num_tokens=257: 257%256=1 → 追加新 block（上一轮用掉了当前 block
              的最后空位，为新 token 准备 block）
        """
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    # ==================== Prefix Cache 注册 ====================

    def hash_blocks(self, seq: Sequence):
        """
        Prefill 完成后，将新计算的 block 注册到 prefix cache。

        只处理本次 prefill 新写入的完整 block（跳过已被缓存命中的前缀 block，
        跳过 prefill 未完成的最后一个不完整 block）。

        算法:
          1. 计算 start: 第一个需要注册的 block 索引
             = num_cached_tokens // block_size（prefill 前已缓存的 block 数）

          2. 计算 end: 最后一个已完成的 block 索引
             = (num_cached_tokens + num_scheduled_tokens) // block_size

          3. 如果 start == end，说明本次 prefill 没有写满任何一个完整的新 block
             （例如只处理了几个 token，不足一个 block），直接返回。

          4. 对于 [start, end) 范围内的每个 block:
             a. 获取前一个 block 的链式 hash 作为 prefix
             b. 链式计算当前 block 的 hash
             c. 更新 Block 对象的 hash 和 token_ids 元数据
             d. 注册到全局 hash_to_block_id 映射

        注册后，后续其他 seq 的 can_allocate 就能通过 hash 找到这些 block，
        实现 prefix cache 共享。这是整个 prefix cache 系统的"写入端"。

        示例:
          block_size=256, num_cached_tokens=500, num_scheduled_tokens=300
          start = 500 // 256 = 1    （block 0 已全部缓存，block 1 部分缓存）
          end = 800 // 256 = 3      （block 1 和 2 完成，block 3 可能未完成）
          需要注册 block 1 和 block 2（i=1 和 i=2）

        Args:
            seq: 刚完成 prefill 的序列
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return    # 没有完整的新 block 需要注册

        # 获取 prefix hashing 的起始 hash：
        # 如果 start>0，用前一个已经注册过的 block 的 hash 作为链起点
        # 如果 start==0，prefix=-1，表示这是序列的第一个 block
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1

        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            # 链式计算: hash = xxhash(前一个block的hash || 当前block的token_ids)
            h = self.compute_hash(token_ids, h)
            # 将计算结果写入 Block 元数据并注册到全局映射
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
