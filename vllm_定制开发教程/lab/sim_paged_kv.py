#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【原理模拟器 1】PagedAttention 的 KV Cache 块管理
====================================================
不依赖 GPU、不依赖 vLLM，纯 Python 标准库实现。
它模拟了 vLLM 中 `vllm/v1/core/kv_cache_manager.py` + `block_pool.py` 的核心逻辑：

  1. 显存被切成固定大小的"块"(block)，每块存 BLOCK_SIZE 个 token 的 KV
  2. 每个请求通过"页表"(block_table) 指向自己的块 —— 块不必连续
  3. 相同前缀的请求共享块 (Prefix Caching)，引用计数管理生命周期
  4. 显存不够时请求被拒绝/抢占，而不是浪费预留空间

运行方式：
    python3 lab/sim_paged_kv.py
"""
from collections import deque

BLOCK_SIZE = 16        # 每块能存多少个 token 的 KV（vLLM 默认 16）
TOTAL_BLOCKS = 12      # 模拟"显存"总共只有 12 个块（12*16=192 个 token 的 KV 空间）

# ---------------------------------------------------------------------------
# 颜色输出（可选美化，不影响逻辑）
# ---------------------------------------------------------------------------
def c(s, code): return f"\033[{code}m{s}\033[0m"
def red(s):    return c(s, "31")
def green(s):  return c(s, "32")
def yellow(s): return c(s, "33")
def cyan(s):   return c(s, "36")
def bold(s):   return c(s, "1")


class Block:
    """一块 KV Cache 物理槽位。对应 vLLM 的 `vllm/v1/core/block_pool.py` 中的块对象。"""
    __slots__ = ("block_id", "ref_cnt", "tokens", "full")

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_cnt = 0      # 引用计数：几个请求正在共享这块
        self.tokens = []      # 这块里已装的 token（演示用，真实 vLLM 里是 KV 向量）
        self.full = False     # 装满后才会被 hash 进前缀缓存表

    @property
    def free_slots(self) -> int:
        return BLOCK_SIZE - len(self.tokens)


class BlockPool:
    """物理块池：管理所有块的分配 / 回收 / 前缀哈希表。"""

    def __init__(self, num_blocks: int):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_ids = deque(range(num_blocks))
        # 前缀缓存表: 「前缀 token 元组的 hash -> block_id」
        # 对应 vLLM BlockPool 中的 `self.cached_block_hash_to_block`
        self.hash_to_block: dict[int, int] = {}
        self.hits = 0        # 统计前缀缓存命中次数（教学用）
        self.misses = 0

    def alloc(self) -> Block | None:
        """从池里拿一个空块。拿不到 = 显存不足。"""
        if not self.free_ids:
            return None
        b = self.blocks[self.free_ids.popleft()]
        b.ref_cnt, b.tokens, b.full = 1, [], False
        return b

    def free(self, block_id: int):
        b = self.blocks[block_id]
        b.ref_cnt -= 1
        if b.ref_cnt == 0:                      # 没人用了才真正回收
            b.tokens, b.full = [], False
            self.hash_to_block = {h: i for h, i in self.hash_to_block.items() if i != block_id}
            self.free_ids.appendleft(block_id)

    def lookup_prefix(self, token_prefix: tuple) -> Block | None:
        """前缀缓存查询：这个 token 前缀对应的满块，是否已在池里？"""
        h = hash(token_prefix)
        if h in self.hash_to_block:
            bid = self.hash_to_block[h]
            if self.blocks[bid].full:
                self.blocks[bid].ref_cnt += 1   # 共享！引用计数 +1
                self.hits += 1
                return self.blocks[bid]
        self.misses += 1
        return None

    def register_full_block(self, b: Block, prefix: tuple):
        """块装满后登记进哈希表：key = 从序列开头到本块末尾的完整前缀。
        （真实 vLLM 中每块持有父块的哈希，形成链式哈希，效果相同）"""
        b.full = True
        self.hash_to_block[hash(prefix)] = b.block_id


class Sequence:
    """一个请求的"页表"：逻辑上连续的 token，物理上散落在若干块里。"""

    def __init__(self, req_id: str, pool: BlockPool):
        self.req_id = req_id
        self.pool = pool
        self.block_table: list[Block] = []   # 页表！PagedAttention 的灵魂
        self.num_tokens = 0
        self._all_tokens: list[int] = []     # 已拥有的全部 token（含缓存复用的部分），用于计算前缀哈希

    # ---- 查询：我这个序列的前缀里，有多少 token 已经有现成的块可以复用？----
    def get_computed_blocks(self, tokens: list[int]) -> int:
        """对应 vLLM 的 KVCacheManager.get_computed_blocks()"""
        num_cached_tokens = 0
        num_full_blocks = len(tokens) // BLOCK_SIZE
        for i in range(num_full_blocks):
            prefix = tuple(tokens[: (i + 1) * BLOCK_SIZE])
            blk = self.pool.lookup_prefix(prefix)
            if blk is None:
                break
            self.block_table.append(blk)     # 直接指过去，不算不存！
            num_cached_tokens = (i + 1) * BLOCK_SIZE
        return num_cached_tokens

    # ---- 写入：prefill 时一次性写入一段 prompt token ----
    def append_tokens(self, tokens: list[int]) -> bool:
        """返回 False 表示显存不够，分配失败。"""
        for t in tokens:
            if not self._append_one(t):
                return False
        self.num_tokens += len(tokens)
        return True

    def _append_one(self, token: int) -> bool:
        # 情况 1：当前最后一块还有空位 —— 共享块满了才是 CoW 的触发点
        if self.block_table and self.block_table[-1].free_slots > 0:
            last = self.block_table[-1]
            # 共享的满块不允许直接写（别人也在引用），需要写时复制(Copy-on-Write)。
            # vLLM 中对应 BlockPool.copy_on_write()。
            if last.ref_cnt > 1 and last.full:
                self.pool.free(last.block_id)     # 本请求放弃共享
                new = self.pool.alloc()           # 复制内容到私有块
                if new is None:
                    return False
                new.tokens = list(last.tokens)
                self.block_table[-1] = new
                last = new
            self._all_tokens.append(token)
            last.tokens.append(token)
            if len(last.tokens) == BLOCK_SIZE:
                self.pool.register_full_block(last, prefix=tuple(self._all_tokens))
            return True
        # 情况 2：需要新块
        nb = self.pool.alloc()
        if nb is None:
            return False
        self.block_table.append(nb)
        return self._append_one(token)

    # ---- 释放：请求结束，归还所有块 ----
    def free(self):
        for b in self.block_table:
            self.pool.free(b.block_id)
        self.block_table = []


class KVCacheManager:
    """把 BlockPool 包装成面向请求的管理器（对应 vllm/v1/core/kv_cache_manager.py）"""

    def __init__(self, num_blocks: int):
        self.pool = BlockPool(num_blocks)
        self.seqs: dict[str, Sequence] = {}

    def add_request(self, req_id: str, prompt_tokens: list[int]) -> bool:
        seq = Sequence(req_id, self.pool)
        # 第一步：先查前缀缓存，能复用多少复用多少
        cached = seq.get_computed_blocks(prompt_tokens)
        seq._all_tokens = list(prompt_tokens[:cached])  # 记账：缓存部分也计入前缀
        # 第二步：只给"没被缓存的部分"分配剩余空间
        need = len(prompt_tokens) - cached
        ok = seq.append_tokens(prompt_tokens[cached:])
        if not ok:
            seq.free()
            return False
        self.seqs[req_id] = seq
        print(f"  请求 {bold(req_id)}: prompt {len(prompt_tokens)} tokens -> "
              f"前缀命中 {green(f'{cached} tokens')} (省去计算!), "
              f"新分配页表 {yellow(len(seq.block_table))} 块")
        return True

    def decode_step(self, req_id: str, new_token: int) -> bool:
        """decode 阶段每步只产 1 个 token，大多数时候写入当前块的空位 —— 零分配!"""
        return self.seqs[req_id].append_tokens([new_token])

    def finish(self, req_id: str):
        self.seqs[req_id].free()
        del self.seqs[req_id]

    # ---- 可视化：画出整块"显存"的占用图 ----
    def draw(self, title="当前显存块占用"):
        print(f"\n{bold('═' * 62)}")
        print(bold(f" {title}"))
        print(f"{'═' * 62}")
        owner = {}
        for rid, seq in self.seqs.items():
            for b in seq.block_table:
                owner.setdefault(b.block_id, set()).add(rid)
        for b in self.pool.blocks:
            if b.block_id in owner:
                rids = owner[b.block_id]
                tag = green("共") if len(rids) > 1 else cyan("占")
                detail = f"{'/'.join(sorted(rids)):<12}"
                usage = f"{len(b.tokens):>2}/{BLOCK_SIZE}"
                print(f"  块{b.block_id:>2} │{tag}│ {usage} │ 引用{b.ref_cnt} │ {detail}")
            else:
                print(f"  块{b.block_id:>2} │{red('·')}│ {' 0/%d' % BLOCK_SIZE} │ 空闲")
        print(f"  空闲块: {bold(len(self.pool.free_ids))}/{TOTAL_BLOCKS}   "
              f"前缀缓存命中: {self.pool.hits}  未命中: {self.pool.misses}\n")


# ===========================================================================
# 实验剧本
# ===========================================================================
def main():
    print(bold("\n########## 实验一：相同系统提示词的请求，共享 KV Cache ##########\n"))
    m = KVCacheManager(TOTAL_BLOCKS)

    # 一段 100 token 的系统提示词（用数字代替真实 token，便于观察）
    SYS_PROMPT = list(range(100))
    req_a = SYS_PROMPT + list(range(100, 130))    # A: 系统 + 自己的问题 (130 tokens)
    req_b = SYS_PROMPT + list(range(200, 218))    # B: 同系统 + 另一个问题 (118 tokens)

    print(cyan("[1] 请求 A 到达，做 prefill："))
    m.add_request("A", req_a)
    m.draw("A prefill 之后")

    print(cyan("[2] 请求 B 到达（同样的系统提示词）："))
    m.add_request("B", req_b)
    m.draw("B prefill 之后  ← 观察【共】标记的块：B 直接复用了 A 算好的 KV，一行计算都没做！")

    print(cyan("[3] 两个请求进入 decode，每步各产出 1 个 token："))
    for step in range(3):
        for rid in list(m.seqs):
            m.decode_step(rid, new_token=900 + step)
        print(f"  decode 第 {step + 1} 步完成")
    m.draw("decode 几步之后 ← 注意：只有最后一块在被慢慢填满，其他块不动（这就是免碎片的原因）")

    print(cyan("[4] 请求 A 完成，释放显存："))
    m.finish("A")
    m.draw("A 释放后 ← 共享块还活着（B 还在用，引用计数>0），A 独占的块被回收")

    print(bold("\n########## 实验二：显存不足时会发生什么？ ##########\n"))
    print("在静态预分配方案里（如早年的 None-attention），显存不足只能直接拒绝请求。")
    print("PagedAttention 方案下：新请求分配失败 -> 排队/抢占最旧的请求，块被腾出来。")
    m2 = KVCacheManager(TOTAL_BLOCKS)
    big = list(range(150))          # 一个需要 ~10 块的大请求
    print(cyan("\n[1] 先放入一个 150 tokens 的大请求 X："))
    if m2.add_request("X", big):
        m2.draw("X 之后：显存几乎被吃满")
    print(cyan("[2] 又来了一个 100 tokens 的请求 Y —— 放不下："))
    if not m2.add_request("Y", list(range(100))):
        print(red("  -> Y 分配失败！调度器会让 Y 排队等待，或抢占 X 释放块后重算。"))
        print("     （这正是 vLLM 调度器里 preemption: RECOMPUTE 策略的动机，详见教程第 05 章）")
    print(cyan("[3] X 完成，Y 重新被调度进来："))
    m2.finish("X")
    m2.add_request("Y", list(range(100)))
    m2.draw("Y 成功进入")


if __name__ == "__main__":
    main()
