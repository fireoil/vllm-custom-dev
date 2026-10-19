# 第 04 章 · PagedAttention 与 KV Cache 管理（vLLM 的灵魂）

> 本章目标：彻底搞懂 vLLM 成名绝技 PagedAttention。
> 学完你能：手算 KV 显存、解释"块/页表/前缀缓存/写时复制"四个概念、
> 并通过模拟器 + 真机实验亲眼验证每一句话。

---

## 4.1 从操作系统借来的智慧

**问题回顾**（第 01 章）：传统引擎给每个请求的 KV Cache 分配"一整块连续显存、按最大长度预留"，导致预留浪费 + 内外碎片，实际利用率常常只有 20~40%。

**操作系统当年也遇到过一模一样的问题**：进程要一大段连续内存，内存很快碎片化。
OS 的答案是**虚拟内存 + 分页**：

- 物理内存切成固定大小的**页**（如 4KB）
- 进程看到的是"连续"虚拟地址，页表把它映射到任意散落的物理页
- 按需分配页，用多少分多少

**vLLM 把这套搬到了 KV Cache 上：**

| 操作系统概念 | vLLM 对应物 | vLLM 里的默认值 |
|---|---|---|
| 物理页 | **块（block）**：存放 16 个 token 的 KV | `block_size=16` |
| 虚拟地址空间 | 请求的逻辑 token 序列 | 逻辑上连续 |
| 页表 | **块表（block_table）**：逻辑块号 → 物理块号 | 每请求一张 |
| 缺页时分配 | 逻辑块写满后才占新物理块 | 按需 |
| 共享内存页 | 相同前缀共享物理块 | 引用计数管理 |
| 写时复制（COW） | 共享块需要写入时复制一份 | fork 后写场景 |

**PagedAttention 内核**就是配合这套分页结构的注意力算子：它不再假设"K/V 在显存中连续排列"，
而是拿着 block_table 逐块取数计算——这就是论文标题 PagedAttention 的由来。

## 4.2 逐帧动画：一个请求的一生（显存视角）

设 `block_size=16`，prompt 有 35 个 token，要生成 12 个：

```
── prefill（第1步，35 个 token 一次算完）──
逻辑块 0:  token[0..15]   → 物理块 #7   (16/16 满)
逻辑块 1:  token[16..31]  → 物理块 #3   (16/16 满)
逻辑块 2:  token[32..34]  → 物理块 #12  ( 3/16 半满)  ← 只浪费了 13 个空位，
                                                       传统方案却要浪费 4061 个！

── decode（每步 +1 token）──
第 1 步: token 35 写进块 #12 → 4/16   (不用分配新块!)
第 2 步: token 36 写进块 #12 → 5/16   (还是不用!)
...
第 12 步: token 46 写进块 #12 → 12/16，请求结束
       → 块 #7 #3 #12 归还，全程只占 3 块，无任何预留

块表(block_table): [7, 3, 12]   ← PagedAttention 内核就靠这 3 个数字找数据
```

**三个收益，一目了然：**

1. **零预留**：只分配实际用到的块
2. **内部碎片 ≤ 1 块**（最多浪费 15 个槽位，而非几千）
3. **免外部碎片**：块是离散的，无需连续空间

## 4.3 前缀缓存（Prefix Caching）：同一个开头，全世界只算一次

现实中大量请求共享前缀：同一个系统提示词、同一个 RAG 文档、同一个 few-shot 模板。
vLLM 的做法：**装满的块按"从头到该块结尾的完整 token 序列哈希"登记进哈希表**
（源码：`vllm/v1/core/block_pool.py` 中 `cached_block_hash_to_block`；块与块之间用父块哈希形成链式哈希，可防篡改地快速匹配任意深度前缀）。

```
请求 A: [系统提示词 100 tok][A 问题]
请求 B: [系统提示词 100 tok][B 问题]   ← B 到达时:

B 的 prompt 前 96 tok (6 块) 哈希命中 → 直接把 A 的 6 个物理块挂进 B 的块表
B 只需重新计算剩下的 4 tok + 自己的问题
   → 省掉了 ~90% 的 prefill 计算和 6 块显存!  TTFT 大幅下降
```

**引用计数与释放**：块的 `ref_cnt` 记录有几个请求在用。请求结束 `ref_cnt-1`，
降到 0 才真正回收。所以"A 结束了，B 还在用共享块"完全安全。

**写时复制（CoW）**：如果共享的前缀块**还没装满**，而 B 要往里写新 token
（往缓存里写会污染 A 的数据！）——vLLM 就复制出一个私有块给 B 写。
这就是"写时复制"，和 OS fork 后写内存页的行为一模一样。
（实战中 prefill 通常按块对齐，CoW 触发不多，但你必须知道它的存在。）

> 📌 **V1 默认已开启前缀缓存**（`enable_prefix_caching=True`），且配合
> `long_prefill_token_threshold` 等参数工作。日志里搜 `Prefix caching` 可见命中统计。

## 4.4 KV Cache 显存是怎么"摸底"出来的？

对应第 03 章启动流程 [4][5]，源码位置：

- `vllm/v1/worker/gpu_worker.py` → `determine_num_available_blocks()`：跑一次 `profile_run()`，测量"模型权重 + 激活 + CUDA 上下文"占了多少；
- 可用 KV 显存 = `总显存 × gpu_memory_utilization − 摸底值`；
- `vllm/v1/core/kv_cache_utils.py`：算出每块多大 → 得到块数 N → **一次性**创建 N 块的 KV 大张量。

**动手算一遍**（以本教程演示机 RTX 4060 8GB、Qwen2.5-0.5B 为例）：

```
总显存:            8.0 GB
× utilization 0.85: 6.8 GB 预算
− 权重 (0.5B fp16): ~1.0 GB
− 激活/上下文:      ~1.0 GB
剩给 KV Cache:     ~4.8 GB
每 token KV (0.5B): 2×24层×2头×128维×2B ≈ 24 KB? （用第01章公式自己算准）
每块 16 token →   ~0.4 MB/块 → 约 12000 块
→ 日志 "# GPU blocks" 应该在 1 万上下
```

> 运行 `vllm serve` 后对照日志验证你的手算。误差 ±20% 正常（中间结果、CUDA Graph 等占额）。

## 4.5 源码走读：核心 3 个类（v0.31.0）

只需要认识这 3 个类的职责和方法（**先读注释和签名，不要逐行抠**）：

```
vllm/v1/core/block_pool.py          # 物理块池
  class BlockPool:
      cached_block_hash_to_block    # 前缀缓存哈希表
      get_cached_block(tokens)      # 查这个前缀有没有现成块
      cache_full_blocks(...)        # 装满的块登记进哈希表
      free(block_id)                # ref_cnt-1, 归零回收
      allocate_block()              # 拿空块

vllm/v1/core/kv_cache_manager.py    # 面向请求的管理层（对 BlockPool 的封装）
  class KVCacheManager:
      get_computed_blocks(request)  # ★ prefill 前查: 多少前缀可复用
      allocate_slots(request, ...)  # ★ 给新 token 分配块, 放不下→None(触发调度器抢占)
      free(request)                 # 请求结束归还

vllm/v1/core/sched/scheduler.py     # 调用方（第05章细讲）
      # schedule() 里调用 allocate_slots；返回 None 则该请求不能进本步
```

**数据流**（背下来）：`Scheduler.schedule() → KVCacheManager.get_computed_blocks() → allocate_slots() → 生成 SchedulerOutput(含每个请求的 block_table) → GPUModelRunner.execute_model() → PagedAttention 内核按 block_table 取 KV 计算`。

## 4.6 动手实验

### 🔧 实验 4-1（零门槛，必做）：跑原理模拟器

```bash
python3 lab/sim_paged_kv.py
```

这个 200 行的纯 Python 模拟器实现了上面 3 个类的简化版。请对照输出做三件事：

1. **实验一**里，请求 B 打印"前缀命中 96 tokens"——找出输出图中 6 个"【共】"标记的块，
   验证 `ref_cnt=2`；
2. 手动把 `BLOCK_SIZE` 改成 8 再跑，观察命中数和块数如何变化（体会块大小的权衡）；
3. 在 `KVCacheManager.add_request` 的 `cached = ...` 一行打断点/加打印，
   确认 B 的页表前 6 项就是 A 的物理块号（**这就是页表共享的铁证**）。

### 🚀 实验 4-2（真机）：前缀缓存加速对比

```bash
python3 lab/real_vllm_prefix_cache.py
```

10 个共享长前缀的请求 vs 10 个前缀各异的请求，对比总耗时与 `num_cached_tokens`。
在 4060 8GB + Qwen2.5-0.5B 上，A 组应显著快于 B 组。

### 🔬 实验 4-3（硬核）：亲眼看见抢占式腾挪

用小显存 + 大上下文逼出调度器的"抢占"行为：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 32768 --gpu-memory-utilization 0.7
# 另开终端，用 openai SDK 并发发 10 个 2 万 token 的请求，同时 tail 日志：
tail -f serve.log | grep -iE "preempt|recompute"
```

日志出现 `Preempted request ... (reason: RECOMPUTE)` 时，结合 4.5 的 `allocate_slots → None`
思考：调度器为什么选择"丢掉一个请求的 KV 重算"而不是"直接拒绝新请求"？（第 05 章揭晓）

## 4.7 本章小结 + 自测

1. block_size=16 意味着内部碎片上限是多少 token？
2. 前缀缓存的哈希 key 是"单个块的内容"还是"从开头到该块的完整前缀"？为什么必须后者？
3. 请求 A 结束时它引用的共享块会被回收吗？
4. vLLM 为什么要在启动时 profile_run？如果不做（直接按理论值分配）会怎样？
5. `allocate_slots` 返回 None 时，调度器有哪些选择？

**下一章**：[第 05 章 · 调度器与连续批处理](05_调度器与连续批处理.md)
