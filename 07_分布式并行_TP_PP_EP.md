# 第 07 章 · 分布式并行：TP / PP / EP，大模型上多卡的姿势

> 本章目标：搞懂 vLLM 三种并行策略的原理、通信代价与选型方法。
> 学完你能：看懂 `--tensor-parallel-size` 等参数背后的机制、估算通信开销、
> 为给定模型+硬件组合选出正确方案。

---

## 7.0 为什么需要并行？

放不下的两大原因：
1. **权重放不下**：70B 模型 FP16 要 140GB，单卡 80GB 装不下；
2. **KV Cache 放不下**：第 01 章算过，高并发大上下文时 KV 也吃几十 GB。

vLLM 的执行器（`vllm/distributed/` + `vllm/v1/executor/`）支持两种进程编排：
**mp**（多进程，默认，单机多卡）和 **ray**（多机集群）。
每张卡一个 Worker 进程，主 Worker 负责与 EngineCore 通信并广播。

## 7.1 张量并行（TP, Tensor Parallelism）——最常用

**思想**：把"一层"的计算切开，每张卡算一部分，层与层之间同步。

以 Qwen2.5-7B（隐藏维 3584、32 个注意力头）配 2 卡 TP 为例：

```
每张卡放:
  - 权重的一半: 每层 attention/FFN 的矩阵按列/行切成两半 (每头维度 1792, 16 头/卡)
  - KV Cache 的一半: GQA 的 4 个 KV 头分给 2 卡 (各 2 头) → KV 也自动减半!

每个 Transformer 层的通信:
  - attention 输出聚合:  all-reduce (卡间同步相加)
  - FFN 输出聚合:        all-reduce
→ 每层 2 次通信, 28 层 = 每步 56 次 all-reduce
```

**关键认知：**

- TP 通信发生在**每一个前向步骤的每一层**，对**卡间带宽极其敏感**——
  这就是为什么 TP 几乎只能用 NVLink/NVSwitch 同机多卡（PCIe 上 TP=2 性能会明显折损）；
- TP 同时把 KV Cache 也按头数切开了，等于显存压力也减半；
- 规律：TP 尺寸一般取 2 的幂，且 **注意力头数和 FFN 维度必须能被整除**（如 7 头的模型不能用 TP=2）。

## 7.2 流水线并行（PP, Pipeline Parallelism）

**思想**：按**层数**切。28 层的模型用 2 卡 PP：卡 A 管 0~13 层，卡 B 管 14~27 层。

```
批内请求:  [R1 R2 R3 ...] 
卡A: 前半层算 R1 → 传给卡B → 后半层算 R1; 同时卡A 可以开始算 R2 ...
     ↑ "微批(microbatch)"流水, 类似工厂流水线, 让两张卡都有活干
```

- 通信量：每步只在**层边界传一次激活**（比 TP 的每层 2 次 all-reduce 小得多），**跨机（以太网）也能用**；
- 代价：**流水线气泡**（一开始卡 B 空等、批尾卡 A 空等），需要足够并发才能填满；
- vLLM 中 PP 与 TP 可叠加：`--pipeline-parallel-size 2 --tensor-parallel-size 2` = 4 卡。

## 7.3 专家并行（EP, Expert Parallelism）——MoE 模型专属

MoE 模型（DeepSeek-V3、Qwen3-MoE 等）的 FFN 层由上百个"专家"组成，每个 token 只激活少数专家。
**EP 把不同专家放到不同卡**，token 路由到所在专家的卡上计算（all-to-all 通信）。

- vLLM 参数：`--enable-expert-parallel`（常与 TP 组合使用）；
- 注意区分：TP 切"稠密矩阵"，EP 切"专家列表"。

## 7.4 数据并行与 DP Attention（补充）

`--data-parallel-size N`：起 N 份完整的引擎副本，各自服务请求（前置负载均衡），
适合"模型单卡放得下、但吞吐不够"的场景；新版本还支持 **DP Attention**（attention 用 DP、
MoE/FFN 用 TP+EP 的混合切法），是超长上下文 MoE 服务的主流配置，了解即可。

## 7.5 选型决策表

| 场景 | 推荐配置 | 理由 |
|---|---|---|
| 7B，单卡 24GB | 不并行 | 最简单，TP 只增通信不减总时间（单卡场景） |
| 7B，单卡放不下权重+KV | 先量化（第 08 章） | 量化收益最大、零通信代价 |
| 70B，单机 4×A100(NVLink) | `--tensor-parallel-size 4` | 标准答案 |
| 70B，单机 2×A100 | `--tensor-parallel-size 2` + FP8/AWQ 量化 | 双管齐下 |
| 跨多台机器 | `--pipeline-parallel-size 机数 --tensor-parallel-size 每机卡数` + ray | PP 承担跨机慢链路 |
| MoE 大模型（如 671B） | TP×EP 组合（官方推荐命令见模型文档） | 专家切分更高效 |

**黄金法则：能不并行就不并行 → 能量化先量化 → 需要并行时优先 TP（同机）→ 跨机用 PP。**

## 7.6 动手实验

### 🚀 实验 7-1：读懂多卡日志

有多卡机器的话（云上按小时租一张 2×4090 实验即可）：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2
# 观察日志:
#  - "Initializing distributed memory group" (NCCL 组网)
#  - 每个 Worker 各自打印显存摸底 → KV blocks 大约是单卡跑 14B 的一半×2
```

### 🔬 实验 7-2（无多卡也能做）：验证"头数整除"约束

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --tensor-parallel-size 3
# 会直接报错: attention heads(14) 无法被 TP=3 整除 —— 亲手理解 7.1 的约束
```

### 🔧 实验 7-3（纸上推演）：通信量计算

7B 模型（28 层，hidden 3584，FP16）TP=2 时，decode 一步的通信量：
`2 次 all-reduce/层 × 28 层 × 3584 × 2B ≈ 0.4 MB/步`。
改为 TP=4（hidden 仍 3584，消息量翻倍）算一算；再想：如果 hidden 是 8192（70B 模型），通信量又是多少？
（这类估算能力是容量规划的核心，第 09 章还会用到。）

## 7.7 本章小结 + 自测

1. TP 为什么对卡间带宽敏感？PP 为什么能跨机？
2. TP=2 时，GQA 模型的 KV Cache 怎么切？
3. 一个 32 头、28 层的模型，TP 和 PP 的最大可行值各是多少？
4. EP 和 TP 切分的东西分别是什么？
5. "能量化先量化"这条法则的依据是什么？

**上一章**：[第 06 章](06_一次请求的完整旅程_源码走读.md) | **下一章**：[第 08 章 · 高级特性](08_高级特性_投机解码_量化_结构化输出.md)
