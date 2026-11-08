# 第 04 章 · KV Cache 与 PagedAttention（vLLM 的灵魂）

> 本章目标：**彻底**搞懂 KV Cache 的原理——文与公式一个不漏。这是理解 vLLM 一切设计的起点；
> 然后搞懂 vLLM 成名绝技 PagedAttention 是如何管理它的。
> 学完你能：手推"为什么只缓存 K/V"、手算任何模型的 KV 显存、解释"块/页表/前缀缓存/写时复制"、
> 并通过模拟器 + 真机实验亲眼验证每一句话。
> （所有文件路径均对照 vLLM **v0.31.0** 真实源码树核对）

---

# 第一部分 · KV Cache 原理彻底研究

## 4.1 一切的起点：注意力的数学

回顾 Transformer 单层自注意力。设当前序列有 $n$ 个 token，每个 token 的隐藏向量
$x_i \in \mathbb{R}^{d_{model}}$，把 $n$ 个按行排成矩阵 $X \in \mathbb{R}^{n \times d_{model}}$，则：

$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
$$

其中 $Q, K, V \in \mathbb{R}^{n \times d_{model}}$。切成 $H$ 个头（每头
$d_{head} = d_{model}/H$ 维）后，每个头独立做"打分—加权"，并加**因果掩码**
（位置 $i$ 只能看 $j \le i$，不能偷看未来）：

$$
A_h = \mathrm{softmax}\!\left( \frac{Q_h K_h^{\top}}{\sqrt{d_{head}}} + M \right) V_h,
\qquad h = 1, \dots, H \tag{1}
$$

$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
\quad \text{（softmax 后未来位置权重严格为 0）}
$$

最后拼接所有头并投影：$O = \mathrm{Concat}(A_1, \dots, A_H)\, W_O$。

解码第 $t$ 步（生成第 $t$ 个 token）的完整计算，用公式写出来只有四行：

$$
\begin{aligned}
K_t &= x_t W_K, \quad V_t = x_t W_V && \text{（只对当前 token 算）} \\
Q_t &= x_t W_Q && \text{（同上，只算这一个）} \\
\text{K缓存} &\leftarrow [\text{K缓存} \mid K_t], \quad
\text{V缓存} \leftarrow [\text{V缓存} \mid V_t] && \text{（追加进缓存）} \\
o_t &= \mathrm{softmax}\!\left( \frac{Q_t\, [K_1 \dots K_t]^{\top}}{\sqrt{d_{head}}} \right)
[V_1 \dots V_t] && \text{（1 行 Q 对 t 行缓存 KV）}
\end{aligned}
$$

**读懂这四行公式，KV Cache 就懂了一半**：第 $t$ 步的全部计算，只涉及
"1 个新 token 的 Q/K/V" + "全部历史的 K/V"。
历史 token 的原始输入 $X$、它们的 $Q$、以及其他中间激活，一步都不需要再碰。

## 4.2 关键推导：为什么只缓存 K 和 V，不缓存 Q？

先把式 (1) 的 softmax 打分记成一个权重符号 $\alpha$，注意力输出就是加权和：

$$
o_t = \sum_{j=1}^{t} \alpha_{t,j}\, v_j,
\qquad
\alpha_{t,j} = \frac{\exp\!\left( q_t \cdot k_j / \sqrt{d_{head}} \right)}{Z_t},
\qquad
Z_t = \sum_{j'=1}^{t} \exp\!\left( q_t \cdot k_{j'} / \sqrt{d_{head}} \right) \tag{2}
$$

关键一步：把下标 $t$ 换成 $t{+}1$，再把求和**逐项写开**，看 token $t$ 的痕迹还剩多少：

$$
o_{t+1} =
\underbrace{\alpha_{t+1,1}\, v_1 + \alpha_{t+1,2}\, v_2 + \cdots}_{\text{老的 } v \text{ 继续被加权}}
+ \underbrace{\alpha_{t+1,t}\, v_t}_{v_t \text{ 在这里}}
+ \underbrace{\alpha_{t+1,t+1}\, v_{t+1}}_{\text{新 token 自己}}
$$

$$
\alpha_{t+1,j} = \frac{\exp\!\left( q_{t+1} \cdot k_j / \sqrt{d_{head}} \right)}{Z_{t+1}}
\;\xrightarrow{\; j = t \;}\;
\frac{\exp\!\left( q_{t+1} \cdot k_t / \sqrt{d_{head}} \right)}{Z_{t+1}}
\quad \leftarrow \; k_t \text{ 在这里}
$$

$$
q_t \text{ 出现在 } o_{t+1} \text{ 的哪里？一个字母都找不到。}
$$

结论逐条对号：

- $v_t$：出现在 $o_{t+1}$ 的加权和里（$\alpha_{t+1,t}\, v_t$ 这一项）→ 未来每步都要用 → **必须缓存**
- $k_t$：出现在打分 $\alpha_{t+1,t} = \exp(q_{t+1} \cdot k_t / \sqrt{d_{head}}) / Z_{t+1}$ 中 → 未来每步都要用 → **必须缓存**
- $q_t$：式 (2) 里 $q$ 永远和"当前行"绑定（$o_t$ 的打分用 $q_t$，$o_{t+1}$ 的打分用 $q_{t+1}$）。
  $o_t$ 算完即定稿，未来没有任何一行会再引用 $q_t$ → **用完即弃**

| 向量 | 第 $t$ 步用 | 第 $t{+}1$ 步用 | 第 $t{+}2$ 步用 | 结论 |
|---|---|---|---|---|
| $q_t$ | ✓（去打分） | ✗ | ✗ | 不缓存（省下 1 份） |
| $k_t$ | ✓（被打分） | ✓ | ✓ | 缓存 |
| $v_t$ | ✓（被加权） | ✓ | ✓ | 缓存 |

所以"KV Cache"不是工程偷懒，而是**数学上精确的最小必要量**。

### append-only：KV Cache 最重要的工程性质

第 $t$ 步对缓存只有两个动作——"追加"和"只读"，且作用于不同的量：

$$
\underbrace{k_t = x_t W_K, \qquad v_t = x_t W_V}_{\text{写路径：attention 之前就已算死，只依赖 } x_t}
\qquad
\underbrace{o_t = \mathrm{softmax}\!\left( \frac{Q_t\, K_{1:t}^{\top}}{\sqrt{d_{head}}} \right) V_{1:t}}_{\text{读路径：只读 } K_{1:t},\, V_{1:t}\text{，不回写}}
$$

$$
[\, \underbrace{k_1, \dots, k_{t-1}}_{\text{冻结，永远只读}} \,] \;\; \xrightarrow{\;\text{append}\;} \;\; [k_1, \dots, k_{t-1}, k_t]
$$

$o_t$ 不回写缓存：它走残差 + FFN 生成 $x_{t+1}$，再投影出**新的** $k_{t+1}, v_{t+1}$
去追加。已写入的值永不改变——这就是 **append-only（只追加、永不修改）**。

这个性质直接支撑了 vLLM 的三大机制：

| append-only 保证什么 | 对应的 vLLM 机制 |
|---|---|
| 值不可变 → 多请求可安全共享同一物理块 | 前缀缓存（4.9 节） |
| 丢弃即回滚 → 多算的 draft token 直接扔 | 投机解码（第 08 章） |
| 同输入必得同 KV → 重算结果与原缓存一致 | 抢占后 RECOMPUTE（第 05 章） |

> 例外说明：KV 量化（有损压缩存储）和滑动窗口（淘汰最老块）会改变缓存内容，
> 但它们是**存储管理策略**，不违反"attention 不改写缓存"这条数学性质。

**append-only 的结构性前提：因果性。** "历史冻结"并非所有注意力都有——
它要求 $k_j$ 与"未来的问题"解耦，这正是因果掩码（decoder-only 天生全程因果）给的：

| 结构 | $k_j$ 依赖什么 | 可缓存？ |
|---|---|---|
| Decoder-only（因果掩码） | 只依赖 $x_j$ 及更早历史 | ✅ 每算一次，终身有效 |
| BERT 双向注意力 | 依赖**整个序列**（含未来 token） | ❌ 序列一变，全部重算 |
| Encoder-Decoder 交叉注意力 | 只依赖 encoder 输出 | ✅ 但只算一次，不随生成增长 |

顺带回答三个常见疑问：

1. **为什么不直接缓存注意力输出 $o$？** 因为 $o_t$ 依赖"未来的问题" $q_{t'}$，
   而 $q_{t'}$ 还没出生。注意力是"按问题动态加权的历史信息混合"，
   不存在一个预先算好的通用答案。
2. **KV Cache 和 RNN 像吗？谁的时间复杂度更友好？** 结构上像——两者都是
   "历史只追加、永不改写"（RNN 的 $h_j$ 同样冻结），所以 decode 每步只付增量成本：
   没有缓存时第 $t$ 步要重算全部 $O(t^2)$ 注意力，总生成 $O(n^3)$；
   有了缓存，每步 $O(t \cdot d)$，总共 $O(n^2 \cdot d)$。但要注意两个易混点：

   - **渐近复杂度上 RNN 才更便宜**，decoder-only 并不占优：

     | | 每步状态更新 | 总时间 | 状态信息 |
     |---|---|---|---|
     | RNN | $O(d^2)$，与历史长度无关 | $O(n \cdot d^2)$ | **有损压缩**，$O(1)$ |
     | Decoder-only + KV Cache | $O(t \cdot d)$，随历史增长 | $O(n^2 \cdot d)$ | **无损**，$O(n)$ |

   - **"历史冻结"也不是 decoder-only 的专利**（RNN 的 $h_j$ 同样冻结），
     它真正的独特优势是：**在无损的前提下把每步成本压到线性**——
     用 $O(n)$ 的状态换质量，再靠缓存省掉平方级重算。
     而 $O(n)$ 状态的显存代价，正是 vLLM 要管理的全部问题（4.4 节展开）。
3. **dropout 会影响 KV cache 吗？** 推理时不会：LLM 推理跑 `model.eval()`，
   dropout 是关闭的，vLLM 里根本没有 dropout。退一步说，即使开启，
   attention dropout 也只随机置零一部分打分 $\alpha_{t,j}$，仍然不改写 $k_t, v_t$ 本身。
   真正致命的是**决定性**被破坏：dropout 会让"同样的 token 重算出不同的 KV"，
   而上表中的前缀复用和 RECOMPUTE 都依赖
   "同样输入 → 完全相同的 KV"。所以推理引擎必须关闭 dropout——
   这不是可选项，是缓存正确性的前提。

## 4.3 Prefill 与 Decode：两种截然不同的计算

**Prefill（prompt 的 $n$ 个 token 一次进）**：$Q K^{\top}$ 是 $n \times n$ 方阵——

$$
\text{计算量} \approx
\underbrace{2\, N_p\, n}_{\text{权重 GEMM}}
+ \underbrace{4\, L\, n^2\, d_{model}}_{\text{注意力（} n^2 \text{ 项!）}}
$$

**Decode（每步只进 1 个新 token，上下文长度 $t$）**：

$$
\text{计算量} \approx 2\, N_p + 4\, L\, t\, d_{model}
\qquad
\text{访存量} \approx 2\, N_p\, b + t \cdot S_{kv}
$$

其中 $N_p$ 为参数量，$b$ 为每参数字节数，$S_{kv}$ 为每 token 的 KV 字节数——
整个权重 + 整个 KV，每步都要从显存搬一遍。

关键比值——算术强度（FLOPs ÷ 显存字节数）：

$$
\text{decode 算术强度} \approx \frac{2 N_p \text{ FLOPs}}{2 N_p\, b \text{ 字节}} \approx 1 \ \text{FLOP/字节}
\quad \ll \quad \text{GPU 平衡点} \approx 100 \sim 300 \ \text{FLOP/字节}
$$

→ decode 时计算单元大量空转，瓶颈全在显存带宽（第01章伏笔的数学化）。

**两个直接推论**（vLLM 两大机制的根）：

1. 搬权重是每步的"固定成本" → 让 B 个请求一起 decode，权重只搬一次 →
   **连续批处理**摊薄成本（第 05 章）；
2. 每步还要搬"全部历史 KV"，且 KV 是每请求**独占**的 →
   显存里能同时躺下多少 KV，决定并发上限 → **KV Cache 管理 = vLLM 的命脉**。

## 4.4 KV Cache 显存公式：完整推导链

从模型结构一步步推，以 **Qwen2.5-7B（FP16）** 为例（参数查 `config.json`）：

**第 1 步 · 单 token、单层的 K/V 形状**：$x_t \in \mathbb{R}^{3584}$；
GQA 下 32 个 Q 头共享 4 组 KV 头，所以 $k_t$ 只需 $4 \times 128 = 512$ 个数，$v_t$ 同。

**第 2 步 · 单 token、单层字节数**：

$$
(K + V) \times 512 \text{ 个数} \times 2 \text{ 字节} = 2 \text{ KB}
$$

**第 3 步 · 乘层数**：

$$
2 \text{ KB} \times 28 \text{ 层} = 56 \text{ KB/token}
\qquad \text{（第01章那个公式的完整来历）}
$$

**通用公式**：

$$
\text{KV每token} =
\underbrace{2}_{K,V} \times L \times H_{kv} \times d_{head} \times \text{dtype字节数}
$$

$$
\text{请求级 KV} = \text{KV每token} \times s
\qquad
\text{显存里的 KV 总量} = \text{请求级 KV} \times B
$$

其中 $L$ 为层数，$H_{kv}$ 为 KV 头数，$d_{head}$ 为每头维度，$s$ 为序列长度，$B$ 为并发请求数。

KV Cache 在显存里的形状，用张量记号一个式子写完：

$$
\text{KV张量} = [L,\ H_{kv},\ s,\ d_{head}] \times 2(K,V) \times \text{dtype字节数}
$$

> vLLM 的做法：沿序列维 $s$ 切成"每 16 个 token 一块"的物理块分散摆放（第二部分）。

**核对表（FP16，2 字节/元素）**：

| 模型 | L | H_kv | d_head | 每 token KV | 4096 上下文单请求 |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 2 | **64** | 12 KB | 49 MB |
| Qwen2.5-7B | 28 | 4 | 128 | 56 KB | 224 MB |
| Llama-3-70B | 80 | 8 | 128 | 320 KB | 1.3 GB |

> 注意 Qwen2.5-0.5B 的 head_dim 是 64（hidden 896 ÷ 14 头）而非 128——
> 别照抄别人的公式，永远对着 config.json 算。

## 4.5 KV 头的进化：MHA → MQA → GQA → MLA

式 (2) 里 decode 每一步都要"读全部历史 KV"，所以 **KV 头数直接决定显存与带宽**。
工程界一路压缩它：

- **MHA**：每个 Q 头独立配一对 K/V 头，$H_{kv} = H$（最大，质量基准）
- **MQA**：全部 Q 头共享 1 对 K/V 头，$H_{kv} = 1$（最省，质量损失明显）
- **GQA**：每 $G$ 个 Q 头共享一对 K/V 头，$1 < H_{kv} < H$（折中，当前主流）。
  Qwen2.5-7B：32 个 Q 头 → 4 组 KV 头（$G=8$），KV 省 8 倍
- **MLA**（DeepSeek）：不压"头数"，压"每头的维度"——把 K/V 联合投影进一个
  低秩潜向量 $c_{KV}$，推理时只缓存它（外加一小段 RoPE 位置键）

| 方案 | KV 开销（相对 MHA） | 质量 | 代表模型 |
|---|---|---|---|
| MHA | 100%（基准） | 最好 | GPT-2、Llama-1 |
| MQA | $1/H$ | 有损 | PaLM、Falcon |
| GQA | $H_{kv}/H$ | 接近 MHA | Llama-3、Qwen2.5 |
| MLA | 低秩（如 576 维/层） | 接近 MHA | DeepSeek-V2/V3 |

MLA 的数字感受（DeepSeek-V3：61 层，每层缓存 512 维潜向量 + 64 维 RoPE 键）：

$$
(512 + 64) \times 61 \text{ 层} \times 2 \text{ 字节} \approx 68.6 \text{ KB/token}
$$

—— 671B 的巨模型，KV 开销却与 7B 级 GQA 模型同一量级。

> **对定制开发的意义**：换 KV 结构 = 换 attention backend。
> MLA 在 vLLM 中走专门的 MLA backend，其 KV 池与块管理也和 GQA 不同（第 11、14、16 章）。

## 4.6 三个推论：KV Cache 如何决定 vLLM 的命运

1. **容量推论**：最大并发 ≈ KV 显存预算 ÷ (每 token KV × 平均序列长度)。
   0.5B 模型轻松几千 token 并发，70B 模型几十个请求就能把显存吃光。
2. **带宽推论**：decode 每步全量重读 KV → 序列越长，每 token 越慢、越贵；
   相同前缀反复重算+重读是纯浪费 → 前缀缓存（4.9 节）。
3. **管理推论**：序列长度不可预知 + 请求随机进出 + 前缀可共享 →
   传统"按最大长度预留一整块连续显存"必然碎片化 → **PagedAttention 登场**。

---

# 第二部分 · PagedAttention：给 KV Cache 装一个"操作系统"

## 4.7 从操作系统借来的智慧

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

## 4.8 逐帧动画：一个请求的一生（显存视角）

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

## 4.9 前缀缓存（Prefix Caching）：同一个开头，全世界只算一次

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

## 4.10 KV Cache 显存是怎么"摸底"出来的？

对应第 03 章启动流程 [4][5]，源码位置：

- `vllm/v1/worker/gpu_worker.py` → `determine_num_available_blocks()`：跑一次 `profile_run()`，测量"模型权重 + 激活 + CUDA 上下文"占了多少；
- 可用 KV 显存 = `总显存 × gpu_memory_utilization − 摸底值`；
- `vllm/v1/core/kv_cache_utils.py`：算出每块多大 → 得到块数 N → **一次性**创建 N 块的 KV 大张量。

**动手算一遍**（以本教程演示机 RTX 4060 8GB、Qwen2.5-0.5B 为例）：

$$
\begin{aligned}
\text{KV 显存预算} &= 8.0 \times 0.85 - 1.0(\text{权重}) - 1.0(\text{激活}) \approx 4.8 \text{ GB} \\
\text{每 token KV} &= 2 \times 24 \text{ 层} \times 2 \text{ 头} \times 64 \text{ 维} \times 2 \text{ B} = 12 \text{ KB} \\
\text{每块（16 token）} &\approx 0.19 \text{ MB} \\
\text{块数} &\approx \frac{4.8 \text{ GB}}{0.19 \text{ MB}} \approx 25000
\end{aligned}
$$

→ 日志 `# GPU blocks` 应该在两万上下。

> 运行 `vllm serve` 后对照日志验证你的手算。误差 ±20% 正常（中间结果、CUDA Graph 等占额）。

## 4.11 源码走读：核心 3 个类（v0.31.0）

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

## 4.12 动手实验

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

日志出现 `Preempted request ... (reason: RECOMPUTE)` 时，结合 4.11 的 `allocate_slots → None`
思考：调度器为什么选择"丢掉一个请求的 KV 重算"而不是"直接拒绝新请求"？（第 05 章揭晓）

## 4.13 本章小结 + 自测

**第一部分（KV Cache 原理）**：

1. 用式 (2) 推导：为什么 q_t 不需要缓存，而 k_t、v_t 必须缓存？
2. decode 的算术强度为什么约等于 1 FLOP/字节？这如何推出"批处理摊薄成本"？
3. 用通用公式手算 Qwen2.5-0.5B（24 层 × 2 KV 头 × 64 维，FP16）的每 token KV 大小。
4. GQA 相比 MHA 省 8 倍 KV 的代价是什么？MLA 压缩的维度和 GQA 有何本质不同？

**第二部分（PagedAttention）**：

5. block_size=16 意味着内部碎片上限是多少 token？
6. 前缀缓存的哈希 key 是"单个块的内容"还是"从开头到该块的完整前缀"？为什么必须后者？
7. 请求 A 结束时它引用的共享块会被回收吗？
8. vLLM 为什么要在启动时 profile_run？如果不做（直接按理论值分配）会怎样？
9. `allocate_slots` 返回 None 时，调度器有哪些选择？

**下一章**：[第 05 章 · 调度器与连续批处理](05_调度器与连续批处理.md)
