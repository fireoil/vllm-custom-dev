# 第 16 章 · TileLang 工业级内核开发完全指南：DeepSeek / GLM / Qwen 多硬件推理加速

> **本章定位**：第 15 章带你用 TileLang 写出了 GEMM / RMSNorm / FlashAttention 三个入门内核；
> 本章是它的**工业级完全版**——目标是让你具备把 **DeepSeek（MLA/MoE/FP8）、GLM、Qwen（GQA/QK-Norm/MoE/W4A16）**
> 的关键算子用 TileLang 实现出来，并**同一份源码适配 NVIDIA / AMD / 昇腾 NPU / Apple**，
> 最终以 vLLM 生产标准（custom op / warmup / 平台门卫 / 回退链）集成回推理引擎。
>
> **版本基线**：TileLang **v0.1.15**（2026-09-30，含官方 Ascend 950 后端）+ vLLM **v0.31.0**（与本教程全系列一致）。
> TileLang 迭代很快（0.1.x 线），个别 API 以 [官方文档](https://tilelang.com/) 为准，但编程模型与工程方法长期稳定。

**与全系列的协同地图**（先知道每一节和前面哪章"接力"）：

| 本章节 | 前置章节 | 前置 lab |
|---|---|---|
| 16.5 DeepSeek MLA / Paged KV | [第 04 章](04_PagedAttention与KV_Cache.md)、[第 11 章](11_主流模型接入_DeepSeek_GLM_Qwen.md) | `lab/sim_paged_kv.py`、`lab/tilelang/03` |
| 16.6 GQA / MoE / W4A16 | [第 11 章](11_主流模型接入_DeepSeek_GLM_Qwen.md)、[第 08 章](08_高级特性_投机解码_量化_结构化输出.md) | `lab/tilelang/01、02` |
| 16.7 跨硬件 | [第 12 章](12_异构硬件部署指南.md)、[第 14 章](14_硬件与模型适配_代码级细节.md) | — |
| 16.8 vLLM 集成 | [第 13 章](13_vLLM插件机制实现细节.md)、[第 15 章](15_性能优化与TileLang算子实战.md) | `lab/my_vllm_plugin/` |
| 16.9 性能工程 | [第 09 章](09_服务化部署与性能调优.md)、[第 14 章](14_硬件与模型适配_代码级细节.md) | `lab/tilelang/01~03` |

---

## 16.0 全景：TileLang 生态与选型决策

### 16.0.1 一句话定位

TileLang（tile-lang）是构建在 TVM 之上的 **tile 级 DSL**：用 Pythonic 语法描述
"一个 CTA 算多大一块数据、怎么在内存层级间搬运、怎么调 TensorCore"，
编译器自动完成**布局推导、向量化、软件流水、TMA/WGMMA 指令映射、bank-conflict 规避**，
同时保留 `T.annotate_layout` 等手动控制出口。产出物是可直接从 Python 调用的编译内核（也可导出源码）。

### 16.0.2 tile-ai 生态六件套（知道有这些，别重复造轮子）

| 项目 | 是什么 | 什么时候用 |
|---|---|---|
| [tilelang](https://github.com/tile-ai/tilelang) | 语言本体 + 编译器（本章主角） | 写/改内核 |
| [TileOPs](https://github.com/tile-ai/TileOPs) | 官方高性能算子库（varlen GQA、W4A16 长 K、DeltaNet、FlashMLA 系等） | 先找现成算子，再决定是否自己写 |
| [TileRT](https://github.com/tile-ai/TileRT) | tile 级推理运行时（GLM-5.1 生产于 Z.ai、DeepSeek-V3.2、1000+ TPS 案例；经 vLLM V1 connector 做 PD 分离） | 极致单流解码速度的参考/替代 |
| [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) | 昇腾早期后端（`ascendc_pto` / `npuir` 两分支，2025-09 起） | 昇腾 910B 等上一代 NPU |
| [TileSight](https://github.com/tile-ai/TileSight) | tile 中心的分析式性能模型（CPU 侧预估算子耗时/缓存流量） | 写 kernel 前估算 roofline |
| [tilelang-puzzles](https://github.com/tile-ai/tilelang-puzzles) / [tilelang-lsp](https://github.com/tile-ai/tilelang-lsp) | 10 道渐进编程题 / LSP（IDE 内联提示 buffer 形状、dtype、layout） | 学习与开发体验 |

### 16.0.3 与其他内核技术路线的对比

| 维度 | 手写 CUDA / CUTLASS | Triton | **TileLang** | DeepGEMM / 专属库 |
|---|---|---|---|---|
| 生产力 | 低（数百行） | 高 | **高（30~80 行一个 FA）** | 开箱即用 |
| 性能上限 | 最高 | 高 | **高（官方 MLA 打平 FlashMLA / Aiter）** | 特定 shape 最高 |
| 硬件覆盖 | 每硬件重写一遍 | NVIDIA / AMD | **NVIDIA / AMD / 昇腾 / Metal / CPU(LLVM) / CuTeDSL / WebGPU** | 绑定特定硬件 |
| 布局与流水控制 | 全手动 | 半自动 | **自动推导 + 可手动（annotate_layout / PassConfig）** | 内置 |
| vLLM 中的地位 | 内核仓大量 | 主力 | **v0.31 起为生产依赖**（`has_tilelang`、`vllm/tilelang_utils/`） | DeepSeek 路线主力 |
| 学习曲线 | 陡 | 缓 | **缓→中**（会 Python 就能起步） | 无（直接用） |

### 16.0.4 选型决策树（背下来，别上来就写 kernel）

```
要不要自己写 kernel？
├─ ① 官方专属库已覆盖且硬件支持（DeepGEMM/FlashMLA/FlashInfer/Aiter）→ 直接用
├─ ③ vLLM 内置 Triton/CUDA 后端性能已达标 → 直接用（第14章内核选择链路）
├─ 满足任一条 → 用 TileLang 写（②层，本章）：
│    · 硬件不被官方库支持（昇腾 NPU / 新卡 / 低配消费卡）
│    · 算子融合机会明确（QK-Norm+RoPE、dequant+GEMM、gate+grouped-GEMM）
│    · 模型结构定制（MLA 变体、新注意力、稀疏/线性注意力）
└─ ② TileLang 到不了目标性能、需要特殊指令 → 再下沉手写 CUDA
```

### 16.0.5 TileLang 在 vLLM 里的真实存在感（不是玩具）

- `vllm.utils.import_utils.has_tilelang`：tilelang 是 vLLM 的**可选依赖**，探针函数专门检测它；
- `vllm/tilelang_utils/`：延迟 JIT 内核封装 + 编译期 warmup 体系（15.5 已拆解）；
- `vllm/model_executor/kernels/mhc/tilelang.py`：真实生产内核 + custom op 注册；
- vLLM 2026-10 的 DeepSeek-V4.1-Flash 适配中，内核层（MegaAttention/Mega-mHC 等）与
  TileLang 生态同属"内核加速层"，遵循同一套集成规范（16.8 展开实现）。

---

## 16.1 环境安装与硬件支持矩阵

### 16.1.1 安装（三种方式，国内环境友好）

```bash
# 方式一: PyPI(国内走清华源)
pip install tilelang -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式二: 跟随 main 分支(特性最新, 风险自担)
pip install git+https://github.com/tile-ai/tilelang.git

# 方式三: 源码(需要改编译器/加后端时)
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang && pip install -e .
```

**版本红线**：Python ≥ 3.9（3.8 支持止于 v0.1.6.post2）；torch 与 CUDA 按你现有 vLLM 环境不动，
tilelang 只是"多装一个包"，与 vLLM 共存无冲突（它自己捆绑 TVM 运行时）。

**容器化模板**（配合你"容器内开发、不动宿主机"的惯例）：

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.10-py3
# ---- 国内源 ----
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# ---- vLLM v0.31 + TileLang v0.1.15 共存 ----
RUN pip install vllm==0.31.0 tilelang==0.1.15 && \
    pip install jupyterlab ipykernel
```

安装验证：

```bash
python -c "import tilelang; print(tilelang.__version__)"     # 0.1.15
python -c "from vllm.utils.import_utils import has_tilelang; print(has_tilelang())"  # True
```

### 16.1.2 硬件支持矩阵（选后端就是选这张表）

| 硬件 | Tensor Core / 矩阵单元 | TileLang 支持状态 | 备注 |
|---|---|---|---|
| NVIDIA V100 / A100 / 3090 / 4090 / A6000 | mma | **官方测试通过** | 消费卡（含本教程 4060 级别）可用 |
| NVIDIA H100/H200 | WGMMA + TMA | **Auto TMA/WGMMA**（官方测试） | warp-specialization 可用但可关 |
| NVIDIA Blackwell（B200/RTX50） | tcgen05 | v0.1.12+ 扩展；`T.mma_gemm_blockscaled`（SM120 NVF4） | FP4/FP8 块缩放 MMA |
| AMD MI250 | Matrix Core | 官方测试通过（自动映射） | ROCm 后端 |
| AMD MI300X / MI355X | Matrix Core + AsyncCopy | 官方测试；FlashMLA 打平 Aiter 汇编内核 | TileRT 在 MI355X 有 AgentX 榜单案例 |
| **华为昇腾 950** | Cube + Vector | **v0.1.15 官方后端**（原生代码生成、自动调度与同步、SIMD/SIMT） | GEMM / FlashAttention 官方示例 |
| 华为昇腾 910B 等 | Cube | [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend)（ascendc_pto / npuir 分支，预览） | 走社区仓库，非主线 |
| Apple M4 / M5 | — | Metal 后端（M5 cooperative tensor GEMM） | 端侧实验 |
| CPU / 其他 | — | LLVM 后端（v0.1.12+）、NVRTC、CuTeDSL、WebGPU | 逻辑调试 / 边缘场景 |

> 实用结论：**NVIDIA/AMD 主线 pip 装；昇腾 950 用主线（v0.1.15+）、910B 用 tilelang-ascend 仓库；
> 消费级小卡完全可开发，性能目标按算力比折算（15.2 的 70~90% cuBLAS 标准同样适用）。**

### 16.1.3 与 vLLM 共存的版本对齐建议

| vLLM | tilelang 建议 | 说明 |
|---|---|---|
| v0.31.x | 0.1.13 ~ 0.1.15 | vLLM 只做 `has_tilelang` 探测 + JIT 调用，无强锁版本 |
| 更旧 (<0.30) | 不建议引入 | `vllm/tilelang_utils/` 体系从 v0.31 起完善 |
| vllm-ascend（昇腾） | 与主线对齐 + 昇腾后端 | vllm 与 vllm-ascend 版本必须成对（第 12 章规则不变） |

---

## 16.2 编程模型：从 15.1 到工业级

第 15 章讲了"四级内存 + 五类原语"。工业级要再吃透四件事：**线程/warp 组织、布局推导、
硬件代际映射、编译器开关**。

### 16.2.1 一次 kernel 执行的完整映射（脑子里要有这张图）

```
T.Kernel(grid_x, grid_y, threads=N)          # ① CTA 网格: 每 tile 一个 CTA
  └─ N threads = N/32 warps                  # ② warp 是 TensorCore 指令的发射单位
       ├─ T.alloc_shared   → 共享内存(smem)   # ③ 搬运缓冲, T.Pipelined 多级化
       ├─ T.alloc_fragment → 寄存器           # ④ 累加器/中间值, 每线程持有切片
       │    └─ T.gemm(A_s, B_s, C_f)          # ⑤ 编译器按 warp policy 把 C_f 切给各 warp
       └─ T.Parallel 循环                     # ⑥ 布局推导入口: 由此向量化/决定数据归属
```

**GemmWarpPolicy 三种策略**（多 warp 分一个 tile 的矩阵乘时怎么切）：

| 策略 | 切法 | 适用 |
|---|---|---|
| `FullRow` | 每个 warp 负责输出的整行块 | **默认首选**（FA/MLA 里都是它） |
| `FullCol` | 每个 warp 负责整列块 | N 维特别大的 shape |
| `SquareSplit` | 行列对半切 | 方形 tile、寄存器压力平衡 |

### 16.2.2 布局推导：自动为主，手动为辅

- **自动**：`T.Parallel` 循环是布局推导入口，编译器据此决定向量化宽度、线程数据归属；
- **手动**：`T.use_swizzle(panel_size=10)` 对 global→smem 地址做交错（防 bank conflict，GEMM 标配）；
  `T.annotate_layout(buffer, layout)` 给单个 buffer 指定布局（如官方 GEMM 例里的
  `make_zn_layout(B_shared)`，属于"快到顶了再拧"的最后一档旋钮，初学不用）。

### 16.2.3 硬件代际映射：你写的是同一份 Python，编译器选不同指令

```
你的 kernel (T.gemm / T.copy / T.Pipelined)
   ├─ NVIDIA Ampere  → mma 流水
   ├─ NVIDIA Hopper  → WGMMA + TMA 异步拷贝 (+ 可选 warp-specialization)
   ├─ NVIDIA Blackwell → tcgen05 / blockscaled MMA (NVF4)
   ├─ AMD MI250/300X → Matrix Core (+ AsyncCopy)
   ├─ 昇腾 950       → Cube 单元 (官方后端自动调度+同步)
   └─ CPU(LLVM)      → 标量/向量指令 (调试利器: 逻辑错误先在 CPU 复现)
```

**这就是"一套代码多硬件"的技术根基**，也是第 12/14 章硬件抽象在内核层的延续。

### 16.2.4 PassConfig：编译器开关的"生产姿态"

vLLM 生产环境（15.5②）的选择及理由，扩展成速查表：

| PassConfigKey | 含义 | vLLM 生产取值 | 你什么时候改 |
|---|---|---|---|
| `TL_DISABLE_WARP_SPECIALIZED` | Hopper warp 专项化调度 | **True（关）** | H100/H200 上实验有稳定收益才开 |
| `TL_DISABLE_TMA_LOWER` | TMA 张量布局降级 | **True（关）** | TMA 路径在你的 shape 上报错/不稳时保持关闭 |
| `TL_PTXAS_REGISTER_USAGE_LEVEL` | ptxas 寄存器压力调优级（0~10） | 10（仅 CUDA） | 寄存器溢出(spill)或占用率权衡时扫参 |

枚举完整清单（版本相关，以装到的为准）：

```bash
python -c "import tilelang; print(list(tilelang.PassConfigKey.__members__))"
```

### 16.2.5 JIT、缓存与调试工具链

| 能力 | 用法 | 价值 |
|---|---|---|
| JIT 装饰器 | `@tilelang.jit(out_idx=[-1])` | 尺寸是编译期常量 → 每 shape 一份专属内核，自动缓存 |
| 导出源码 | `kernel.get_kernel_source()` | 检查生成的 CUDA/汇编质量（第 14 章"看编译器底牌"） |
| 跨主机二进制缓存 | v0.1.13+（PR #2459） | 编译产物跨兼容主机复用 → **分发 wheel 化的基础**（16.8） |
| `T.print` | 打印 buffer/变量 | kernel 内看数值（官方 debug 教程） |
| IR Lower Trace / Pass Visualizer | v0.1.13 / v0.1.12+ | 逐 pass 看编译器怎么改你的 IR（性能玄学变实证） |
| 编译期正确性 | Z3 定理证明器接入 Arith（v0.1.6+） | 越界/整除类错误编译期暴露 |
| LSP | tilelang-lsp | IDE 内联显示 buffer 形状/dtype/layout 推导结果 |

---

## 16.3 API 速查大全（工业版八类原语）

在第 15 章"五类原语"基础上补全为**八类**，写 kernel 按表索骥：

| 类别 | API | 要点 |
|---|---|---|
| 声明 | `T.Tensor((shape), dtype)` / `T.alloc_shared` / `T.alloc_fragment` | 全局/共享/寄存器；fragment 也用于标量组（如每行 m、l） |
| 搬运 | `T.copy(src_view, dst)` | **从"视图起点"拷"dst 形状"个元素**（支持 stride、dtype 转换）；自动选向量化/异步路径 |
| 计算 | `T.gemm(A, B, C, transpose_B=, policy=)` | 自动 TensorCore；M/N/K ≥ 16；累加器 fp32 |
| 稀疏/量化 MMA | `T.gemm_sp`（2:4 稀疏）、`T.mma_gemm_blockscaled`（Blackwell NVF4） | 量化/稀疏路线 |
| 逐元素 | `T.Parallel(i, j)` 循环 + `T.exp2/T.sqrt/T.max/T.if_then_else` | 布局推导入口；mask 用有限大负数（见 16.4 陷阱） |
| 规约 | `T.reduce_max/max/sum(src, dst, dim=, clear=True)` | 按维规约到 fragment；`clear=True` 覆写而非累加 |
| 原子/填充 | `T.atomic_add`、`T.fill`、`T.clear` | split-K、累加器初始化 |
| 流程/调试 | `T.Pipelined(n, num_stages=k)`、`T.use_swizzle`、`T.print` | 动态循环上界可用运行时标量（分页解码的关键，见 16.5） |

三个高频细节再强调：

1. **`T.copy` 的视图语义**：`T.copy(A[by*block_M, k*block_K], A_shared)` 表示"从 A 的
   (by*block_M, k*block_K) 起拷一块 block_M×block_K"——所以取 KV 行内切片时改 dst 形状即可
   （16.5 中 `V = cache[..., :512]` 就这么实现）。
2. **动态循环上界**：`T.Pipelined(T.ceildiv(seq_lens[b], BLOCK), ...)` 合法（官方 MLA 例同款）
   ——分页解码按真实长度收敛循环的钥匙。
3. **fragment 间 `T.copy`** = 带类型转换的元素拷贝（fp32 fragment → fp16 fragment 喂第二个 gemm）。

---

## 16.4 六大工业算子模式地图

写任何 LLM 算子前，先对号入座（判定方法见 14.5）：

| # | 模式 | 判定 | 代表算子 | 核心原语组合 | 成功指标 | 本教程实现位置 |
|---|---|---|---|---|---|---|
| 1 | GEMM 型 | compute-bound | qkv_proj / o_proj / lm_head | `T.gemm`+`Pipelined`+`swizzle` | TFLOPS ≥ 70% cuBLAS | 15.2 / 官方 `examples/gemm` |
| 2 | 带宽融合型 | memory-bound | RMSNorm / RoPE / QK-Norm / act | `T.Parallel` 向量化 + `reduce` | 有效带宽 ≥ 70% 理论值 | 15.3、**16.6.2** |
| 3 | Online-softmax 注意力 | 混合 | FA prefill/decode | 双 `T.gemm` + `exp2` + `reduce` | 与 SDPA 数值一致 | 15.4、**16.6.1** |
| 4 | Paged-KV 解码 | 混合 | GQA/MLA decode | `block_table` + 动态循环 | TPOT 延迟 | **16.5.2** |
| 5 | 分组 GEMM | MoE | gate_up / down | `expert_ids` 按块选权 | 端到端吞吐 | **16.6.3** |
| 6 | Dequant 融合 | 量化 | W4A16 / FP8-block GEMM | 位运算反量化 + `T.gemm` | 带宽/TFLOPS | **16.6.4** |

**一个贯穿全章的工程陷阱先立此存照**：在线 softmax 的掩码不要用 `-inf`。
全掩码行会出现 `-inf - (-inf) = NaN`，并沿 `exp2` 传播污染输出。工业做法是用
**有限大负数哨兵**（如 `-1e6`），配合"越界行不写回"——下面所有 kernel 都这么写。

---

## 16.5 DeepSeek 适配实战：MLA / MoE / FP8

### 16.5.1 DeepSeek-V3 系结构与内核挑战速览（衔接第 11 章）

| 组件 | 结构参数（V3/V3.2） | 内核挑战 | 本章方案 |
|---|---|---|---|
| 注意力 | **MLA**：128 头；`kv_lora_rank=512`；`qk_rope_head_dim=64`；KV cache **576/ token** | 吸收后 decode 变"多行共享单 KV"的 GEMV-like | 16.5.2 |
| FFN | **MoE**：256 专家激活 8；FP8 128×128 块缩放 | 分组 GEMM + 细粒度反量化 | 16.6.3 模式 + 块缩放扩展 |
| MTP | 多 token 预测（投机解码） | 系统层 | 第 08/11 章（vLLM 特性层解决，无需新 kernel） |

**MLA 吸收的数学**（为什么 KV cache 只要 576 维）：

```
标准注意力:  softmax(q·K^T/√d)·V
MLA:        k_c = W_UK·c , q_nope·k_c = q_nope·(W_UK·c) = (W_UK^T·q_nope)·c
            → 令 q_latent = W_UK^T·q_nope (512 维), 则
            Q' = [q_latent | q_pe] (512+64 = 576)
            K' = [c       | k_pe]  (512+64 = 576)   ← 这就是 cache 的 576!
            V  = c (512)                            ← v_c = W_UV·c 被吸收, 输出再乘 W_UV
softmax scale 用未吸收的 qk 维度: 1/√(128+64) = 1/√192
```

vLLM 的 MLA cache 布局即 `(num_blocks, block_size, 576)`，前 512 维既是 K 的主体也是 V。

### 16.5.2 完整实现：MLA Paged Decode kernel（工业参考版）

设计要点（每一条都对应官方 FlashMLA / tilelang `examples/deepseek_mla` 的一个手法）：

1. **head-as-row**：128 个 Q 头共享同一份 KV → 把"头"当 M 维行（block_H=64 头/CTA），
   M=64 足够 TensorCore 吃满，且 KV 只需加载一次服务 64 行；
2. **分页 + 动态循环**：外层按 `block_table` 逐页取 KV，循环上界用**运行时** `seq_lens[b]`；
3. **V 复用 cache 前缀**：`T.copy(cache[page,0,0], V_s)` 拷 (16,512) 即取前 512 维；
4. 有限哨兵掩码；`T.exp2` 预乘 log2(e)；fp32 累加。

> 注：教学版按"每迭代处理一个 16-token 页"展开（block_N=BLOCK_SIZE=16），逻辑最清晰。
> 生产内核会一次拷多页拼成 64~128 的块再算（减少迭代、加深流水）——读者优化练习（16.11 实验 16-1）。

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def mla_paged_decode(
    B: int,                    # 并发请求数(调度器给出)
    num_blocks: int,           # KV cache 总页数
    H: int = 128,              # 头数
    D_LAT: int = 512,          # kv_lora_rank
    D_ROPE: int = 64,          # rope 维度
    BLOCK_SIZE: int = 16,      # vLLM 分页大小
    MAX_PAGES: int = 4096,     # block_table 行宽
    block_H: int = 64,         # 每 CTA 的头数(=M 维行数)
    num_stages: int = 2, threads: int = 256,
    dtype: str = "bfloat16", accum_dtype: str = "float",
):
    D_Q = D_LAT + D_ROPE                                # 576
    scale = (1.0 / 192) ** 0.5 * 1.44269504             # 1/√192 × log2(e)

    @T.prim_func
    def main(
        Q:         T.Tensor((B, H, D_Q), dtype),        # 吸收后的 Q: [q_latent|q_pe]
        KV_cache:  T.Tensor((num_blocks, BLOCK_SIZE, D_Q), dtype),  # [c(512)|k_pe(64)]
        block_table: T.Tensor((B, MAX_PAGES), "int32"),
        seq_lens:  T.Tensor((B,), "int32"),
        O:         T.Tensor((B, H, D_LAT), dtype),      # 输出(512), 后续模型代码再乘 W_UV
    ):
        with T.Kernel(T.ceildiv(H, block_H), B, threads=threads) as (hb, b):
            Q_s   = T.alloc_shared((block_H, D_Q), dtype)
            K_s   = T.alloc_shared((BLOCK_SIZE, D_Q), dtype)
            V_s   = T.alloc_shared((BLOCK_SIZE, D_LAT), dtype)
            acc_s = T.alloc_fragment((block_H, BLOCK_SIZE), accum_dtype)
            acc_p = T.alloc_fragment((block_H, BLOCK_SIZE), dtype)   # cast 后喂第二个 gemm
            acc_o = T.alloc_fragment((block_H, D_LAT), accum_dtype)
            m_prev = T.alloc_fragment((block_H,), accum_dtype)
            m_new  = T.alloc_fragment((block_H,), accum_dtype)
            alpha  = T.alloc_fragment((block_H,), accum_dtype)
            l_prev = T.alloc_fragment((block_H,), accum_dtype)
            l_new  = T.alloc_fragment((block_H,), accum_dtype)

            kv_len = seq_lens[b]                        # 运行时标量
            T.copy(Q[b, hb * block_H, 0], Q_s)          # 连续 64 个头 × 576 维
            T.fill(m_prev, -1e6)                        # 有限哨兵(非 -inf!)
            T.fill(l_prev, 0)
            T.clear(acc_o)

            # ★ 动态循环上界: 只处理真实长度, 不存在的页绝不访存
            for nb in T.Pipelined(T.ceildiv(kv_len, BLOCK_SIZE), num_stages=num_stages):
                page = block_table[b, nb]
                T.copy(KV_cache[page, 0, 0], K_s)       # (16, 576) 完整行
                T.copy(KV_cache[page, 0, 0], V_s)       # (16, 512) 同起点取前 512 = V=c
                T.clear(acc_s)
                T.gemm(Q_s, K_s, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, BLOCK_SIZE):
                    acc_s[i, j] = T.if_then_else(
                        nb * BLOCK_SIZE + j < kv_len,   # 页内尾部掩码
                        acc_s[i, j] * scale, -1e6)
                # ---- 在线 softmax(同 15.4 手法) ----
                T.reduce_max(acc_s, m_new, dim=1, clear=True)
                for i in T.Parallel(block_H):
                    m_new[i] = T.max(m_prev[i], m_new[i])
                    alpha[i] = T.exp2(m_prev[i] - m_new[i])
                for i, d in T.Parallel(block_H, D_LAT):
                    acc_o[i, d] = acc_o[i, d] * alpha[i]
                for i, j in T.Parallel(block_H, BLOCK_SIZE):
                    acc_s[i, j] = T.exp2(acc_s[i, j] - m_new[i])
                T.reduce_sum(acc_s, l_new, dim=1, clear=True)
                for i in T.Parallel(block_H):
                    l_prev[i] = l_prev[i] * alpha[i] + l_new[i]
                T.copy(acc_s, acc_p)                    # fp32 → bf16
                T.gemm(acc_p, V_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.copy(m_new, m_prev)

            for i, d in T.Parallel(block_H, D_LAT):
                acc_o[i, d] = acc_o[i, d] / l_prev[i]
            T.copy(acc_o, O[b, hb * block_H, 0])
    return main
```

**对拍基准**：无分页小长度场景可与 `torch SDPA`（把 576 当 head_dim）对拍；
生产等价物是 [FlashMLA](https://github.com/deepseek-ai/FlashMLA)。
官方战绩参考：tilelang 的 MLA decode **80 行打平 H100 上的 FlashMLA**（2025-03）；
AMD MI300X 版本**打平 Aiter 汇编内核**（2025-04）——这是"TileLang 工业可用"的最硬证据。

### 16.5.3 集成点与回退关系（接 14.2 内核选择链路）

```
vLLM DeepSeek 注意力入口
   ├─ NVIDIA + has_deep_gemm + FlashMLA 可用 → 官方专属内核(①层, 首选)
   ├─ 昇腾/其他硬件 或 需要定制变体 → 本章 TileLang MLA kernel(②层)
   │      └─ 注册为 custom op(16.8) + warmup 固定 shape(16.8③)
   └─ 都不可用 → FlashInfer/Triton 后端(③层) → SDPA(④层)
```

**FP8 块缩放 GEMM**（DeepSeek FFN 的 128×128 块缩放）属于模式 6 的 W8A8 变体：
在 16.6.4 的 dequant kernel 里把"4bit→fp16"换成"fp8→fp16 + 按 128×128 块取 scale"即可，
结构完全不变。NVIDIA 上 vLLM 会优先 DeepGEMM（15.5⑥ 的双实现回退同款逻辑）。

---

## 16.6 GLM / Qwen 适配实战：GQA / 融合核 / MoE / W4A16

### 16.6.0 结构对照与内核机会

| 模型 | 注意力 | 特有结构 | TileLang 内核机会 |
|---|---|---|---|
| Qwen2.5 | GQA（head_dim=128，KV 头 2~8） | — | varlen prefill（16.6.1） |
| Qwen3 | GQA + **QK-Norm**（q/k 逐头 RMSNorm 再 RoPE） | Qwen3-MoE：128 家激活 8 | ★QK-Norm+RoPE 融合（16.6.2）；MoE（16.6.3） |
| GLM-4 / GLM-4.5 | GQA | **partial RoPE**（部分维度旋转）；GLM-4.5：MoE 160/8、MTP | partial RoPE 改造（16.6.2 注）；MoE（16.6.3） |
| 消费卡量化部署 | — | AWQ/GPTQ W4A16 | dequant GEMM（16.6.4） |

**诚实的工程建议（decode 路线）**：Qwen/GLM 的 GQA group = H/H_KV 只有 2~8，
"head-as-row" 的 M 维太小，TensorCore 吃不满——**decode 注意力直接用 vLLM 内置的
FlashAttention/FlashInfer 后端更优**。TileLang 在这些模型上的价值在
**prefill 变长注意力、逐元素融合、MoE 分组 GEMM、量化 GEMM**。给模型选 kernel 要扬长避短。

### 16.6.1 完整实现：varlen GQA Prefill kernel（Qwen/GLM 通用）

变长打包（batch 内所有序列首尾相接存成 `(Q_total, H, DK)`）+ `cu_seqlens` 偏移 +
因果掩码（含前缀：query 对应 kv 序列的**末尾** q_len 段）。

```python
@tilelang.jit(out_idx=[-1])
def gqa_varlen_prefill(
    Q_total: int, K_total: int,          # 打包总 token 数(编译期常量)
    H: int, H_KV: int, DK: int, DV: int,
    B: int,                              # 序列数
    MAX_Q: int,                          # 最长 query(决定网格)
    block_M: int = 64, block_N: int = 64,
    num_stages: int = 2, threads: int = 256,
    dtype: str = "bfloat16", accum_dtype: str = "float",
):
    group = H // H_KV
    scale = (1.0 / DK) ** 0.5 * 1.44269504

    @T.prim_func
    def main(
        Q: T.Tensor((Q_total, H, DK), dtype),
        K: T.Tensor((K_total, H_KV, DK), dtype),
        V: T.Tensor((K_total, H_KV, DV), dtype),
        cu_q: T.Tensor((B + 1,), "int32"),   # 每 batch 的 q 起止
        cu_k: T.Tensor((B + 1,), "int32"),   # 每 batch 的 kv 起止
        O: T.Tensor((Q_total, H, DV), dtype),
    ):
        with T.Kernel(H, T.ceildiv(MAX_Q, block_M), B, threads=threads) as (h, mb, b):
            Q_s   = T.alloc_shared((block_M, DK), dtype)
            K_s   = T.alloc_shared((block_N, DK), dtype)
            V_s   = T.alloc_shared((block_N, DV), dtype)
            acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_p = T.alloc_fragment((block_M, block_N), dtype)
            acc_o = T.alloc_fragment((block_M, DV), accum_dtype)
            m_prev = T.alloc_fragment((block_M,), accum_dtype)
            m_new  = T.alloc_fragment((block_M,), accum_dtype)
            alpha  = T.alloc_fragment((block_M,), accum_dtype)
            l_prev = T.alloc_fragment((block_M,), accum_dtype)
            l_new  = T.alloc_fragment((block_M,), accum_dtype)

            q_start = cu_q[b]
            q_len   = cu_q[b + 1] - q_start
            k_start = cu_k[b]
            kv_len  = cu_k[b + 1] - k_start
            kv_head = h // group                    # GQA: 多个 Q 头映射同一 KV 头

            # Q 块带掩码装入(块尾越界行填 0; 这些行永不写回)
            for i, d in T.Parallel(block_M, DK):
                Q_s[i, d] = T.if_then_else(mb * block_M + i < q_len,
                                           Q[q_start + mb * block_M + i, h, d], 0)
            T.fill(m_prev, -1e6)
            T.fill(l_prev, 0)
            T.clear(acc_o)

            for nb in T.Pipelined(T.ceildiv(kv_len, block_N), num_stages=num_stages):
                for i, d in T.Parallel(block_N, DK):          # K 块带掩码装入
                    K_s[i, d] = T.if_then_else(nb * block_N + i < kv_len,
                        K[k_start + nb * block_N + i, kv_head, d], 0)
                for i, d in T.Parallel(block_N, DV):
                    V_s[i, d] = T.if_then_else(nb * block_N + i < kv_len,
                        V[k_start + nb * block_N + i, kv_head, d], 0)
                T.clear(acc_s)
                T.gemm(Q_s, K_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # 因果掩码(含前缀偏移): in-range 行 q_pos < kv_len, 故单条件即够
                for i, j in T.Parallel(block_M, block_N):
                    q_pos = (kv_len - q_len) + mb * block_M + i
                    acc_s[i, j] = T.if_then_else(nb * block_N + j <= q_pos,
                                                 acc_s[i, j] * scale, -1e6)
                # ---- 在线 softmax ----
                T.reduce_max(acc_s, m_new, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    m_new[i] = T.max(m_prev[i], m_new[i])
                    alpha[i] = T.exp2(m_prev[i] - m_new[i])
                for i, d in T.Parallel(block_M, DV):
                    acc_o[i, d] = acc_o[i, d] * alpha[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] - m_new[i])
                T.reduce_sum(acc_s, l_new, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    l_prev[i] = l_prev[i] * alpha[i] + l_new[i]
                T.copy(acc_s, acc_p)
                T.gemm(acc_p, V_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.copy(m_new, m_prev)

            for i, d in T.Parallel(block_M, DV):
                acc_o[i, d] = acc_o[i, d] / l_prev[i]
                if mb * block_M + i < q_len:              # 带谓词写回, 不越序列界
                    O[q_start + mb * block_M + i, h, d] = acc_o[i, d]
    return main
```

> 注意 `Q_total/K_total/B/MAX_Q` 都是编译期常量 → 每个 batch 形状组合都会编译一份内核。
> 这正是 16.8 warmup 体系存在的原因（vLLM 的 CUDA Graph capture 尺寸表就是 warmup 清单）。

### 16.6.2 完整实现：QK-Norm + RoPE 融合 kernel（Qwen3 同款，GLM partial RoPE 一行改）

Qwen3 在 RoPE 前对 q/k 做**逐头 RMSNorm**——两次逐元素访存变一次，带宽型融合的标准机会。
kernel 按 token 切 CTA，一次处理该 token 的全部 Q 头与 K 头，就地写回（与 vLLM 的 in-place
rotary 语义一致）。

```python
@tilelang.jit
def fused_qk_norm_rope(
    T_tok: int, H: int, H_KV: int, DK: int, MAX_POS: int,
    eps: float = 1e-6, threads: int = 128, dtype: str = "bfloat16",
):
    HALF = DK // 2

    @T.prim_func
    def main(
        Q: T.Tensor((T_tok, H, DK), dtype),          # 就地修改
        K: T.Tensor((T_tok, H_KV, DK), dtype),
        qw: T.Tensor((DK,), dtype),                  # q-norm 逐通道权重
        kw: T.Tensor((DK,), dtype),                  # k-norm 逐通道权重
        cos_cache: T.Tensor((MAX_POS, HALF), "float"),   # 预计算 rope 表
        sin_cache: T.Tensor((MAX_POS, HALF), "float"),
        positions: T.Tensor((T_tok,), "int32"),
    ):
        with T.Kernel(T_tok, threads=threads) as (t):
            q_f  = T.alloc_fragment((H, DK), "float")
            k_f  = T.alloc_fragment((H_KV, DK), "float")
            tmp  = T.alloc_fragment((H, DK), "float")      # 平方暂存(不破坏原值)
            tmpk = T.alloc_fragment((H_KV, DK), "float")
            sq   = T.alloc_fragment((H,), "float")
            sk   = T.alloc_fragment((H_KV,), "float")
            invq = T.alloc_fragment((H,), "float")
            invk = T.alloc_fragment((H_KV,), "float")

            pos = positions[t]
            T.copy(Q[t, 0, 0], q_f)          # shared↔fragment copy 兼做 fp32 转换
            T.copy(K[t, 0, 0], k_f)

            # ---- Q: 逐头 RMSNorm ----
            for h, d in T.Parallel(H, DK):
                tmp[h, d] = q_f[h, d] * q_f[h, d]
            T.reduce_sum(tmp, sq, dim=1, clear=True)
            for h in T.Parallel(H):
                invq[h] = 1.0 / T.sqrt(sq[h] / DK + eps)
            for h, d in T.Parallel(H, DK):
                q_f[h, d] = q_f[h, d] * invq[h] * qw[d]
            # ---- Q: RoPE(NEOX 对半拆分) ----
            for h, d in T.Parallel(H, HALF):
                c = cos_cache[pos, d]; s = sin_cache[pos, d]
                x1 = q_f[h, d]; x2 = q_f[h, d + HALF]
                q_f[h, d]        = x1 * c - x2 * s
                q_f[h, d + HALF] = x2 * c + x1 * s
            T.copy(q_f, Q[t, 0, 0])          # 写回(隐式 fp32→bf16)

            # ---- K: 同上(H_KV 行) ----
            for h, d in T.Parallel(H_KV, DK):
                tmpk[h, d] = k_f[h, d] * k_f[h, d]
            T.reduce_sum(tmpk, sk, dim=1, clear=True)
            for h in T.Parallel(H_KV):
                invk[h] = 1.0 / T.sqrt(sk[h] / DK + eps)
            for h, d in T.Parallel(H_KV, DK):
                k_f[h, d] = k_f[h, d] * invk[h] * kw[d]
            for h, d in T.Parallel(H_KV, HALF):
                c = cos_cache[pos, d]; s = sin_cache[pos, d]
                x1 = k_f[h, d]; x2 = k_f[h, d + HALF]
                k_f[h, d]        = x1 * c - x2 * s
                k_f[h, d + HALF] = x2 * c + x1 * s
            T.copy(k_f, K[t, 0, 0])
    return main
```

> **复用提示**：Q/K 两段是复制粘贴级同构，落地时可抽成 `norm_rope(frag, weight)` 辅助函数；
> TileLang 的 `T.alloc_*` 在工厂闭包内声明、`T.Parallel` 循环可参数化，重构零成本。

**GLM partial RoPE 改法**：GLM 系只有前 `rotary_dim` 个通道参与旋转（且部分版本用 GPT-J 相邻配对
`(2i, 2i+1)` 而非 NEOX 对半）。改法就是把 RoPE 循环限幅并用 `T.if_then_else` 让
`d >= rotary_dim/2` 的通道保持原值，配对方式按模型 config 对应实现。

**成功指标**：融合前 RMSNorm、RoPE 两次读写，融合后一次；有效带宽对照 15.3 的 GB/s 标准。

### 16.6.3 完整实现：MoE 分组 GEMM（Qwen3-MoE / GLM-4.5 / DeepSeek 通用）

前置（vLLM `moe_align_block_size` 已做）：把 token 按专家**排序**并把总行数 pad 到
`block_M` 整数倍，产出一个"每行块 → 专家"的映射表。kernel 与普通 GEMM 的唯一区别：
**权重按行块查表选择**——却因此把 256 次（DeepSeek）小 GEMM 合成了一次大网格。

```python
@tilelang.jit(out_idx=[-1])
def moe_grouped_gemm(
    M_pad: int, N: int, K: int, E: int,   # M_pad = pad 后的排序 token 行数
    block_M: int = 64, block_N: int = 128, block_K: int = 64,
    num_stages: int = 3, threads: int = 128,
    dtype: str = "bfloat16", accum_dtype: str = "float",
):
    @T.prim_func
    def main(
        X_sorted:   T.Tensor((M_pad, K), dtype),
        W:          T.Tensor((E, N, K), dtype),              # 每专家权重
        expert_ids: T.Tensor((M_pad // block_M,), "int32"),  # 每行块 → 专家
        O:          T.Tensor((M_pad, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M_pad, block_M),
                      threads=threads) as (bx, bm):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
            e = expert_ids[bm]                    # ★ 与普通 GEMM 的唯一差异
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(X_sorted[bm * block_M, k * block_K], A_shared)
                T.copy(W[e, bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, O[bm * block_M, bx * block_N])
    return main
```

生产扩展（按 16.4 模式 1→6 组合）：
- **gate_up 融合激活**：`N` 维即 `2*moe_intermediate`，在写回前对两半做 SiLU×mul（模式 2 手法）；
- **三段合一**：gate_up GEMM + act + down GEMM 融合进一个 kernel（中间结果留 fragment/shared，
  省两次全局往返）——vLLM `fused_moe` 的 TileLang 化路线；
- **FP8 块缩放**：16.5.3 的说明同样适用（DeepSeek/Qwen3-FP8 权重）。

### 16.6.4 完整实现：W4A16 dequant GEMM（AWQ/GPTQ 量化模型跑消费卡）

把"反量化 + GEMM"融进一个 kernel：权重以 4bit 打包驻留显存，块内解包升精度后直接 TensorCore。

```python
@tilelang.jit(out_idx=[-1])
def w4a16_gemm(
    M: int, N: int, K: int,
    group_size: int = 128,               # 教学版约束: block_K == group_size
    block_M: int = 64, block_N: int = 64,
    num_stages: int = 2, threads: int = 128,
    dtype: str = "float16", accum_dtype: str = "float",
):
    @T.prim_func
    def main(
        A:     T.Tensor((M, K), dtype),
        B_q:   T.Tensor((N, K // 8), "int32"),                  # 4bit×8 打包
        scales:T.Tensor((N, K // group_size), dtype),
        O:     T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            A_shared  = T.alloc_shared((block_M, group_size), dtype)
            Bq_shared = T.alloc_shared((block_N, group_size // 8), "int32")
            B_deq     = T.alloc_shared((group_size, block_N), dtype)  # 解包后的 B tile
            sc_shared = T.alloc_shared((block_N,), dtype)
            C_local   = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for kb in T.Pipelined(T.ceildiv(K, group_size), num_stages=num_stages):
                T.copy(A[by * block_M, kb * group_size], A_shared)
                T.copy(B_q[bx * block_N, kb * (group_size // 8)], Bq_shared)
                T.copy(scales[bx * block_N, kb], sc_shared)
                # ★ 逐元素解包: int32 里每 4bit 一个权重, 对称量化零点=8
                for kk, nn in T.Parallel(group_size, block_N):
                    w4 = (Bq_shared[nn, kk >> 3] >> ((kk & 7) << 2)) & 0xF
                    B_deq[kk, nn] = (w4 - 8) * sc_shared[nn]   # int→float 自动升 fp32, 存回隐式转换
                T.gemm(A_shared, B_deq, C_local)
            T.copy(C_local, O[by * block_M, bx * block_N])
    return main
```

- AWQ 的 asymmetric 变体还要减 zero-point（打包在权重的另一 4bit 字段），官方
  `examples/dequantize_gemm` 有完整处理；group_size ≠ block_K 的通用情形按块内分段处理。
- 官方 [TileOPs](https://github.com/tile-ai/TileOPs) 已有 W4A16 长 K 流式版本，**先抄再用**。

---

## 16.7 跨硬件适配体系：同一份 kernel 源码跑 NVIDIA / AMD / 昇腾

### 16.7.1 总架构：回退链 + 平台门卫（全系列 12/14/15 章的原则收敛）

```
调用点(attention / mlp / norm)
   │ ① 专属极致: DeepGEMM / FlashMLA / Aiter / DeepSeek 官方内核族     ← 硬件绑定的 C++/汇编
   │    条件: has_deep_gemm() / 平台白名单 / shape 命中
   ├─ ② 通用优化: 你的 TileLang kernel                                 ← 一份源码, 多硬件
   │    条件: has_tilelang() and 平台受支持(16.1.2 矩阵)
   ├─ ③ 兜底: Triton / FlashInfer / vLLM 内置后端
   └─ ④ 最后回退: torch 原生(SDPA / matmul)
```

平台门卫 + 延迟导入的教科书写法（14.3 四条军规的内核版）：

```python
# my_ops/__init__.py
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

_TILELANG_ENABLED = has_tilelang() and (
    current_platform.is_cuda() or current_platform.is_rocm())

if _TILELANG_ENABLED:
    import tilelang                      # ← 只在门卫通过后才 import(第13/14章军规)
    import tilelang.language as T
```

### 16.7.2 PassConfig 按硬件切换

```python
def get_platform_pass_configs() -> dict:
    """激进优化 ≠ 万金油: 按目标硬件裁剪编译器(与 vLLM 15.5② 同构)"""
    import tilelang
    cfg = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    if current_platform.is_cuda():
        cfg[tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL] = 10
    elif current_platform.is_rocm():
        pass   # MI300X: AsyncCopy 路径由编译器自动选择, 无需额外开关
    return cfg

# 注入方式: 工厂函数 JIT 时传入
kernel = tilelang.jit(out_idx=[-1], pass_configs=get_platform_pass_configs())(_factory)(args...)
```

### 16.7.3 昇腾 NPU 路线（两条，按芯片代际选）

| 路线 | 芯片 | 做法 |
|---|---|---|
| **主线官方后端**（推荐起点） | 昇腾 950 | TileLang **v0.1.15+** 内置：原生代码生成、自动调度与同步、SIMD/SIMT 向量编程；官方提供 GEMM / FlashAttention 示例。kernel 源码与 NVIDIA 版**基本一致**，差异交给编译器 |
| 社区仓库 | 910B 等 | [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend)：`ascendc_pto`（AscendC 路线）/ `npuir`（NPU IR 路线）两分支，预览级 |

昇腾上的工程注意：
1. 与 **vllm-ascend** 协同时（第 12 章⑤），平台判定以 vllm-ascend 注册的 Platform 为准，
   门卫写成 `getattr(current_platform, "device_type", None) == "npu"` 风格的防御式检查；
2. 优先跑官方示例验证工具链（GEMM/FA），再上自研 kernel；Cube 单元的 tile 尺寸约束与
   NVIDIA 不同，`block_M/N/K` 扫参表要按 NPU 重扫（16.9 sweep 脚本直接复用）；
3. 昇腾暂无 DeepGEMM 等专属层 → **TileLang 在这里的地位从"②层"升为"首选优化层"**，
   这正是跨硬件策略的价值：同一份工程，缺谁补谁。

### 16.7.4 AMD 与 Apple

- AMD：MI250/MI300X 主线支持（MatrixCore/AsyncCopy 自动映射）；MLA 在 MI300X 打平 Aiter
  （16.5.2）；ROCm 下 PassConfig 无需 TMA/warp-spec 项（本来就不走）；
- Apple：Metal 后端可用作端侧原型，M5 起有 cooperative tensor GEMM；生产推理不在主战场。

### 16.7.5 CI 测试矩阵（工业必备）

```text
shape 轴:  decode(B∈{1,8,64}; S∈{1K,32K}) × prefill(T∈{512,4K,32K})
dtype 轴:  fp16 / bf16 / fp8-block / w4a16
硬件轴:    A100 / H100 / 4090 / MI300X / 昇腾950(或910B)
每格:      数值对拍(16.9.3) + 延迟 + 带宽/TFLOPS  → 一张 CSV 报告
```

---

## 16.8 vLLM 生产集成手册（六件事的完整代码）

15.5 列了六件事的"是什么"，这里给"怎么写"。目标结构（作为 vLLM 插件包，入口点注册见第 13 章）：

```
my_vllm_tilelang_ops/
├── pyproject.toml                 # vllm.general_plugins 入口(第13章)
└── my_vllm_tilelang_ops/
    ├── __init__.py                # 平台门卫 + register()
    ├── kernels.py                 # kernel 工厂 + Deferred 缓存
    ├── ops.py                     # custom op 注册(①)
    ├── backend.py                 # (可选)注意力后端级集成, 见 ⑧
    └── warmup.py                  # 启动期编译预热(③)
```

### ① custom op 注册（torch.compile / CUDA Graph 的通行证）

```python
# ops.py
import torch
from vllm.utils.torch_utils import direct_register_custom_op
from .kernels import get_rmsnorm_kernel, TILELANG_ENABLED

def _tilelang_rmsnorm_impl(input: torch.Tensor, weight: torch.Tensor,
                           out: torch.Tensor) -> None:
    kernel = get_rmsnorm_kernel(input.shape[0], input.shape[1], input.dtype)
    kernel(input, weight, out)          # 生产版: 输出作 prim_func 参数直接写入(零拷贝)

_OP_REGISTERED = False

def register_tilelang_ops() -> bool:
    global _OP_REGISTERED
    if _OP_REGISTERED or not TILELANG_ENABLED:
        return False
    direct_register_custom_op(
        op_name="tilelang_rmsnorm",
        op_func=_tilelang_rmsnorm_impl,
        mutates_args=["out"],           # 声明就地修改 → 图内合法, 不打断 CUDA Graph
    )
    _OP_REGISTERED = True
    return True

def rmsnorm(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    register_tilelang_ops()
    if _OP_REGISTERED:
        out = torch.empty_like(input)
        torch.ops.vllm.tilelang_rmsnorm(input, weight, out)
        return out
    # ④层回退: 纯 torch
    inv = torch.rsqrt(input.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return (input.float() * inv).to(input.dtype) * weight.float()
```

**为什么必须注册**：未注册的 Python 侧 TileLang 调用在 `torch.compile` 看来是黑盒副作用 →
图断裂；在 CUDA Graph 捕获里则是非法操作。注册后它成为原子算子，与 15.5⑤ 一致。

### ② kernel 工厂 + Deferred 缓存（kernels.py 骨架）

```python
_CACHE: dict = {}
TILELANG_ENABLED = False
try:
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import has_tilelang
    TILELANG_ENABLED = has_tilelang() and current_platform.is_cuda()
except Exception:
    pass
if TILELANG_ENABLED:
    import tilelang
    import tilelang.language as T

def get_rmsnorm_kernel(num_tokens: int, hidden: int, dtype: torch.dtype):
    """(num_tokens, hidden, dtype) 是编译期常量 → 同键复用, 异键新编"""
    key = (num_tokens, hidden, dtype)
    if key not in _CACHE:
        assert TILELANG_ENABLED
        dt = "bfloat16" if dtype == torch.bfloat16 else "float16"

        @tilelang.jit
        def _factory(rows: int, cols: int, dt: str):
            @T.prim_func
            def main(X: T.Tensor((rows, cols), dt),
                     W: T.Tensor((cols,), dt),
                     Y: T.Tensor((rows, cols), dt)):       # ★ 输出由调用方传入 → 零拷贝
                with T.Kernel(rows, threads=256) as (r):
                    x_f  = T.alloc_fragment((cols,), "float")
                    sqr  = T.alloc_fragment((cols,), "float")
                    red  = T.alloc_fragment((1,), "float")
                    T.copy(X[r, 0], x_f)
                    for d in T.Parallel(cols):
                        sqr[d] = x_f[d] * x_f[d]
                    T.reduce_sum(sqr, red, dim=0, clear=True)
                    inv = 1.0 / T.sqrt(red[0] / cols + 1e-6)
                    for d in T.Parallel(cols):
                        x_f[d] = x_f[d] * inv * W[d]
                    T.copy(x_f, Y[r, 0])
            return main
        _CACHE[key] = _factory(num_tokens, hidden, dt)
    return _CACHE[key]
```

> **零拷贝要点**：教学版 `@tilelang.jit(out_idx=[-1])` 让 tilelang 内部分配输出，wrapper 里还得
> `out.copy_(result)` 白付一次 D2D；生产版把输出张量作为 `prim_func` 参数（如上）直接写入
> vLLM 预分配的 buffer。

### ③ 启动期 warmup（对齐 vLLM `jit_warmup_tilelang_helper`）

```python
# warmup.py —— 只编译不执行, 把 JIT 开销从首个真实请求挪到启动(与 CUDA Graph 预录制同哲学)
from .kernels import get_rmsnorm_kernel

def warmup(cuda_graph_sizes=(1, 2, 4, 8, 16, 32), hidden: int = 4096):
    """★ capture 尺寸表就是 warmup 清单: CUDA Graph 按这几个 M 录图, kernel 就按这几个 M 编译"""
    import torch
    for m in cuda_graph_sizes:
        get_rmsnorm_kernel(m, hidden, torch.bfloat16)
```

编译产物落盘复用：TileLang v0.1.13+ 支持跨主机二进制缓存——把缓存目录打进镜像或挂载卷，
多机同型集群只需编译一次。

### ④⑤⑥ PassConfig 注入 / 平台门卫 / 回退链

分别见 16.7.2、16.7.1、16.8① 的回退写法——三个模式拼起来即完整工业位形。

### ⑦ 完整集成配置示例：一个可落地的 TileLang 插件包

第 13 章讲过 vLLM 的三类入口点，**内核类插件只需要 `vllm.general_plugins`**
（所有进程加载、必须幂等——重复注册由 ops.py 的 `_OP_REGISTERED` 门闩兜住）。

**`pyproject.toml`**（打包与发现配置）：

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "my-vllm-tilelang-ops"
version = "0.1.0"
description = "TileLang 高性能算子插件（教程第 16 章配套）"
requires-python = ">=3.10"
dependencies = ["vllm", "tilelang>=0.1.13"]   # ★ tilelang 声明为硬依赖, has_tilelang() 才可靠为 True

# vLLM 通过该 entry-point 组发现插件; 指向零参函数, vLLM 在每个进程调用一次
[project.entry-points."vllm.general_plugins"]
tilelang_ops = "my_vllm_tilelang_ops:register"

[tool.setuptools.packages.find]
include = ["my_vllm_tilelang_ops*"]
```

**`__init__.py`**（插件入口：平台门卫 + op 注册 + 环境变量配置 + 可选 warmup + 模型接线）：

```python
# -*- coding: utf-8 -*-
"""TileLang 算子插件入口。
vLLM 在 process0 / EngineCore / 每个 Worker 进程各调用一次 register()
→ 必须幂等; warmup 只在持有 GPU 的 worker 里真正编译, 无 GPU 进程静默跳过。
"""
import logging
import os

logger = logging.getLogger(__name__)


def register():
    from my_vllm_tilelang_ops import ops, warmup as warmup_mod
    from vllm.platforms import current_platform

    # ---- 平台门卫(16.7.1): 仅 CUDA 启用; 昇腾/AMD 的扩展写法见 16.7.3/16.7.4 ----
    if not (current_platform.is_cuda() and ops.register_tilelang_ops()):
        logger.info("[tilelang_ops] 平台不启用 TileLang, 走内置回退链(16.7.1)")
        return

    # ---- 配置项: 环境变量驱动(风格对齐 vLLM 自身的环境变量约定) ----
    # MY_TILELANG_WARMUP: 逗号分隔的 warmup token 数; "0" = 关闭; 默认对齐 CUDA Graph 尺寸
    warmup_env = os.environ.get("MY_TILELANG_WARMUP", "1,2,4,8,16,32")
    if warmup_env != "0":
        sizes = tuple(int(s) for s in warmup_env.split(","))
        hidden = int(os.environ.get("MY_TILELANG_HIDDEN", "4096"))
        try:
            warmup_mod.warmup(sizes, hidden)
            logger.info("[tilelang_ops] warmup 完成: sizes=%s hidden=%d", sizes, hidden)
        except Exception:
            # 引擎/前端进程没有 GPU 属预期 → 运行期 JIT 兜底, 不能让插件加载失败
            logger.warning("[tilelang_ops] warmup 跳过(本进程无 GPU 或编译失败)")

    # ---- 模型接线(可选, MY_TILELANG_PATCH_MODELS=1 打开) ----
    if os.environ.get("MY_TILELANG_PATCH_MODELS") == "1":
        patch_target_rmsnorm()
```

**模型接线**（被 `register()` 调用；两种深度，原型用 A、上线用 B）：

```python
# 方式 A: 猴子补丁 —— 无侵入, 快速验证收益(注意守好残差分支!)
def patch_target_rmsnorm():
    from my_vllm_tilelang_ops import ops
    from vllm.model_executor.layers.layernorm import RMSNorm   # 路径以 v0.31 源码为准

    _orig_forward = RMSNorm.forward_cuda
    def forward(self, x, residual=None):
        if residual is not None:          # 残差融合分支语义不同, 保守回退原实现
            return _orig_forward(self, x, residual)
        return ops.rmsnorm(x, self.weight)   # ← 注册过的 custom op, 图内合法
    RMSNorm.forward_cuda = forward

# 方式 B: 自定义模型 + ModelRegistry(第 10 章正路; 在插件 general 里一行接入):
#   from vllm.model_executor.models.registry import ModelRegistry
#   ModelRegistry.register_model("MyQwen3ForCausalLM", "my_ops.models:MyQwen3ForCausalLM")
#   模型代码的 attention/MLP 中直接调用 torch.ops.vllm.tilelang_xxx(...)
```

**启动配置**（容器内，命令风格与第 02/09 章一致）：

```bash
pip install -e /path/to/my-vllm-tilelang-ops   # 安装即被发现, general 类无需 VLLM_PLUGINS 白名单

# 配置: hidden 与 warmup 尺寸都要和目标模型/启动参数对齐
export MY_TILELANG_HIDDEN=4096                  # Qwen3-8B: 4096 / GLM-4.5: 4096+ / DeepSeek-V3: 7168
export MY_TILELANG_WARMUP=1,2,4,8,16,32,64,128  # ★ 与 --max-cuda-graph-sizes 完全一致
export MY_TILELANG_PATCH_MODELS=1               # 打开 RMSNorm 替换(先用注释态跑基线)

vllm serve Qwen/Qwen3-8B --max-model-len 8192 \
  --max-cuda-graph-sizes 1,2,4,8,16,32,64,128   # 与 warmup 清单一一对应(16.8③ 的铁律)
```

两点强调：
- `dependencies` 写明 `tilelang` 后，插件的 `has_tilelang()` 才可靠为 True——对你自己的插件
  而言"可选依赖"应升级为硬依赖（版本对齐表见 16.1.3）；
- general 插件在**所有进程**执行，warmup 只在持 GPU 的 worker 里生效；其他进程的失败必须
  被 try/except 吞掉，否则会拖垮整个 vLLM 启动（第 13 章幂等军规的内核版）。

### ⑧ 注意力后端级集成：让 DeepSeek 用上你的 MLA kernel（进阶）

16.5.2 的 MLA kernel 想真正发挥威力，要接进 vLLM 的注意力后端选择链（第 14 章 14.2），
而不只是替换一个 RMSNorm。集成分三层深度，按收益/成本递增：

| 深度 | 做法 | 工程量 | 适用算子 |
|---|---|---|---|
| 算子级（⑦） | 补丁/自定义模型中调 custom op | 低 | RMSNorm / RoPE / MoE GEMM / W4A16 |
| 后端级（本节） | 自定义 AttentionBackend + 接入选择链 | 中 | MLA / GQA decode 等核心注意力 |
| 运行时级 | TileRT 式 V1 connector（PD 分离，vLLM 管其余、它管解码） | 高 | 整段解码接管 |

自定义后端骨架（V1 架构；类名/方法签名以 v0.31 源码 `vllm/v1/attention_backends/` 为准）：

```python
# backend.py —— 最小骨架: 只接管 decode 路径, 其余显式路由到内置后端(回退链 ②→③)
import torch
from vllm.v1.attention_backends.backend import AttentionBackend   # 路径以源码为准


class TileLangMLABackend(AttentionBackend):
    @staticmethod
    def get_supported_dtypes():
        return (torch.bfloat16,)

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
        # 与 16.5.2 kernel 严格对齐: MLA cache 每行 576 = [c(512) | k_pe(64)]
        return (num_blocks, block_size, head_size)

    def forward_decode(self, q, kv_cache, attn_metadata, **kwargs):
        # 命中 16.5.2 的 mla_paged_decode(shape/dtype 命中时)
        return self._kernels.mla_decode(
            q, kv_cache,
            attn_metadata.block_table,      # (B, MAX_PAGES) int32
            attn_metadata.seq_lens)         # (B,) int32

    # forward_prefill / forward_insert 等其余方法: 委托给内置 FlashMLA/FlashInfer
    # —— "只做我们最快的那段" 是回退链在注意力层的落点
```

注册与选择（与 ⑦ 的 `register()` 合流）：

```python
# 插件 __init__.py 追加(注册 API 以 v0.31 backend registry 源码为准):
#   from vllm.v1.attention_backends.registry import register_backend
#   register_backend("TILELANG_MLA", TileLangMLABackend)

# 启动时强制选择(vLLM 官方开关):
export VLLM_ATTENTION_BACKEND=TILELANG_MLA
```

工业建议：后端级接入的收益/成本比通常低于算子级——**先用 ⑦ 的算子级集成跑第 9 章压测，
确认收益存在，再决定是否下沉**。且必须保住回退语义：shape/dtype 不命中时显式路由到
FlashMLA/FlashInfer（16.7.1 回退链），不允许"半支持"状态在线上裸奔。

### ⑨ Docker / 集群级配置

```dockerfile
# 在 16.1.1 的基础镜像上追加(生产集成镜像)
COPY my-vllm-tilelang-ops /opt/my-vllm-tilelang-ops
RUN pip install -e /opt/my-vllm-tilelang-ops

# ★ 预热进镜像: 构建期把 warmup 清单全部编译进 JIT 缓存 → 运行期零编译
#   (需带 GPU 的构建机; 或改在容器入口脚本 preheat)
RUN python -c "from my_vllm_tilelang_ops.warmup import warmup; \
    warmup((1,2,4,8,16,32,64,128), hidden=4096)"

ENV MY_TILELANG_WARMUP=1,2,4,8,16,32,64,128 \
    MY_TILELANG_HIDDEN=4096
VOLUME ["/root/.tilelang/cache"]     # TileLang JIT 缓存目录: 挂卷后同型节点扩容零编译(16.2.5)
```

- 缓存卷化后，集群扩容/多机分发**不再重复编译**（16.8③ 的跨主机二进制缓存）；
- 构建期预热需要 nvidia container toolkit 的带卡构建机；无卡 CI 环境跳过该步，靠运行期 JIT 兜底；
- 灰度路径：先 1 台开 `MY_TILELANG_PATCH_MODELS=1` 压测对比 → 通过 16.7.5 CI 矩阵 → 全量。

### 验收集成成功

```bash
# 0) 插件被发现: entry point 已注册(应输出包含 tilelang_ops)
python -c "from importlib.metadata import entry_points; \
  print([e.name for e in entry_points(group='vllm.general_plugins')])"
# 1) 图完整性: torch.compile 下无 graph break 日志
# 2) 端到端: vllm serve 起服务, 第 9 章压测脚本对比 TTFT/TPOT
# 3) 日志确认 warmup 命中(无运行期编译)
```

---

## 16.9 性能工程与精度保障

### 16.9.1 工具链全景

| 阶段 | 工具 | 用途 |
|---|---|---|
| 写前估算 | TileSight | tile 中心分析式性能模型，预估耗时/缓存流量 |
| 编译诊断 | LSP、source-aware diagnostics、IR Lower Trace | 看布局推导结果、逐 pass 检查 |
| kernel 内调试 | `T.print` | 打印中间 buffer（官方 debug 教程） |
| 生成质量 | `kernel.get_kernel_source()` | 检查生成的 CUDA/汇编（有无 TMA、流水深度） |
| GPU 剖析 | ncu（第 14 章）、IKET profiler（CUDA 时间线） | bank conflict / 占用率 / 溢出 |
| 编译期开销 | pass timing（PR #2622） | 定位慢编译 pass |

### 16.9.2 参数 sweep 脚本模板（一劳永逸）

```python
# lab/tilelang/04_sweep_gemm.py —— 把 15.7 的"实验 15-4"脚本化
import itertools, torch, tilelang
from importlib import import_module
gemm = import_module("01_gemm_tilelang")           # 15 章 lab 的 kernel 工厂

def bench(fn, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters * 1e3        # us

M = N = K = 4096
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
rows = []
for bm, bn, bk, st in itertools.product(
        (64, 128, 256), (64, 128, 256), (32, 64, 128), (2, 3, 4)):
    tag = f"bm={bm},bn={bn},bk={bk},st={st}"
    try:
        k = gemm.matmul(M, N, K, block_M=bm, block_N=bn, block_K=bk, num_stages=st)
        t = bench(lambda: k(a, b))
        rows.append((tag, t, 2*M*N*K/(t*1e-6)/1e12))
    except Exception as e:                          # smem 超限/寄存器溢出等 → 记录跳过
        rows.append((tag, -1, f"FAIL:{type(e).__name__}"))
for tag, t, x in sorted(rows, key=lambda r: -(r[2] if isinstance(r[2], float) else 0)):
    print(f"{tag:36s} {t:9.1f}us {x}")
```

扫参顺序按影响从大到小：`block_M/N/K` → `num_stages` → `threads` → swizzle →（最后）`annotate_layout`。

### 16.9.3 精度对拍体系（上线前必过）

三层对拍：**单算子 → 子图（norm+rope+attn）→ 端到端（vLLM 跑通 + 输出分布 sanity）**。

| 精度路径 | 对拍基准 | 容差起点 | 备注 |
|---|---|---|---|
| fp16 GEMM | cuBLAS | rtol=atol=1e-2 | 15.2 同款 |
| bf16 逐元素/融合核 | torch 实现（fp32 中间） | rtol=atol=1.6e-2 | bf16 精度宽 |
| 注意力 | torch SDPA | rtol=1e-3 / atol=1e-2 | softmax 后幅值有界 |
| fp8/w4a16 | 逐层 dequant 参考实现 + 端到端 ppl | 单层 1e-1 量级，端到端看 ppl 漂移 | 按模型校准 |
| 必测边界 | 空序列、S=1、seqlen=block 整数倍±1、全掩码行 | NaN/Inf 零容忍 | 哨兵值方案(16.4)的回归测试 |

---

## 16.10 工业 Checklist 与排错 FAQ

**上线 Checklist**（15.6 的超集）：

- [ ] 回退链完整（①专属 → ②TileLang → ③Triton/内置 → ④torch），每层有平台门卫
- [ ] custom op 注册 + `mutates_args` 正确；torch.compile 无 graph break
- [ ] warmup 覆盖全部 CUDA Graph capture 尺寸；运行期零 JIT 编译
- [ ] PassConfig 与目标硬件绑定（16.7.2）
- [ ] 三层精度对拍通过（16.9.3），边界 shape 含 NaN 检查
- [ ] sweep 过 tile/stages；ncu 无恶性 bank conflict / 寄存器溢出
- [ ] CI 矩阵（16.7.5）至少覆盖目标硬件 × 主力 dtype
- [ ] 二进制缓存/预编译分发方案就位（16.8③）

**FAQ·编译失败**

| 症状 | 原因 | 处方 |
|---|---|---|
| shared memory exceeded | tile×stages 太大 | 减 block 或 num_stages；看 `get_kernel_source` 查缓冲 |
| TMA lowering 报错 | Hopper 新路径不稳 | `TL_DISABLE_TMA_LOWER=True`（vLLM 同款） |
| warp-specialize 崩溃 | 同上 | `TL_DISABLE_WARP_SPECIALIZED=True` |
| 大量寄存器 spill | fragment 过大 / 策略不当 | 换 GemmWarpPolicy、减 block、调 `TL_PTXAS_REGISTER_USAGE_LEVEL` |
| shape 整除断言失败 | gemm 维度 <16 或不整除 | pad 到 ≥16；`T.ceildiv`+掩码（16.6.1 模板） |

**FAQ·精度**

| 症状 | 原因 | 处方 |
|---|---|---|
| 输出 NaN | `-inf` 掩码自减 | 改有限哨兵 `-1e6`（16.4） |
| 长序列精度漂移 | fp16 累加 | 累加器必须 fp32（`accum_dtype="float"`） |
| attention 与 SDPA 有偏 | 忘了 scale 预乘 log2e / 掩码边界 | 检查 `* 1.44269504` 与页尾掩码 |
| 量化模型离群 | zero-point 处理错 | 对照官方 `examples/dequantize_gemm`，区分对称/asymmetric |

**FAQ·性能**

| 症状 | 原因 | 处方 |
|---|---|---|
| TFLOPS <50% cuBLAS | tile/流水没调对 | 按 16.9.2 顺序 sweep；开 swizzle |
| decode 不如内置后端 | group 小，M 维不足 | 认清战场：decode 走官方，TileLang 做 prefill/融合（16.6.0） |
| 每请求首次调用卡顿 | 运行期 JIT | warmup 清单补 shape（16.8③） |
| e2e 无提升 | 瓶颈不在该算子 | 回 14.5 先量化再动手 |

**FAQ·集成**

| 症状 | 原因 | 处方 |
|---|---|---|
| `has_tilelang()` False | 包没装/环境不符 | 16.1.1 重装并验证 |
| graph break | 未注册 custom op | 16.8① |
| CUDA Graph 捕获报错 | kernel 内部分配/同步 | 输出改由调用方传入（零拷贝版） |
| 多进程 worker 重复编译 | 缓存不共享 | 二进制缓存目录挂载/进镜像（16.8③） |

**FAQ·昇腾 NPU**

| 症状 | 原因 | 处方 |
|---|---|---|
| 后端找不到 | 版本不含 Ascend | 950→主线 v0.1.15+；910B→tilelang-ascend 仓库 |
| 算子行为差异 | Cube tile 约束不同 | 按 NPU 重扫 tile 参数（16.9.2） |
| 与 vllm-ascend 冲突 | 平台判定写死 CUDA | 16.7.3 的防御式门卫 |

---

## 16.11 实验总表 + 自测 + 参考资料

**实验**（本章 kernel 代码均为完整可落地版本，存入 `lab/tilelang/` 即可跑）：

| 实验 | 落地文件（建议名） | 学习点 | 完成标准 |
|---|---|---|---|
| 🔬 16-1 MLA paged decode | `05_mla_decode.py` | 分页+吸收+动态循环 | 与 FlashMLA/SDPA 对拍；记录 H100/4060 成绩 |
| 🔬 16-2 varlen GQA prefill | `06_gqa_varlen.py` | 变长+因果+谓词写回 | 与 SDPA 等价（含前缀场景） |
| 🔬 16-3 QK-Norm+RoPE 融合 | `07_qk_norm_rope.py` | 融合减访存 | 带宽 ≥70% 理论值，数值一致 |
| 🔬 16-4 MoE grouped GEMM | `08_moe_grouped.py` | 专家按块查表 | 对比逐专家循环的加速比 |
| 🔬 16-5 W4A16 dequant | `09_w4a16.py` | 位运算解包 | 对拍 dequant 参考实现 |
| 🔬 16-6 vLLM 插件集成 | 16.8 模板建包 | custom op/warmup | vLLM e2e 跑通 + 无 graph break |
| 🔬 16-7 昇腾验证 | 16.7.3 路线 | 同源码跨硬件 | 任一 NPU 上 GEMM/FA 出数并对拍 |

**自测**：

1. MLA 的 KV cache 为什么是 576 维？吸收前后 softmax scale 为什么不变？
2. `T.Pipelined` 的循环上界用运行时标量（`seq_lens[b]`）在什么场景是必需的？编译器如何配合？
3. 为什么全章用 `-1e6` 而非 `-inf` 做掩码？写出会出 NaN 的具体表达式。
4. head-as-row 技巧解决什么问题？为什么对 Qwen decode 反而不适用？
5. `mutates_args` 写错会发生什么（从 CUDA Graph 与 torch.compile 两个角度）？
6. vLLM 为什么在生产 PassConfig 里关 warp-specialized 与 TMA lowering？你的 H100 上何时该开？
7. 昇腾 950 与 910B 的 TileLang 路线有何不同？在 NPU 上 TileLang 的生态位为什么更重要？
8. 二进制跨主机缓存对集群部署意味着什么？

**参考资料**：

- 官方文档 [tilelang.com](https://tilelang.com/) ｜ 仓库 [tile-ai/tilelang](https://github.com/tile-ai/tilelang)
  ｜ 示例库 [examples/](https://github.com/tile-ai/tilelang/tree/main/examples)（GEMM / FlashAttention /
  deepseek_mla / dequantize_gemm / linear_attention / native_sparse_attention）
- 生态：[TileOPs](https://github.com/tile-ai/TileOPs) ｜ [TileRT](https://github.com/tile-ai/TileRT) ｜
  [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) ｜ [TileSight](https://github.com/tile-ai/TileSight) ｜
  [tilelang-puzzles](https://github.com/tile-ai/tilelang-puzzles)
- vLLM：`vllm/utils/import_utils.py`（`has_tilelang`）｜ `vllm/tilelang_utils/` ｜
  [DeepSeek-V4.1-Flash on vLLM 博客](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash)
- 本教程内：[第 15 章](15_性能优化与TileLang算子实战.md)（入门与五类原语）｜
  [第 11 章](11_主流模型接入_DeepSeek_GLM_Qwen.md)（三大模型结构）｜
  [第 12 章](12_异构硬件部署指南.md)、[第 14 章](14_硬件与模型适配_代码级细节.md)（硬件抽象与内核链路）｜
  [第 13 章](13_vLLM插件机制实现细节.md)（插件打包）

---

**上一章**：[第 15 章](15_性能优化与TileLang算子实战.md) | **下一章**：[附录](附录_速查表与常见问题.md)
