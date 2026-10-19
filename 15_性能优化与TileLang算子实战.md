# 第 15 章 · 性能优化方法论与 TileLang 算子实战

> 本章目标：掌握"优化"这门手艺的完整闭环——
> **定位瓶颈 → 分类（算力/带宽）→ 用 TileLang 写/改内核 → 正确集成回 vLLM**。
> TileLang 是 tile 级 DSL（TVM 编译器底座，Python 语法，官方版本线 0.1.x），
> 也是 **vLLM v0.31 的真实生产依赖**（源码树内 `vllm/tilelang_utils/`、
> `vllm/model_executor/kernels/mhc/tilelang.py`、JIT warmup 体系）。
> 配套三个渐进实验：`lab/tilelang/01 GEMM / 02 RMSNorm / 03 FlashAttention`（已通过语法检查）。

---

## 15.0 优化的总路线图

```
第14章: ncu/profiler 定位 → 判定 compute-bound 还是 memory-bound
   │
   ├─ memory-bound(归一化/逐元素/小 GEMM) → 拼: 合并访存、向量化、算子融合
   ├─ compute-bound(大 GEMM/注意力)      → 拼: tile 调优、软件流水、TensorCore
   └─ 都不满(小 kernel 铺天盖地)         → 拼: 融合 kernel / CUDA Graph(第03章)
   │
   ▼
用 TileLang 重写该算子(本章 15.1~15.3) → 对拍 → 调 tile → 集成回 vLLM(15.6)
```

**为什么选 TileLang 而不是手写 CUDA/Triton？**
三者的投入产出：手写 CUDA 最强但成本极高；Triton 生态大（vLLM 内大量使用）；
TileLang 居中——**用 30~50 行 Python 表达一个 FlashAttention**，编译器自动生成
TensorCore 指令、双缓冲流水、bank-conflict 避免，同时保留 `T.annotate_layout` 等
手动控制出口。尤其适合：快速原型 → 打平官方内核 → 必要时再下沉 CUDA。

## 15.1 TileLang 编程模型：四级内存 + 五类原语

（对应 `lab/tilelang/01_gemm_tilelang.py` 逐行讲解）

```python
@tilelang.jit(out_idx=[-1])           # JIT 编译; out_idx 声明"返回哪个参数作输出"
def matmul(M, N, K, block_M=128, block_N=128, block_K=64, num_stages=3, threads=128):
    @T.prim_func                      # ① kernel 本体
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, block_N),      # ② 网格: 每 tile 一个 CTA
                      T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")   # ③ 共享内存
            C_local  = T.alloc_fragment((block_M, block_N), "float")   #    寄存器(FP32累加)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):  # ④ 软件流水
                T.copy(A[by * block_M, k * block_K], A_shared)          # ⑤ 数据搬运
                T.gemm(A_shared, B_shared, C_local)                     #    TensorCore GEMM
            T.copy(C_local, C[by * block_M, bx * block_N])
    return main
```

**五类原语速查**（写 kernel 就用这五类）：

| 类别 | API | 说明 |
|---|---|---|
| 声明 | `T.Tensor / T.alloc_shared / T.alloc_fragment` | 全局内存 / 共享内存 / 寄存器 fragment |
| 搬运 | `T.copy(src, dst)` | 编译器自动选向量化+异步拷贝路径 |
| 计算 | `T.gemm(A, B, C, transpose_B=, policy=)` | 自动映射 TensorCore；`policy=T.GemmWarpPolicy.FullRow` 控制 warp 分片 |
| 逐元素/规约 | `T.Parallel(i, j)` 循环、`T.reduce_max/sum`、`T.exp2/T.sqrt/T.max` | `T.Parallel` 是布局推导入口，编译器向量化 |
| 流程 | `T.Pipelined(n, num_stages=k)`、`T.fill / T.clear` | 多级缓冲流水；`T.exp2` 硬件加速(FA 里 `*scale` 常预乘进 log2e) |

**调优旋钮**（对性能影响从大到小）：`block_M/N/K` → `num_stages` → `threads` →
`T.use_swizzle`（地址交错防 bank conflict）→ `T.annotate_layout`（手动布局）→
PassConfig（15.6 讲 vLLM 怎么关 TMA/warp-specialization）。

## 15.2 实验 1：GEMM（compute-bound 的标准打法）

`lab/tilelang/01_gemm_tilelang.py`。要点回顾：
- **累加器必须 fp32**（`accum_dtype="float"`）——fp16 累加是精度事故的头号来源；
- `T.Pipelined(num_stages=3)`：编译器生成 global→smem 的多级预取流水，
  把"搬运等待"和"矩阵乘"重叠起来——这就是 cuBLAS 快的核心手法之一；
- 实验脚本自带 `torch.testing.assert_close` 对拍 + TFLOPS 计时与 cuBLAS 对比。

**预期学习成果**：在 4060 上，调优后的 TileLang GEMM 能到 cuBLAS 的 70~90%
（fp16 1024³ 矩阵）。到不了 50% 说明 tile/pipeline 没调对——这正是实验要训练的手感。

## 15.3 实验 2：RMSNorm（memory-bound 的标准打法）

`lab/tilelang/02_rmsnorm_tilelang.py`。vLLM 每层都跑它（解码时每步 61 次量级），
是"白捡延迟"的经典目标。与 GEMM 相反的优化哲学：

- **没有 matmul** → 全部功夫在"让显存带宽跑满"；
- 每个 CTA 独占一行 → `T.Parallel(hidden_size)` 展开触发向量化访存；
- reduce 用 `T.reduce_sum(x_f32, sqr_sum, dim=0)`；中间精度 fp32；
- 评价指标不是 TFLOPS 而是**有效带宽 GB/s**（脚本已内置计算：
  `(读x+读w+写y)/耗时`），对照你显卡的理论带宽算"吃了几成"。

## 15.4 实验 3：FlashAttention（两类的合体，本教程压轴）

`lab/tilelang/03_flash_attention_tilelang.py`。把第 01 章的注意力公式 + 第 04 章的
块状思想变成 kernel，核心是**在线 softmax**：

```
每来一个 KV 块:
  S = Q·K^T * scale            (TileLang: T.gemm(..., transpose_B=True) + T.Parallel 缩放/mask)
  m_new = max(m_old, rowmax(S))              (T.reduce_max(dim=1, clear=True))
  P   = exp2(S - m_new)                      (历史结果已用旧 max 归过, 需重缩放)
  l_i = l_i * exp2(m_old - m_new) + rowsum(P)
  O   = O * exp2(m_old - m_new) + P·V        (第二个 T.gemm, policy=FullRow)
最后 O /= l_i
```

三个性能细节（官方 example 同款）：
1. 用 `T.exp2` 而非 `exp`（GPU 有原生 exp2 指令），scale 预乘进 log2e；
2. `S` 用 fp32 fragment 算 softmax，`T.copy` 成 fp16 fragment 再喂第二个 GEMM
   （`S_cast`——精度与 TensorCore 输入要求的平衡）；
3. causal mask 直接在 `T.Parallel` 循环里 `if` 置 `-inf`，无需额外 mask 张量。

对拍基准用 `torch.nn.functional.scaled_dot_product_attention`——它内部就是
官方 FlashAttention，**你的 60 行 kernel 和工业级实现数值一致时，本课程闭环达成**。

## 15.5 生产集成四件事：vLLM 是怎么"收编" TileLang 的

真实源码拆解（`vllm/tilelang_utils/__init__.py` 等，v0.31.0），写生产内核必须学会：

**① 平台门卫 + 延迟导入**（第 14 章 14.3 已拆解的四条军规，出处即此文件）。

**② PassConfig：按平台裁剪编译器优化**

```python
@cache
def _get_pass_configs():
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,   # 关 warp 专项化
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,          # 关 TMA 降级
    }
    if current_platform.is_cuda():
        pass_configs[tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL] = 10  # 寄存器压力调优
    return pass_configs
```
启示：**激进优化 ≠ 万金油**。vLLM 生产环境选择关掉两个最新特性换兼容性稳定
（TMA 需要 Hopper+，warp-specialized 路径在部分场景出问题）——优化决策要绑定目标硬件。

**③ 延迟 JIT（`_DeferredTileLangJitKernel`）**：内核不在 import 时编译，
首次调用时才 `@tilelang.jit` 装饰——既加速进程启动，又适配"多平台共用代码"。

**④ Compile-only warmup（`warmup/jit_warmup_tilelang_helper.py`）**：
vLLM 启动期用假数据**只编译不执行**（`compile_tilelang` 直接填充 `_kernel_cache`），
把 JIT 编译开销从"第一个真实请求"挪到启动阶段——与第 03 章 CUDA Graph 预录制
是同一个哲学：**运行期零意外**。

**⑤ 注册为 torch custom op**：真实内核（`kernels/mhc/tilelang.py`）用
`vllm.utils.torch_utils.direct_register_custom_op` 把 TileLang 函数注册成
`torch.library` custom op——使其在 `torch.compile`/CUDA Graph 中成为不可分的原子算子
（避免图断裂）。**这是任何自研内核接入 vLLM 执行图的标准姿势。**

**⑥ 双实现回退模式**：同一位点的选择逻辑
`use_deep_gemm = is_deep_gemm_supported() or not use_tilelang_fallback`——
硬件支持 DeepGEMM（NVIDIA 专属极致优化库）走 DeepGEMM，否则回退 TileLang 版。
"**专属极致 + 通用回退**"的双路线是跨硬件内核工程的通用模式。

## 15.6 优化 Checklist（把 15.1~15.5 收进口袋）

写完 kernel 后按序自查：
- [ ] 对拍通过？（多 shape、边界行、fp32 中间精度）
- [ ] 分类正确？compute-bound 调 tile/pipeline；memory-bound 拼带宽利用率
- [ ] 累加器 fp32？exp 用 exp2？mask 用 -inf 常量？
- [ ] tile size 扫过一轮？num_stages 扫过一轮？（脚本化 sweep）
- [ ] bank conflict？（ncu 看 l1tex 指标 / `T.use_swizzle`）
- [ ] 集成姿势：custom op 注册 + 平台门卫 + warmup + PassConfig 绑定目标硬件
- [ ] 有"通用回退路线"吗？低端卡/其他平台跑什么？

## 15.7 动手实验总表

| 实验 | 文件 | 学习点 | 完成标准 |
|---|---|---|---|
| 🚀 15-1 GEMM | `lab/tilelang/01_*.py` | tile/流水/TensorCore | TFLOPS 达到 cuBLAS ≥70%，记录最优配置 |
| 🚀 15-2 RMSNorm | `lab/tilelang/02_*.py` | 带宽型优化 | 有效带宽 ≥ 显存理论带宽 70% |
| 🚀 15-3 FlashAttention | `lab/tilelang/03_*.py` | 在线 softmax 全流程 | 与 torch SDPA 数值一致 |
| 🔬 15-4 sweep | 改 15-1 参数循环跑 | 参数敏感度地图 | 产出一张 block×stages 性能表 |
| 🔬 15-5 源码精读 | `vllm/tilelang_utils/__init__.py` | 生产集成模式 | 画出 15.5 六件事的调用图 |

## 15.8 本章小结 + 自测

1. TileLang 五类原语各是什么？`T.Parallel` 为什么被称为"布局推导入口"？
2. compute-bound 与 memory-bound 的 kernel 分别往哪里使劲？各自的成功指标？
3. 在线 softmax 中，历史累加器为什么要乘 `exp2(m_old - m_new)`？不乘会怎样？
4. vLLM 为什么在生产 PassConfig 里关掉 warp-specialized 与 TMA lowering？
5. `direct_register_custom_op` 解决什么问题？不注册会发生什么（提示：torch.compile/CUDA Graph）？

> 📚 **深入篇**：本章是 TileLang 入门（五类原语 + 三个基础 kernel）。
> 工业级完全指南——MLA/GQA/MoE/W4A16 完整内核、昇腾等多硬件适配、vLLM 生产集成代码——见
> [第 16 章](16_TileLang工业级内核开发完全指南.md)。

**上一章**：[第 14 章](14_硬件与模型适配_代码级细节.md) | **下一章**：[附录](附录_速查表与常见问题.md)
