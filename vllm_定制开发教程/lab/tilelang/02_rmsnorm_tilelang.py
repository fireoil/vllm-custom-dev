#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【TileLang 实验 2】RMSNorm —— vLLM 每层都在跑的访存型算子（第 15 章配套）
==========================================================================
动机: LLM 推理里, RMSNorm/LayerNorm 这类"访存密集"算子的耗时占比不容小觑
(每层 2 次, 61 层就是 122 次), 优化它们 = 白捡延迟。

与 GEMM 的关键差异:
  - 没有 matmul → 不用 TensorCore, 重点是 **合并访存 + reduce**
  - 每个 CTA 处理一行(hidden_size), 用 T.Parallel 展开 + T.reduce_sum
  - 学习点: reduce 语法 / fp32 中间精度 / 向量化访存

运行条件: pip install tilelang torch (CUDA)
    python3 lab/tilelang/02_rmsnorm_tilelang.py
"""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def rmsnorm(
    num_tokens: int,
    hidden_size: int,
    eps: float = 1e-6,
    threads: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @T.prim_func
    def main(
        X: T.Tensor((num_tokens, hidden_size), dtype),   # 输入激活
        W: T.Tensor((hidden_size,), dtype),              # 可学习缩放
        Y: T.Tensor((num_tokens, hidden_size), dtype),   # 输出
    ):
        with T.Kernel(num_tokens, threads=threads) as bx:   # 一行一个 CTA
            # smem 暂存一行, fragment 寄存器做 fp32 精度计算
            x_shared = T.alloc_shared((hidden_size,), dtype)
            x_f32 = T.alloc_fragment((hidden_size,), accum_dtype)
            sqr_sum = T.alloc_fragment((1,), accum_dtype)

            T.copy(X[bx, :], x_shared)                     # global -> smem
            T.copy(x_shared, x_f32)                        # smem -> 寄存器
            T.copy(W[:], x_shared)                         # 权重复用 smem 缓冲

            # 平方和 (elementwise 用 T.Parallel = 并行展开 + 自动向量化)
            T.fill(sqr_sum, 0.0)
            for i in T.Parallel(hidden_size):
                x_f32[i] = x_f32[i] * x_f32[i]
            T.reduce_sum(x_f32, sqr_sum, dim=0)            # 规约到 sqr_sum[0]

            # rms = rsqrt(mean(x^2) + eps)
            rms: T.float32 = T.sqrt(sqr_sum[0] / hidden_size + eps)

            # 归一化 * 权重 (先转回 fp32 精度算, 最后写回原 dtype)
            for i in T.Parallel(hidden_size):
                x_f32[i] = x_f32[i] / rms * T.cast(W[i], accum_dtype)
            T.copy(x_f32, Y[bx, :])
    return main


def main():
    num_tokens, hidden = 4096, 2048
    print("=== TileLang RMSNorm ===")
    kernel = rmsnorm(num_tokens, hidden)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.float16)
    w = torch.randn(hidden, device="cuda", dtype=torch.float16)
    y = kernel(x, w)

    # 对拍: 手写 torch 参考实现
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).half()
    torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)
    print("[OK] 数值正确")

    # 对比 torch 原生 rmsnorm(vLLM 内部有优化版 custom op, 这里比较朴素实现)
    def bench(fn, iters=100):
        torch.cuda.synchronize(); t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record(); torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters * 1000

    t_tile = bench(lambda: kernel(x, w))
    t_torch = bench(lambda: (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).half())
    bw = 3 * num_tokens * hidden * 2 / (t_tile * 1e-6) / 1e9   # 读x读w写y
    print(f"TileLang : {t_tile:8.1f} us  ({bw:6.0f} GB/s 有效带宽)")
    print(f"torch朴素 : {t_torch:8.1f} us")
    print("""
思考题(教程 15.4):
 1. 为什么 rows 用 Kernel(num_tokens) 而不是把 rows 塞进同一 CTA?
 2. hidden_size 不是 threads 整数倍时本内核会怎样? 如何改成 T.dynamic 支持动态?
 3. 试试把 fp32 中间计算改回 fp16, 精度掉多少?
""")


if __name__ == "__main__":
    main()
