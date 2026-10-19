#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【TileLang 实验 1】从零写一个高性能 GEMM（教程第 15 章配套）
==============================================================
TileLang: tile 级 DSL，TVM 编译器底座，Python 语法写 GPU 内核。
本例是 tilelang 官方 quick-start 同款结构，附带对比 torch 的正确性验证与计时。

运行条件：
    pip install tilelang torch   (CUDA 环境, 参考 tilelang 官方安装文档)
    python3 lab/tilelang/01_gemm_tilelang.py

学习点：
  1. T.Tensor / T.Kernel / T.alloc_shared / T.alloc_fragment 分层内存
  2. T.Pipelined 软件流水(多级缓冲, 类似 CUDA double buffering)
  3. T.gemm 自动映射到 Tensor Core (wmma/mma)
  4. tile size 与 threads 对性能的影响(动手改!)
"""
import torch
import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# kernel 工厂: 尺寸是编译期常量 → 每个 (M,N,K,block) 组合生成专属内核
# @tilelang.jit(out_idx=[-1]): 自动分配输出张量; 调用时只传输入
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1])
def matmul(
    M: int, N: int, K: int,
    block_M: int = 128, block_N: int = 128, block_K: int = 64,
    num_stages: int = 3, threads: int = 128,
    dtype: str = "float16", accum_dtype: str = "float",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),       # 全局内存中的输入 A (M x K)
        B: T.Tensor((K, N), dtype),       # 输入 B (K x N)
        C: T.Tensor((M, N), dtype),       # 输出 C = A @ B (M x N)
    ):
        # 网格: N 方向 x M 方向切成 block_N x block_M 的 tile
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            # 三级内存分层:
            A_shared = T.alloc_shared((block_M, block_K), dtype)     # 共享内存(sm)
            B_shared = T.alloc_shared((block_K, block_N), dtype)     # 共享内存(sm)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)  # 寄存器

            T.clear(C_local)                       # FP32 累加器清零

            # K 维主循环: 软件流水(num_stages=3 → 提前 3 级预取 global→smem)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)   # global -> smem
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)              # smem -> TensorCore

            T.copy(C_local, C[by * block_M, bx * block_N])       # 寄存器 -> global
    return main


def main():
    M, N, K = 1024, 1024, 1024
    print("=== TileLang GEMM (naive baseline 同款结构) ===")

    kernel = matmul(M, N, K)                       # 编译(首次会触发 TVM 编译, 需数秒)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kernel(a, b)                               # 运行!

    # ---- 正确性: 与 cuBLAS 对拍 ----
    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print("[OK] 数值正确 (fp16 累加 fp32, 与 cuBLAS 误差在容差内)")

    # ---- 计时: 让初学者感受 TensorCore 的威力 ----
    def bench(fn, iters=50):
        torch.cuda.synchronize(); t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record(); torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters * 1000  # us

    t_tile = bench(lambda: kernel(a, b))
    t_cublas = bench(lambda: a @ b)
    tflops_tile = 2 * M * N * K / (t_tile * 1e-6) / 1e12
    tflops_cublas = 2 * M * N * K / (t_cublas * 1e-6) / 1e12
    print(f"TileLang : {t_tile:8.1f} us  ({tflops_tile:6.1f} TFLOPS)")
    print(f"cuBLAS   : {t_cublas:8.1f} us  ({tflops_cublas:6.1f} TFLOPS)")

    print("""
动手调优(对应教程 15.4 优化清单):
  1. 改 block_M/N/K (尝试 64/128/256 组合) — tile 大小决定 smem 占用与并行度
  2. 改 num_stages (1/2/3/4) — 流水深度 vs smem 容量
  3. 改 threads (128/256) — 每个 CTA 的线程数
  每次改动都会重新编译, 记录 TFLOPS, 亲手找出你 GPU 上的最优配置!
""")


if __name__ == "__main__":
    main()
