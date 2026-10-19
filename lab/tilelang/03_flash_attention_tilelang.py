#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【TileLang 实验 3】简化版 FlashAttention —— 在线 softmax 的全流程（第 15 章配套）
==================================================================================
这是教程第 15 章的压轴实验: 用 TileLang 把第 01/04 章讲的注意力理论
变成一个真实的高性能 kernel。结构参考 tilelang 官方 examples/flash_attention。

核心挑战(Online Softmax, FlashAttention 的灵魂):
    softmax 需要"整行最大值/总和", 但 KV 是分块流式到来的——
    所以每处理一个 KV 块, 都要用块内 max 重缩放历史累加器:
        m_new = max(m_old, rowmax(S));  acc *= exp2(m_old - m_new);  ...

运行条件: pip install tilelang torch (CUDA, 建议显存>=6GB)
    python3 lab/tilelang/03_flash_attention_tilelang.py
"""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def flash_attention(
    batch: int, heads: int, seq_len: int, dim: int,
    is_causal: bool = True,
    block_M: int = 64, block_N: int = 64, num_stages: int = 2,
    threads: int = 128,
    dtype: str = "float16", accum_dtype: str = "float",
):
    scale = 1.0 / (dim ** 0.5)
    shape = [batch, seq_len, heads, dim]          # (B, S, H, D) 布局
    NEG_INF = -1.0e30

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        # 网格: (S/block_M) x heads x batch —— 每个 CTA 负责 Q 的一个 tile
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch,
                      threads=threads) as (bx, by, bz):
            # ---- 内存分层 ----
            Q_shared   = T.alloc_shared((block_M, dim), dtype)
            K_shared   = T.alloc_shared((block_N, dim), dtype)
            V_shared   = T.alloc_shared((block_N, dim), dtype)
            S_tile     = T.alloc_fragment((block_M, block_N), accum_dtype)  # QK^T
            S_cast     = T.alloc_fragment((block_M, block_N), dtype)        # gemm 需要 dtype
            O_acc      = T.alloc_fragment((block_M, dim), accum_dtype)      # 输出累加器
            m_i        = T.alloc_fragment((block_M,), accum_dtype)          # 行 max(历史)
            m_new      = T.alloc_fragment((block_M,), accum_dtype)
            l_i        = T.alloc_fragment((block_M,), accum_dtype)          # 行 sum(历史)

            T.copy(Q[bz, bx * block_M, by, :], Q_shared)
            T.fill(m_i, NEG_INF)
            T.fill(l_i, 0.0)
            T.clear(O_acc)

            # ---- 遍历所有 KV 块 ----
            for n in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=num_stages):
                T.copy(K[bz, n * block_N, by, :], K_shared)
                # S = Q·K^T * scale  (transpose_B=True 对 K 转置)
                T.clear(S_tile)
                T.gemm(Q_shared, K_shared, S_tile, transpose_B=True)
                # 缩放 + causal mask (并行元素级操作)
                for i, j in T.Parallel(block_M, block_N):
                    S_tile[i, j] = S_tile[i, j] * scale
                    if is_causal and bx * block_M + i < n * block_N + j:
                        S_tile[i, j] = NEG_INF

                # ---- online softmax 步骤 1: 新 max ----
                T.reduce_max(S_tile, m_new, dim=1, clear=True)   # 块内行 max
                for i in T.Parallel(block_M):
                    m_new[i] = T.max(m_i[i], m_new[i])
                # 历史累加器按新旧 max 差值缩放
                for i, j in T.Parallel(block_M, block_N):
                    S_tile[i, j] = T.exp2(S_tile[i, j] - m_new[i])   # exp2 比 exp 快
                for i in T.Parallel(block_M):
                    l_i[i] = l_i[i] * T.exp2(m_i[i] - m_new[i])
                    m_i[i] = m_new[i]

                # ---- 步骤 2: P = S/softmax_sum(未归一化), P·V 累加 ----
                T.copy(S_tile, S_cast)                            # fp32 -> fp16
                T.copy(V[bz, n * block_N, by, :], V_shared)
                T.gemm(S_cast, V_shared, O_acc, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_M, block_N):
                    l_i[i] = l_i[i] + S_tile[i, j]

            # ---- 步骤 3: 最终归一化并写回 ----
            for i, j in T.Parallel(block_M, dim):
                O_acc[i, j] = O_acc[i, j] / l_i[i]
            T.copy(O_acc, Output[bz, bx * block_M, by, :])
    return main


def main():
    B, H, S, D = 2, 4, 512, 64     # 小尺寸教学配置, 可自行调大
    print("=== TileLang 简化版 FlashAttention ===")
    kernel = flash_attention(B, H, S, D, is_causal=True)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    o = kernel(q, k, v)

    # 对拍: torch sdpa (它内部是官方 flash attention!)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    ).transpose(1, 2)
    torch.testing.assert_close(o, ref, rtol=1e-2, atol=1e-2)
    print("[OK] 与 torch SDPA(官方 FA) 数值一致 —— 你写的 kernel 是对的!")

    print("""
下一步(教程 15.5):
 1. 把 T.exp2 里的 scale 合并进 log2e 预乘(官方 example 的做法)再计时
 2. 调 block_M/block_N/num_stages, 在你的 4060 上画性能地图
 3. 进阶: 读 vLLM 源码 vllm/tilelang_utils/__init__.py, 看 vLLM 如何
    封装延迟 JIT + PassConfig + warmup(教程 15.6 全程拆解)
""")


if __name__ == "__main__":
    main()
