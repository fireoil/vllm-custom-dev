#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【真实实验 1】在 vLLM 中亲眼验证 Prefix Caching（前缀缓存）
=============================================================
需要：已安装 vLLM + 一块 GPU（本教程机器：RTX 4060 Laptop 8GB）
模型：Qwen2.5-0.5B-Instruct（约 1GB，小显存友好）

实验内容：
  A 组：10 个请求，只有结尾不同，共享同一段超长前缀
  B 组：同样的 10 个请求，但前缀各不相同
对比两组的 TTFT（首 token 延迟）和预填充耗时。

运行：
    python3 lab/real_vllm_prefix_cache.py
"""
import time

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# 一段约 800 token 的长前缀（模拟"长系统提示词 / RAG 上下文 / few-shot 例子"）
LONG_PREFIX = (
    "你是一位资深的皮肤科医生。请始终按照以下规范回答问题：\n" + "医学知识条目 %d：皮肤分为表皮、真皮和皮下组织三层，表皮更新周期约 28 天。\n" * 50
) % tuple(range(50))

def make_requests(with_prefix: bool):
    """生成 10 个请求。with_prefix=True 时共享长前缀，False 时用各自不同的短前缀。"""
    reqs = []
    for i in range(10):
        if with_prefix:
            reqs.append(LONG_PREFIX + f"\n\n患者问题：脸上长痘痘应该怎么护理？（第{i}问）")
        else:
            reqs.append(f"患者情况{i}：来诊原因各不相同。皮肤分几层？\n请问脸上长痘痘应该怎么护理？（第{i}问）")
    return reqs


def bench(llm: LLM, prompts, label: str):
    sp = SamplingParams(max_tokens=64, temperature=0.0)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    wall = time.perf_counter() - t0
    # vLLM 会返回每个请求的指标
    ttfts = [o.metrics.first_token_ts - o.metrics.arrival_time for o in outputs] if hasattr(outputs[0], "metrics") else []
    n_cached = sum(getattr(o, "num_cached_tokens", 0) for o in outputs)
    print(f"\n[{label}] 总耗时 {wall:.2f}s | 命中缓存的 token 总数: {n_cached}")
    if ttfts:
        print(f"[{label}] 平均 TTFT: {sum(ttfts)/len(ttfts)*1000:.0f} ms")
    return wall


def main():
    llm = LLM(
        model=MODEL,
        max_model_len=2048,
        gpu_memory_utilization=0.85,   # 8GB 小卡建议 <=0.9
        enable_prefix_caching=True,    # V1 版本默认已开启，显式写出来便于教学
    )

    print("\n================ 实验开始 ================")
    # 预热一轮（加载/编译等一次性开销不参与对比）
    llm.generate(["warmup"], SamplingParams(max_tokens=1))

    t_a = bench(llm, make_requests(with_prefix=True), "A组：共享长前缀")
    t_b = bench(llm, make_requests(with_prefix=False), "B组：前缀各不相同")

    print("\n================ 结论 ================")
    print(f"A组(前缀缓存命中): {t_a:.2f}s   B组(无前缀可复用): {t_b:.2f}s")
    if t_b > 0:
        print(f"前缀缓存让这批请求快了约 {(1 - t_a / t_b) * 100:.0f}%")
    print("观察启动日志中的 'Prefix caching hit' 或输出里的 num_cached_tokens 字段，")


if __name__ == "__main__":
    main()
