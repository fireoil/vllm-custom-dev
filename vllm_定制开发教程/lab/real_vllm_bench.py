#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【真实实验 2】压测 vLLM：观察连续批处理与调度参数的影响
=========================================================
需要：已安装 vLLM + GPU

实验内容：用离线接口模拟"输出长度参差不齐"的 16 个请求，
分别在  max_num_seqs=4 和 max_num_seqs=32 下运行，
对比总吞吐 (tokens/s)，直观感受调度器参数对性能的影响。

运行：
    python3 lab/real_vllm_bench.py
"""
import time

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "用一句话解释什么是大语言模型。",
    "写一首关于秋天的五言绝句。",
    "解释 TCP 三次握手的过程。",
    "给一个 8 岁孩子解释彩虹是怎么形成的。",
    "把这句话翻译成英文：今天天气真好。",
    "写一个 Python 函数判断回文。",
    "简单介绍一下唐朝的历史。",
    "为什么天空是蓝色的？",
] * 2  # 16 个请求


def run(max_num_seqs: int):
    llm = LLM(
        model=MODEL,
        max_model_len=1024,
        max_num_seqs=max_num_seqs,      # 同一时刻最多并行处理多少请求
        gpu_memory_utilization=0.85,
    )
    sp = SamplingParams(max_tokens=128, ignore_eos=True, temperature=0.0)  # ignore_eos 让长度可控
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    wall = time.perf_counter() - t0
    total = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"\n>>> max_num_seqs={max_num_seqs:<3} | 生成 {total} tokens / {wall:.2f}s "
          f"= 吞吐 {total / wall:.1f} tok/s")
    return total / wall


def main():
    print("温馨提示：第一次运行会下载模型（约 1GB），请耐心等待。\n")
    print("================ 实验一：小并发 (max_num_seqs=4) ================")
    thr_small = run(4)

    print("\n================ 实验二：大并发 (max_num_seqs=32) ================")
    thr_big = run(32)

    print("\n================ 结论 ================")
    print(f"吞吐变化：{thr_small:.1f} -> {thr_big:.1f} tok/s（{thr_big / thr_small:.2f} 倍）")
    print("""
思考题：
 1. 为什么更大的 max_num_seqs 并不总是更快？（提示：小模型+大batch 时
    attention 计算变碎片、单步变慢；这就是调优第 09 章要讲的内容）
 2. 试着把 max_num_batched_tokens 降到 256 再跑一次，观察长 prompt
    请求的 TTFT 变化。""")


if __name__ == "__main__":
    main()
