#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【真实实验 0】你的第一个 vLLM 程序（教程第 02 章配套）
=======================================================
需要：已安装 vLLM + GPU。第一次运行会下载模型（约 1GB）。

运行：
    python3 lab/first_run.py
"""
from vllm import LLM, SamplingParams

# 1. 创建引擎：加载模型 + 显存摸底 + 预分配 KV Cache（这一步做了什么，第04章揭晓）
llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=1024,
    gpu_memory_utilization=0.85,
)

# 2. 采样参数：控制"怎么生成"
sp = SamplingParams(temperature=0.0, max_tokens=64)

# 3. 生成！注意 prompts 是列表 —— 一次喂多个请求，这就是批处理
prompts = [
    "用一句话解释什么是大语言模型",
    "写一句关于大海的诗",
    "1+1等于几",
]
outputs = llm.generate(prompts, sp)

# 4. 看结果
for out in outputs:
    print(f"prompt : {out.prompt[:20]}...")
    print(f"output : {out.outputs[0].text}")
    print(f"生成了 {len(out.outputs[0].token_ids)} 个 token\n")
