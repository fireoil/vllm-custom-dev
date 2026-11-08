#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【硬核实验】飞行记录仪：给 vLLM 引擎主循环插桩（教程第 06 章配套）
====================================================================
用 monkeypatch 包一层 EngineCore.step，不改任何源码文件，
就能观察调度器的 waiting/running 队列在每个节拍的实时变化。

需要：已安装 vLLM + GPU。
运行：
    python3 lab/instrument.py
"""
import vllm.v1.engine.core as core_mod
from vllm import LLM, SamplingParams

orig_step = core_mod.EngineCore.step


def my_step(self):
    sched = self.scheduler
    print(f"[step] running={len(sched.running)} waiting={len(sched.waiting)}")
    return orig_step(self)


core_mod.EngineCore.step = my_step   # 猴子补丁: 包一层

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=512,
    gpu_memory_utilization=0.85,
)
llm.generate(
    [f"问题{i}：介绍一下春天。" for i in range(6)],
    SamplingParams(max_tokens=32, ignore_eos=True),
)

print("""
观察要点:
 1. 开始时 waiting 快速清零(请求被接纳)
 2. running 稳定在某个数字(≈并发上限/请求总数较小者)
 3. 结束前 running 逐个下降(完成一个让位一个, 连续批处理!)
""")
