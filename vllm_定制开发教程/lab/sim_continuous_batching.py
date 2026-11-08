#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【原理模拟器 2】连续批处理 (Continuous Batching) vs 静态批处理 (Static Batching)
================================================================================
不依赖 GPU、不依赖 vLLM，纯 Python 标准库实现。
模拟 vLLM 调度器 (`vllm/v1/core/sched/scheduler.py`) 的核心行为：

  - Static Batching（传统方式）：凑一批请求 -> 全部跑完才进下一批
      缺点：先完成的请求"占着位置空转"，GPU 在浪费
  - Continuous Batching（vLLM 方式）：每一步(token 级)调度，
      谁完成了立刻让出位置，新请求立刻插队进来，GPU 永远满负荷

运行方式：
    python3 lab/sim_continuous_batching.py
"""

MAX_NUM_SEQS = 4            # 同批最多几个请求（对应 vLLM --max-num-seqs）
MAX_NUM_BATCHED_TOKENS = 64 # 每步最多处理多少 token（对应 --max-num-batched-tokens）


def bold(s): return f"\033[1m{s}\033[0m"


class Request:
    def __init__(self, rid: str, num_prompt: int, num_output: int):
        self.rid = rid
        self.num_prompt = num_prompt    # prompt 长度
        self.num_output = num_output    # 需要生成的输出长度
        self.generated = 0
        self.prefilled = 0              # prompt 已经算完的部分（chunked prefill 用）

    @property
    def done(self):
        return self.generated >= self.num_output

    def __repr__(self):
        return f"{self.rid}"


def run_static(requests: list[Request], verbose=True):
    """静态批处理：凑满一批 -> 一起 prefill -> 一起 decode 到全部结束。"""
    batches, waiting = [], list(requests)
    step = 0
    timeline = []
    while waiting:
        batch = waiting[:MAX_NUM_SEQS]
        waiting = waiting[MAX_NUM_SEQS:]
        batches.append(batch)
        # prefill 一步
        step += 1
        timeline.append((step, list(batch), "prefill"))
        if verbose:
            print(f"  step {step:>2} | prefill {batch}")
        # decode 到最长的那个结束为止 —— 短的早就完成了却还占着位置!
        max_out = max(r.num_output for r in batch)
        for _ in range(max_out):
            step += 1
            alive = [r for r in batch if not r.done]
            for r in batch:
                r.generated += 1
            timeline.append((step, list(alive), "decode"))
            if verbose:
                waste = MAX_NUM_SEQS - len(alive)
                mark = f"  ← {waste} 个位置空转!" if waste else ""
                print(f"  step {step:>2} | decode {alive or '(全空)'}{mark}")
    return step, batches, timeline


def run_continuous(requests: list[Request], verbose=True):
    """连续批处理：每一步先看有没有空位接纳新请求，再做一步计算。"""
    waiting = list(requests)
    running: list[Request] = []
    finished: list[Request] = []
    step = 0
    timeline = []
    while waiting or running:
        step += 1
        newly_admitted = []
        token_budget = MAX_NUM_BATCHED_TOKENS

        # ---- 调度器第一步：给 waiting 里的请求做 prefill（可被 chunked 切分）----
        while waiting and len(running) < MAX_NUM_SEQS and token_budget >= 2:
            r = waiting[0]
            budget = min(token_budget - 1, r.num_prompt - r.prefilled)  # 留 1 token 给 decode
            if budget <= 0:
                break
            r.prefilled += budget
            token_budget -= budget
            if r.prefilled == r.num_prompt:      # prefill 完成，进入 running
                waiting.pop(0)
                running.append(r)
                newly_admitted.append(r)
            else:
                # chunked prefill：prompt 太长，这一步只算了一部分，下步继续
                break

        # ---- 调度器第二步：所有 running 请求统一 decode 一步 ----
        for r in running:
            r.generated += 1
        done_now = [r for r in running if r.done]
        for r in done_now:
            running.remove(r)
            finished.append(r)

        batch = newly_admitted + running
        timeline.append((step, list(batch), "mixed" if newly_admitted and running else ("prefill" if newly_admitted else "decode")))
        if verbose:
            extra = f"  ← 新加入 {newly_admitted}" if newly_admitted else ""
            extra += f"  ← 完成 {done_now}" if done_now else ""
            print(f"  step {step:>2} | batch {batch or '(空)'}{extra}")

    return step, finished, timeline


def report(name, steps, requests, timeline):
    total_out = sum(r.num_output for r in requests)
    # 每步"有效计算"的平均值 = 产生 token 的请求数 / 批大小上限
    busy = sum(len(t[1]) for t in timeline) / steps
    print(f"\n  ┌─ {name} 结果 ─────────────────────────")
    print(f"  │ 总步数(step):        {steps}")
    print(f"  │ 平均每步活跃请求数:  {busy:.2f} / {MAX_NUM_SEQS}")
    print(f"  │ GPU 利用率(近似):    {busy / MAX_NUM_SEQS * 100:.0f}%")
    print(f"  └──────────────────────────────────────")


def main():
    # 8 个长短不一的请求：现实中非常常见（有人要 5 个字，有人要长文）
    requests = [Request(f"R{i}", num_prompt=20, num_output=n)
                for i, n in enumerate([2, 20, 3, 25, 4, 10, 2, 18])]

    print(bold("请求列表（输出长度参差不齐）:"))
    print("  " + ", ".join(f"{r.rid}({r.num_output}tok)" for r in requests))
    print(bold(f"\n【方案一】Static Batching  (批大小={MAX_NUM_SEQS}, 全批同步结束)"))
    print("-" * 64)
    s1, _, t1 = run_static([Request(f"R{i}", 20, n) for i, n in enumerate([2, 20, 3, 25, 4, 10, 2, 18])])
    report("Static Batching", s1, requests, t1)

    print(bold(f"\n【方案二】Continuous Batching  (vLLM 的方式, 每步动态进出)"))
    print("-" * 64)
    s2, _, t2 = run_continuous([Request(f"R{i}", 20, n) for i, n in enumerate([2, 20, 3, 25, 4, 10, 2, 18])])
    report("Continuous Batching", s2, requests, t2)

    print(bold(f"\n结论: 连续批处理用 {s2} 步干完了静态批处理 {s1} 步的活,"))
    print(f"      效率提升约 {s1 / s2:.1f} 倍 —— 这就是 vLLM 高吞吐的第一大来源!")


if __name__ == "__main__":
    main()
