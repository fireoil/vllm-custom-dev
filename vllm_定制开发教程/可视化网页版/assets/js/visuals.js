/* ============================================================
   vLLM 教程网页版 · 交互式图形组件库（纯原生 JS + SVG，离线可用）
   用法：<div class="viz" data-viz="组件名">
          <script type="application/json" class="viz-cfg">{ ...可选配置... }</script>
        </div>
   ============================================================ */
(function () {
  "use strict";

  /* ---------- 公共工具 ---------- */
  const PALETTE = ["#4f9cf9", "#22c55e", "#f59e0b", "#ec4899", "#a78bfa", "#22d3ee", "#fb923c", "#34d399"];
  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  function fmtBytes(b) {
    if (b >= 1073741824) return (b / 1073741824).toFixed(2) + " GB";
    if (b >= 1048576) return (b / 1048576).toFixed(1) + " MB";
    if (b >= 1024) return (b / 1024).toFixed(1) + " KB";
    return b + " B";
  }
  function svgEl(tag, attrs) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function esc(s) { return String(s).replace(/</g, "&lt;"); }

  const REGISTRY = {};

  /* ============================================================
     1. bar-compare —— 通用横向条形图（性能对比）
     cfg: { unit, max, note, items: [{label, value, color, badge}] }
     ============================================================ */
  REGISTRY["bar-compare"] = function (box, cfg) {
    const body = box.querySelector(".viz-body");
    const items = cfg.items || [];
    const max = cfg.max || Math.max(...items.map(i => i.value)) * 1.15;
    items.forEach(function (it) {
      const row = el("div");
      row.style.cssText = "display:flex;align-items:center;gap:12px;margin:10px 0;";
      const label = el("div", null, esc(it.label));
      label.style.cssText = "width:150px;text-align:right;font-size:13.5px;color:var(--text-dim);flex-shrink:0;";
      const track = el("div");
      track.style.cssText = "flex:1;height:26px;background:var(--bg-code);border-radius:6px;overflow:hidden;border:1px solid var(--border-soft);position:relative;";
      const fill = el("div");
      fill.style.cssText = `height:100%;width:0;border-radius:6px;background:${it.color || "#4f9cf9"};transition:width 1s cubic-bezier(.2,.8,.2,1);display:flex;align-items:center;`;
      const val = el("span", null, esc(it.badge || (it.value + (cfg.unit || ""))));
      val.style.cssText = "font-size:12px;font-weight:700;color:#fff;padding-left:8px;white-space:nowrap;font-family:var(--mono);";
      fill.appendChild(val);
      track.appendChild(fill);
      row.appendChild(label); row.appendChild(track);
      body.appendChild(row);
      requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = (it.value / max * 100).toFixed(1) + "%"; }));
    });
    if (cfg.note) body.appendChild(el("div", "viz-note", cfg.note));
  };

  /* ============================================================
     2. kv-calc —— KV Cache 大小交互计算器
     ============================================================ */
  REGISTRY["kv-calc"] = function (box) {
    const body = box.querySelector(".viz-body");
    const state = { layers: 28, heads: 4, dim: 128, bytes: 2, seq: 4096, gpu: 8 };
    const sliders = [
      { key: "layers", label: "层数 L", min: 1, max: 80, step: 1, suffix: " 层" },
      { key: "heads", label: "KV 头数 Hkv (GQA)", min: 1, max: 32, step: 1, suffix: " 个" },
      { key: "dim", label: "每头维度 dhead", min: 32, max: 256, step: 8, suffix: "" },
      { key: "bytes", label: "每元素字节 (精度)", min: 0.5, max: 2, step: 0.5, suffix: " B" },
      { key: "seq", label: "序列长度 (上下文)", min: 128, max: 32768, step: 128, suffix: " token" },
    ];
    const ctrl = el("div");
    const readout = el("div");
    readout.style.cssText = "display:flex;gap:14px;flex-wrap:wrap;margin-top:16px;";
    const barBox = el("div");
    barBox.style.marginTop = "18px";
    body.appendChild(ctrl); body.appendChild(readout); body.appendChild(barBox);

    const precName = { 2: "FP16/BF16", 1: "FP8/INT8", 0.5: "FP4/INT4" };
    function update() {
      const perToken = 2 * state.layers * state.heads * state.dim * state.bytes;
      const total = perToken * state.seq;
      const gpuB = state.gpu * 1073741824;
      readout.innerHTML =
        `<div style="flex:1;min-width:150px;border:1px solid var(--border);border-radius:10px;padding:12px 16px;background:var(--bg-card);">
           <div style="font-size:12px;color:var(--text-faint);">每 token KV 大小</div>
           <div style="font-size:24px;font-weight:800;color:var(--accent);font-family:var(--mono);">${fmtBytes(perToken)}</div>
           <div style="font-size:11.5px;color:var(--text-faint);">精度：${precName[state.bytes] || state.bytes + "B"}</div></div>
         <div style="flex:1;min-width:150px;border:1px solid var(--border);border-radius:10px;padding:12px 16px;background:var(--bg-card);">
           <div style="font-size:12px;color:var(--text-faint);">单个请求占用（${state.seq} token）</div>
           <div style="font-size:24px;font-weight:800;color:var(--accent-3);font-family:var(--mono);">${fmtBytes(total)}</div>
           <div style="font-size:11.5px;color:var(--text-faint);">= 2 × ${state.layers} × ${state.heads} × ${state.dim} × ${state.bytes}B × ${state.seq}</div></div>
         <div style="flex:1;min-width:150px;border:1px solid var(--border);border-radius:10px;padding:12px 16px;background:var(--bg-card);">
           <div style="font-size:12px;color:var(--text-faint);">${state.gpu}GB 显存理论可并发请求数</div>
           <div style="font-size:24px;font-weight:800;color:var(--accent-2);font-family:var(--mono);">${total > 0 ? Math.floor(gpuB * 0.85 / total) : "∞"}</div>
           <div style="font-size:11.5px;color:var(--text-faint);">按 85% 可用显存估算（未扣权重）</div></div>`;
      drawBar(total, gpuB);
    }
    function drawBar(kvTotal, gpuB) {
      const W = 100;
      let weightGB = { 8: 4.5, 24: 13.5, 80: 45 }[state.gpu] || state.gpu * 0.55;
      let kvPct = Math.min(96, kvTotal / gpuB * 100 * 10); // 放大 10 倍示意单请求占比
      barBox.innerHTML =
        `<div style="font-size:13px;color:var(--text-dim);margin-bottom:8px;">显存"座位"示意（${state.gpu}GB 显卡，KV 部分按 10 倍放大示意单个请求占比）</div>
         <div style="height:34px;border-radius:8px;overflow:hidden;display:flex;border:1px solid var(--border-soft);">
           <div style="width:14%;background:linear-gradient(180deg,#a78bfa,#7c5ce0);display:flex;align-items:center;justify-content:center;font-size:11.5px;color:#fff;font-weight:700;">权重</div>
           <div style="width:${Math.max(4, kvPct * 0.6)}%;background:linear-gradient(180deg,#4f9cf9,#2f6fd0);display:flex;align-items:center;justify-content:center;font-size:11.5px;color:#fff;font-weight:700;">KV Cache 座位</div>
           <div style="flex:1;background:repeating-linear-gradient(45deg,#131a24,#131a24 8px,#0d1117 8px,#0d1117 16px);display:flex;align-items:center;justify-content:center;font-size:11.5px;color:var(--text-faint);">传统引擎：按 max_len 全部锁死 → vLLM：按需分块，座位利用率拉满</div>
         </div>`;
    }
    const gpuSel = el("div", "slider-row",
      `<label style="min-width:130px;color:var(--text-dim);">显卡显存</label>`);
    const sel = el("select");
    sel.style.cssText = "flex:1;background:var(--bg-card);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:6px;";
    [[8, "8GB（RTX 4060 Laptop）"], [24, "24GB（RTX 4090）"], [80, "80GB（A100/H100）"]].forEach(function (o) {
      const op = el("option"); op.value = o[0]; op.textContent = o[1]; sel.appendChild(op);
    });
    sel.addEventListener("change", function () { state.gpu = +sel.value; update(); });
    gpuSel.appendChild(sel);

    sliders.forEach(function (s) {
      const row = el("div", "slider-row");
      row.innerHTML = `<label>${s.label}</label>`;
      const inp = el("input"); inp.type = "range";
      inp.min = s.min; inp.max = s.max; inp.step = s.step; inp.value = state[s.key];
      const val = el("span", "val");
      function sync() {
        state[s.key] = +inp.value;
        val.textContent = inp.value + (s.key === "bytes" ? " B (" + (precName[+inp.value] || "") + ")" : s.suffix);
        update();
      }
      inp.addEventListener("input", sync);
      sync();
      row.appendChild(inp); row.appendChild(val);
      ctrl.appendChild(row);
    });
    ctrl.appendChild(gpuSel);
    update();
  };

  /* ============================================================
     3. paged-blocks —— PagedAttention 显存块模拟器
     ============================================================ */
  REGISTRY["paged-blocks"] = function (box) {
    const body = box.querySelector(".viz-body");
    const COLS = 10, ROWS = 4, TOTAL = COLS * ROWS, BLOCK_TOKENS = 4;
    const reqs = [];  // {name, color, physBlocks[], prefixShared}
    let nextId = 0, sharedBlocks = []; // 物理块号（前缀共享块）
    const grid = el("div");
    grid.style.cssText = `display:grid;grid-template-columns:repeat(${COLS},1fr);gap:6px;`;
    const cells = [];
    for (let i = 0; i < TOTAL; i++) {
      const c = el("div");
      c.style.cssText = `aspect-ratio:1.35;border-radius:6px;background:var(--bg-code);border:1px dashed var(--border);display:flex;align-items:center;justify-content:center;font-size:11px;font-family:var(--mono);color:var(--text-faint);transition:all .25s;`;
      grid.appendChild(c); cells.push(c);
    }
    const log = el("div", "viz-note");
    const mapBox = el("div");
    mapBox.style.cssText = "margin-top:14px;font-family:var(--mono);font-size:12.5px;color:var(--text-dim);line-height:2;";
    const status = el("div", "viz-status");

    function freeCells() { const a = []; for (let i = 0; i < TOTAL; i++) if (!cells[i].dataset.req) a.push(i); return a; }
    function paint() {
      for (let i = 0; i < TOTAL; i++) {
        const c = cells[i], r = c.dataset.req;
        if (r) {
          const req = reqs[+r];
          c.style.background = req.color;
          c.style.border = "1px solid " + req.color;
          c.style.color = "#0b0f16";
          c.style.fontWeight = "700";
          c.textContent = req.name + (c.dataset.shared ? "⇄" : "");
          c.style.boxShadow = c.dataset.full ? "" : "inset 0 -6px 0 rgba(0,0,0,.28)";
        } else {
          c.style.background = "var(--bg-code)"; c.style.border = "1px dashed var(--border)";
          c.style.color = "var(--text-faint)"; c.style.boxShadow = "";
          c.textContent = c.dataset.shared === "1" ? "共享" : "";
        }
      }
      /* 逻辑→物理映射表 */
      mapBox.innerHTML = reqs.map(function (r) {
        return `<div><span style="color:${r.color};font-weight:700;">请求 ${r.name}</span>
          逻辑块 [${r.physBlocks.map(function (p, i) {
            return `<span style="border:1px solid ${r.color};border-radius:4px;padding:0 6px;margin:0 2px;color:${r.color};">物理#${p}${cells[p].dataset.shared === "1" ? "⇄" : ""}</span>`;
          }).join("")}]</div>`;
      }).join("") || "<div>（尚无请求，点击下方按钮试试）</div>";
      const used = TOTAL - freeCells().length;
      status.textContent = `物理块：${used}/${TOTAL} 已用 · 利用率 ${Math.round(used / TOTAL * 100)}%`;
    }
    function logMsg(html) { log.innerHTML = "▶ " + html; }

    function allocBlock(req, shared) {
      const free = freeCells();
      if (!free.length) return -1;
      const idx = free[Math.floor(Math.random() * free.length)];
      const c = cells[idx];
      c.dataset.req = reqs.indexOf(req);
      c.dataset.full = "";
      if (shared) c.dataset.shared = "1";
      req.physBlocks.push(idx);
      return idx;
    }

    function addRequest(name, usePrefix) {
      const PRE = usePrefix ? 1 : 0; // 前缀共享 1 个块（相同 prompt 的前 4 token）
      const req = { name: name, color: PALETTE[(nextId + 1) % PALETTE.length], physBlocks: [] };
      reqs.push(req); nextId++;
      let ok = true;
      if (usePrefix) {
        if (sharedBlocks.length) {
          sharedBlocks.forEach(function (b) {
            cells[b].dataset.req = reqs.indexOf(req); req.physBlocks.push(b);
          });
          logMsg(`<span class="hl">前缀共享！</span>请求 ${name} 复用了已有请求的物理块 #${sharedBlocks.join("、#")}（引用计数 +1，<b>零拷贝</b>），只为自己的新内容分配块。`);
        } else {
          const b = allocBlock(req, true);
          if (b >= 0) sharedBlocks.push(b);
          ok = b >= 0;
          logMsg(ok ? `请求 ${name} 到来，前缀块分配到物理 #${b}。后续相同前缀的请求将直接共享它。` : "显存不足！");
        }
      } else {
        const b = allocBlock(req, false);
        ok = b >= 0;
        logMsg(ok ? `请求 ${name} 到来：按需分配 1 个物理块（16 token 的小块，而不是按 max_len 预留一大块）→ 物理块 #${b}。` : "显存不足！触发抢占/换出。");
      }
      paint();
      return ok;
    }

    function stepGenerate(n) {
      for (let t = 0; t < (n || 1); t++) {
        reqs.forEach(function (req) {
          const last = req.physBlocks[req.physBlocks.length - 1];
          const c = cells[last];
          if (c.dataset.full === "1") {
            const nb = allocBlock(req, false);
            if (nb < 0) { logMsg(`<span class="hl">显存耗尽！</span>请求 ${req.name} 无法再分配新块 → vLLM 会抢占/换出最久未用的请求（recompute / swap）。`); return; }
            logMsg(`请求 ${req.name} 生成中新 token，旧块装满 → <b>从空闲块池取任意散落块</b> #${nb}（逻辑连续、物理不必连续 → 无外部碎片）。`);
            req.physBlocks.push(nb);
          } else {
            c.dataset.full = "1";
          }
        });
      }
      paint();
    }

    function finishRandom() {
      if (!reqs.length) return;
      const i = Math.floor(Math.random() * reqs.length);
      const req = reqs[i];
      req.physBlocks.forEach(function (b) {
        const c = cells[b];
        if (c.dataset.shared === "1") {
          /* 引用计数：还有别人共享则不释放 */
          const others = reqs.some(function (r, j) { return j !== i && r.physBlocks.includes(b); });
          if (!others) { delete c.dataset.req; delete c.dataset.shared; delete c.dataset.full; c.dataset.shared = "0"; }
        }
        delete c.dataset.req; delete c.dataset.full;
      });
      reqs.splice(i, 1);
      logMsg(`请求 ${req.name} 完成 → 其占用的物理块全部归还空闲块池，<b>任何位置的新请求都能立刻复用</b>（内部碎片 ≤ 1 块）。`);
      paint();
    }

    function reset() {
      reqs.length = 0; sharedBlocks = []; nextId = 0;
      cells.forEach(function (c) { delete c.dataset.req; delete c.dataset.full; delete c.dataset.shared; });
      logMsg("已重置。对比传统引擎「按 max_len 预留 + 必须连续分配」，这里每一格是一个 16-token 的小块。");
      paint();
    }

    const tb = el("div", "viz-toolbar");
    const mk = function (txt, fn, cls) { const b = el("button", "viz-btn" + (cls ? " " + cls : ""), txt); b.addEventListener("click", fn); return b; };
    tb.appendChild(mk("＋ 请求 A（普通 prompt）", function () { addRequest("A", false); }, "primary"));
    tb.appendChild(mk("＋ 请求 B（相同前缀）", function () { addRequest("B", true); }));
    tb.appendChild(mk("＋ 请求 C（相同前缀）", function () { addRequest("C", true); }));
    tb.appendChild(mk("▶ 生成 5 步", function () { stepGenerate(5); }));
    tb.appendChild(mk("✔ 随机完成一个请求", finishRandom));
    tb.appendChild(mk("↺ 重置", reset));
    body.appendChild(tb);
    body.appendChild(grid);
    body.appendChild(mapBox);
    body.appendChild(log);
    body.appendChild(status);
    reset();
  };

  /* ============================================================
     4. batch-sim —— 静态批处理 vs 连续批处理 对比动画
     ============================================================ */
  REGISTRY["batch-sim"] = function (box) {
    const body = box.querySelector(".viz-body");
    const SLOTS = 4;
    const jobs = [3, 7, 4, 9, 5, 6, 2, 8, 4, 6].map(function (len, i) {
      return { id: "R" + (i + 1), len: len, done: 0, color: PALETTE[i % PALETTE.length] };
    });
    const panels = el("div");
    panels.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:18px;";
    if (window.matchMedia("(max-width:760px)").matches) panels.style.gridTemplateColumns = "1fr";

    function makePanel(title, mode) {
      const p = el("div");
      p.style.cssText = "border:1px solid var(--border);border-radius:10px;padding:14px;background:var(--bg-card);";
      p.innerHTML = `<div style="font-weight:700;margin-bottom:10px;">${title}</div>`;
      const lanes = el("div");
      lanes.style.cssText = "display:flex;flex-direction:column;gap:6px;";
      const laneEls = [];
      for (let i = 0; i < SLOTS; i++) {
        const lane = el("div");
        lane.style.cssText = "display:flex;gap:3px;height:22px;";
        const cells = [];
        for (let j = 0; j < 24; j++) {
          const c = el("div");
          c.style.cssText = "flex:1;border-radius:3px;background:var(--bg-code);border:1px solid var(--border-soft);";
          lane.appendChild(c); cells.push(c);
        }
        lanes.appendChild(lane); laneEls.push(cells);
      }
      const stat = el("div");
      stat.style.cssText = "margin-top:10px;font-size:12.5px;font-family:var(--mono);color:var(--text-dim);";
      p.appendChild(lanes); p.appendChild(stat);
      panels.appendChild(p);
      return { lanes: laneEls, stat: stat, queue: [], active: [], t: 0, finished: 0, utilSteps: 0, waiting: jobs.slice() };
    }
    const P1 = makePanel("静态批处理（凑满 4 个一起跑，全批结束才换）", "static");
    const P2 = makePanel("连续批处理（每步 token 级进出，完成即让位）", "cont");

    function draw(P) {
      P.lanes.forEach(function (cells, s) {
        const job = P.active[s];
        cells.forEach(function (c, j) {
          c.style.background = "var(--bg-code)"; c.style.borderColor = "var(--border-soft)"; c.innerHTML = "";
          if (job && j < P.t && j < job.len) {
            const local = j - (job.start || 0);
            if (local >= 0 && local < job.done) {
              c.style.background = job.color; c.style.borderColor = job.color;
            }
          }
        });
      });
      P.stat.innerHTML = `时钟 t=${P.t} · 完成 ${P.finished}/${jobs.length} · GPU 槽位利用率 <b style="color:${P.mode === "cont" ? "var(--accent-2)" : "var(--accent-3)"}">${P.utilSteps ? Math.round(P.busySteps / P.utilSteps * 100) : 0}%</b>（${P.mode === "static" ? "整批空转等待" : "即完即补"}）`;
    }
    function stepStatic(P) {
      P.t++; P.utilSteps++;
      /* 填充空槽 */
      while (P.active.length < SLOTS && P.waiting.length) {
        const job = P.waiting.shift(); job.start = P.t; P.active.push(job);
      }
      let busy = 0;
      P.active.forEach(function (job) { job.done++; busy++; });
      P.busySteps = (P.busySteps || 0) + busy;
      /* 全部跑完才算完 */
      const allDone = P.active.every(function (j) { return j.done >= j.len; });
      if (allDone && P.active.length) {
        P.finished += P.active.length;
        P.active = [];
      }
      draw(P);
    }
    function stepCont(P) {
      P.t++; P.utilSteps++;
      for (let s = 0; s < SLOTS; s++) {
        const job = P.active[s];
        if (!job) {
          if (P.waiting.length) { const nj = P.waiting.shift(); nj.start = P.t; P.active[s] = nj; }
          continue;
        }
        job.done++; P.busySteps = (P.busySteps || 0) + 1;
        if (job.done >= job.len) { P.finished++; P.active[s] = null; }
      }
      /* 同一步内即完成即让位：先移除完成的，再补位 */
      for (let s = 0; s < SLOTS; s++) {
        const job = P.active[s];
        if (job && job.done >= job.len) { P.finished++; P.active[s] = null; }
        if (!P.active[s] && P.waiting.length) { const nj = P.waiting.shift(); nj.start = P.t; P.active[s] = nj; }
      }
      draw(P);
    }
    function allFinished(P) { return P.finished >= jobs.length; }

    const tb = el("div", "viz-toolbar");
    const stepBtn = el("button", "viz-btn primary", "▶ 单步推进");
    const autoBtn = el("button", "viz-btn", "⏩ 自动播放");
    const resetBtn = el("button", "viz-btn", "↺ 重置");
    let timer = null;
    function tick() {
      if (!allFinished(P1)) stepStatic(P1);
      if (!allFinished(P2)) stepCont(P2);
      if (allFinished(P1) && allFinished(P2)) {
        stop();
        note2.innerHTML = "🏁 对比结果：静态批处理 t=" + P1.t + " 步跑完、槽位利用率 " +
          Math.round(P1.busySteps / P1.utilSteps * 100) + "%；连续批处理 t=" + P2.t +
          " 步跑完、利用率 " + Math.round(P2.busySteps / P2.utilSteps * 100) + "%。请求越多、长度越不均，差距越大。";
      }
    }
    function stop() { clearInterval(timer); timer = null; autoBtn.textContent = "⏩ 自动播放"; }
    stepBtn.addEventListener("click", tick);
    autoBtn.addEventListener("click", function () {
      if (timer) { stop(); return; }
      autoBtn.textContent = "⏸ 暂停";
      timer = setInterval(tick, 260);
    });
    function resetAll() {
      stop();
      [P1, P2].forEach(function (P) {
        P.t = 0; P.finished = 0; P.utilSteps = 0; P.busySteps = 0;
        P.active = []; P.waiting = jobs.map(function (j) {
          return { id: j.id, len: j.len, done: 0, start: 0, color: j.color };
        });
      });
      note2.innerHTML = "";
      draw(P1); draw(P2);
    }
    stepBtn.addEventListener("click", tick);
    autoBtn.addEventListener("click", function () {
      if (timer) { stop(); return; }
      autoBtn.textContent = "⏸ 暂停";
      timer = setInterval(tick, 260);
    });
    resetBtn.addEventListener("click", resetAll);
    tb.appendChild(stepBtn); tb.appendChild(autoBtn); tb.appendChild(resetBtn);
    body.appendChild(tb);
    body.appendChild(panels);
    body.appendChild(el("div", "viz-legend",
      `<span class="key"><span class="sw" style="background:${PALETTE[0]}"></span>每个颜色 = 一个请求</span>
       <span class="key"><span class="sw" style="background:var(--bg-code);border:1px solid var(--border-soft)"></span>空闲槽位（GPU 空转）</span>`));
    const note2 = el("div", "viz-note", "同一批 10 个请求（长度 2~9 不等）。观察右侧连续批处理：任一请求完成，下一拍立刻补位；左侧静态批处理必须等整批最慢者结束。最终利用率差距一目了然。");
    body.appendChild(note2);
    /* 初始化 P2 的补位逻辑检查（cont 模式双保险已在 stepCont 内处理） */
    draw(P1); draw(P2);
  };

  /* ============================================================
     5. arch-diagram —— vLLM 架构分层图
     cfg: { layers: [{name, color, items: [{name, desc}]}] }
     ============================================================ */
  REGISTRY["arch-diagram"] = function (box, cfg) {
    const body = box.querySelector(".viz-body");
    const desc = el("div", "viz-note", "👆 点击任意组件，查看它在源码中的职责说明。");
    (cfg.layers || []).forEach(function (L) {
      const layer = el("div");
      layer.style.cssText = `margin:12px 0;padding:10px 14px;border-radius:10px;border:1px solid ${L.color}55;background:${L.color}12;`;
      layer.innerHTML = `<div style="font-size:12px;font-weight:700;color:${L.color};letter-spacing:1px;margin-bottom:8px;">${esc(L.name)}</div>`;
      const row = el("div");
      row.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;";
      (L.items || []).forEach(function (it) {
        const chip = el("button", "viz-btn", esc(it.name));
        chip.style.cssText += `border-color:${L.color}66;font-size:12.5px;`;
        chip.addEventListener("click", function () {
          desc.innerHTML = `<span class="hl">${esc(it.name)}</span> — ${esc(it.desc)}`;
          row.querySelectorAll(".viz-btn").forEach(function (b) { b.style.outline = ""; });
          chip.style.outline = "2px solid " + L.color;
        });
        row.appendChild(chip);
      });
      layer.appendChild(row);
      body.appendChild(layer);
    });
    body.appendChild(desc);
  };

  /* ============================================================
     6. journey-flow —— 一次请求的完整旅程（逐步走读）
     cfg: { steps: [{title, detail, code}] }
     ============================================================ */
  REGISTRY["journey-flow"] = function (box, cfg) {
    const body = box.querySelector(".viz-body");
    const steps = cfg.steps || [];
    let cur = -1;
    const flow = el("div");
    flow.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:16px;";
    const detail = el("div");
    detail.style.cssText = "border:1px solid var(--border);border-radius:10px;background:var(--bg-card);padding:16px;min-height:120px;";
    const tb = el("div", "viz-toolbar");
    const prev = el("button", "viz-btn", "← 上一步");
    const next = el("button", "viz-btn primary", "下一步 →");
    prev.addEventListener("click", function () { if (cur > 0) { cur--; show(); } });
    next.addEventListener("click", function () { if (cur < steps.length - 1) { cur++; show(); } else { next.disabled = true; } });
    tb.appendChild(prev); tb.appendChild(next);

    steps.forEach(function (s, i) {
      if (i) flow.appendChild(el("span", null, '<span style="color:var(--text-faint);font-size:11px;">→</span>'));
      const chip = el("button", "viz-btn", `${i + 1}. ${esc(s.title)}`);
      chip.style.fontSize = "12px";
      chip.addEventListener("click", function () { cur = i; show(); });
      chip.dataset.i = i;
      flow.appendChild(chip);
    });
    function show() {
      flow.querySelectorAll(".viz-btn").forEach(function (b, i) {
        const idx = +b.dataset.i;
        b.style.background = idx === cur ? "var(--accent)" : "var(--bg-card)";
        b.style.borderColor = idx === cur ? "var(--accent)" : "var(--border)";
        b.style.color = idx === cur ? "#fff" : "var(--text)";
      });
      const s = steps[cur];
      next.disabled = cur >= steps.length - 1;
      prev.disabled = cur <= 0;
      detail.innerHTML = `<div style="font-weight:700;color:var(--accent);margin-bottom:6px;">Step ${cur + 1} · ${esc(s.title)}</div>
        <div style="color:var(--text-dim);font-size:14px;">${s.detail}</div>
        ${s.code ? `<pre style="margin-top:12px;"><code>${esc(s.code)}</code></pre>` : ""}`;
    }
    body.appendChild(flow); body.appendChild(tb); body.appendChild(detail);
    cur = 0; show();
  };

  /* ============================================================
     7. parallel-diagram —— TP / PP / EP 并行策略可视化
     ============================================================ */
  REGISTRY["parallel-diagram"] = function (box) {
    const body = box.querySelector(".viz-body");
    const tabs = el("div", "viz-toolbar");
    const stage = el("div");
    const note = el("div", "viz-note");
    const TABS = {
      TP: {
        title: "张量并行 Tensor Parallel —— 把「一层」切给多卡",
        render: function () {
          const g = el("div");
          g.style.cssText = "display:grid;grid-template-columns:repeat(4,1fr);gap:10px;";
          for (let i = 0; i < 4; i++) {
            const c = el("div");
            c.style.cssText = `border:1px solid ${PALETTE[i]}66;background:${PALETTE[i]}18;border-radius:10px;padding:12px;text-align:center;`;
            c.innerHTML = `<div style="font-weight:700;color:${PALETTE[i]};">GPU ${i}</div>
              <div style="margin:10px auto;width:80%;height:46px;border-radius:6px;background:repeating-linear-gradient(90deg,${PALETTE[i]},${PALETTE[i]} 6px,transparent 6px,transparent 10px);opacity:.8;"></div>
              <div style="font-size:12px;color:var(--text-dim);">权重列切片 1/4<br>（W 按列切，激活按行切，AllReduce 汇总）</div>`;
            g.appendChild(c);
          }
          note.innerHTML = `<span class="hl">TP：</span>每一层的权重矩阵按列/行切成 4 份，4 张卡<b>同时算同一层</b>；每层结束后做 AllReduce 通信。通信频繁 → 通常限于<b>单机 NVLink 内</b>。对应 vLLM 源码：各模型 forward 里的 <code>RowParallelLinear / ColumnParallelLinear</code>。`;
          return g;
        }
      },
      PP: {
        title: "流水线并行 Pipeline Parallel —— 把「层序列」切段",
        render: function () {
          const g = el("div");
          g.style.cssText = "display:grid;grid-template-columns:repeat(4,1fr);gap:0;position:relative;";
          for (let i = 0; i < 4; i++) {
            const c = el("div");
            const hot = `background:linear-gradient(180deg,${PALETTE[i]}30,transparent);`;
            c.style.cssText = `border:1px dashed ${PALETTE[i]}88;border-left:${i ? "none" : "1px dashed " + PALETTE[i] + "88"};padding:12px;text-align:center;${i < 2 ? hot : ""}`;
            c.innerHTML = `<div style="font-weight:700;color:${PALETTE[i]};">Stage ${i}</div>
              <div style="font-size:12px;color:var(--text-dim);margin:8px 0;">GPU ${i}<br>第 ${i * 7 + 1}~${(i + 1) * 7} 层</div>
              <div style="font-size:11px;color:${i < 2 ? "var(--accent-2)" : "var(--text-faint)"};">${i < 2 ? "🏃 正在处理 micro-batch" : "⏳ 等待激活传入"}</div>`;
            g.appendChild(c);
          }
          note.innerHTML = `<span class="hl">PP：</span>28 层模型切 4 段，各卡托管一段；像流水线一样传递激活（send/recv），只有阶段边界通信。<b>气泡（空闲）</b>是主要代价 → 用 micro-batch 填满流水线。适合跨机（通信量小）。`;
          return g;
        }
      },
      EP: {
        title: "专家并行 Expert Parallel —— MoE 专家分散到多卡",
        render: function () {
          const g = el("div");
          g.style.cssText = "display:grid;grid-template-columns:repeat(4,1fr);gap:10px;";
          const expertsPerGpu = 8;
          for (let i = 0; i < 4; i++) {
            const c = el("div");
            c.style.cssText = `border:1px solid ${PALETTE[i]}66;background:${PALETTE[i]}14;border-radius:10px;padding:12px;`;
            let gridHtml = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin:8px 0;">';
            for (let e = 0; e < expertsPerGpu; e++) {
              const active = (i * expertsPerGpu + e) % 5 === 0;
              gridHtml += `<div title="Expert ${i * expertsPerGpu + e}" style="height:16px;border-radius:3px;background:${active ? PALETTE[i] : PALETTE[i] + "40"};${active ? "box-shadow:0 0 6px " + PALETTE[i] : ""}"></div>`;
            }
            gridHtml += "</div>";
            c.innerHTML = `<div style="font-weight:700;color:${PALETTE[i]};">GPU ${i}</div>${gridHtml}<div style="font-size:11.5px;color:var(--text-dim);">托管 Expert ${i * 8}~${i * 8 + 7}<br><span style="color:${PALETTE[i]};">高亮 = 本步被路由命中</span></div>`;
            g.appendChild(c);
          }
          note.innerHTML = `<span class="hl">EP：</span>MoE 模型的 32 个专家权重分散到 4 卡，每步由路由器（router/gate）决定 token 发给哪些专家 → All-to-All 通信。vLLM 中 EP 常与 TP/DP 组合（如 <code>--enable-expert-parallel</code>），DeepSeek-V3 这类大 MoE 必备。`;
          return g;
        }
      },
    };
    let cur = "TP";
    function show() {
      tabs.querySelectorAll(".viz-btn").forEach(function (b) {
        const on = b.dataset.k === cur;
        b.classList.toggle("primary", on);
      });
      stage.innerHTML = "";
      stage.appendChild(TABS[cur].render());
    }
    Object.keys(TABS).forEach(function (k) {
      const b = el("button", "viz-btn", k + " 并行");
      b.dataset.k = k;
      b.addEventListener("click", function () { cur = k; show(); });
      tabs.appendChild(b);
    });
    body.appendChild(tabs); body.appendChild(stage); body.appendChild(note);
    show();
  };

  /* ============================================================
     8. spec-decode —— 投机解码（草稿-验证）动画
     ============================================================ */
  REGISTRY["spec-decode"] = function (box) {
    const body = box.querySelector(".viz-body");
    const tb = el("div", "viz-toolbar");
    const play = el("button", "viz-btn primary", "▶ 播放一轮");
    tb.appendChild(play);
    const lane = el("div");
    lane.style.cssText = "display:flex;gap:5px;flex-wrap:wrap;min-height:96px;align-items:center;padding:10px;background:var(--bg-code);border-radius:10px;border:1px solid var(--border-soft);";
    const note = el("div", "viz-note");
    const tokens = [];
    function addTok(txt, color, glow) {
      const t = el("div");
      t.style.cssText = `padding:6px 10px;border-radius:8px;font-family:var(--mono);font-size:13px;border:1.5px solid ${color};color:${color};background:${color}14;transition:all .3s;${glow ? "box-shadow:0 0 10px " + color + ";font-weight:700;" : ""}`;
      t.textContent = txt;
      lane.appendChild(t); tokens.push(t);
      return t;
    }
    const ROUNDS = [
      { draft: ["北京", "是", "中国"], verify: [1, 1, 1], bonus: "的" },
      { draft: ["首都", "，", "人口"], verify: [1, 1, 0], bonus: null },
      { draft: ["约", "2100", "万"], verify: [1, 1, 1], bonus: "人" },
    ];
    let ri = 0;
    play.addEventListener("click", function () {
      if (ri >= ROUNDS.length) {
        note.innerHTML = "🎯 本轮投机解码结束：目标模型只做了 <b>3 次前向</b>，却产出了 <b>10 个 token</b> —— 这就是投机解码用「小模型草稿 + 大模型验证」换加速的原理。验证可一次并行打分，所以吞吐显著提升。";
        play.disabled = true; return;
      }
      const r = ROUNDS[ri];
      note.innerHTML = `① 草稿模型（1B）快速连猜 3 个 token…`;
      const drafted = [];
      r.draft.forEach(function (d, i) {
        setTimeout(function () { drafted.push(addTok(d, "#a78bfa", true)); }, i * 350);
      });
      setTimeout(function () {
        note.innerHTML = "② 目标模型（70B）<b>一次前向并行验证</b>这 3 个草稿 token…";
        drafted.forEach(function (t) { t.style.boxShadow = "0 0 14px #f59e0b"; t.style.borderColor = "#f59e0b"; t.style.color = "#f59e0b"; });
      }, 1200);
      setTimeout(function () {
        let accepted = 0;
        drafted.forEach(function (t, i) {
          if (r.verify[i]) { t.style.borderColor = "#22c55e"; t.style.color = "#22c55e"; t.style.boxShadow = ""; accepted++; }
          else { t.style.borderColor = "#f87171"; t.style.color = "#f87171"; t.style.textDecoration = "line-through"; t.style.opacity = ".55"; }
        });
        note.innerHTML = `③ 验证结果：前 ${accepted} 个被接受（绿），第 ${accepted + 1} 个被拒绝（红）→ 回退后由目标模型自己补 1 个 token。<b>本轮：目标模型 1 次前向 = 产出 ${accepted + 1} 个 token</b>。`;
        if (r.bonus) addTok(r.bonus, "#22c55e", true);
        ri++;
      }, 2300);
    });
    body.appendChild(tb); body.appendChild(lane); body.appendChild(note);
  };

  /* ============================================================
     9. waste-compare —— 传统三种浪费 vs 分页（ch01/ch04 用）
     ============================================================ */
  REGISTRY["waste-compare"] = function (box, cfg) {
    const body = box.querySelector(".viz-body");
    const modes = cfg.modes || [
      { name: "按最大长度预留", desc: "请求只要 50 token，却按 4096 锁死一整条", waste: 92 },
      { name: "连续大块分配", desc: "300MB 碎片明明空闲，400MB 请求却放不进（必须连续）", waste: 60 },
      { name: "静态批处理", desc: "R1/R3 早已完成，GPU 空转 20+ 步等最慢的 R2", waste: 45 },
      { name: "vLLM 分页 + 连续批处理", desc: "用多少分多少，完成即让位 → 利用率 90%+", waste: 8, good: true },
    ];
    modes.forEach(function (m) {
      const row = el("div");
      row.style.cssText = "margin:14px 0;";
      row.innerHTML = `<div style="display:flex;justify-content:space-between;font-size:14px;margin-bottom:5px;">
          <b style="color:${m.good ? "var(--accent-2)" : "var(--accent-3)"};">${esc(m.name)}</b>
          <span style="font-family:var(--mono);color:${m.good ? "var(--accent-2)" : "var(--danger)"};">浪费 ${m.waste}%</span></div>
        <div style="height:20px;border-radius:6px;background:var(--bg-code);border:1px solid var(--border-soft);overflow:hidden;">
          <div style="height:100%;width:0;border-radius:6px;background:${m.good ? "linear-gradient(90deg,#22c55e,#4ade80)" : "linear-gradient(90deg,#f59e0b,#f87171)"};transition:width 1.2s cubic-bezier(.2,.8,.2,1);"></div></div>
        <div style="font-size:12.5px;color:var(--text-dim);margin-top:4px;">${esc(m.desc)}</div>`;
      body.appendChild(row);
      const fill = row.querySelector("div div div");
      requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = m.waste + "%"; }));
    });
  };

  /* ============================================================
     10. block-map —— KV 块 16 token 的逻辑/物理映射静态示意
     cfg: { rows: [{req, blocks:[phys...]}] }（可选，缺省用内置示例）
     ============================================================ */
  REGISTRY["block-map"] = function (box, cfg) {
    const body = box.querySelector(".viz-body");
    const data = cfg.rows || [
      { req: "请求 A", blocks: [7, 2, 11] },
      { req: "请求 B", blocks: [0, 2, 5, 9] },
    ];
    const grid = el("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(12,1fr);gap:6px;";
    const total = 12;
    const cells = [];
    for (let i = 0; i < total; i++) {
      const c = el("div");
      c.style.cssText = "aspect-ratio:1.4;border-radius:6px;border:1px dashed var(--border);display:flex;align-items:center;justify-content:center;font-size:11px;font-family:var(--mono);color:var(--text-faint);background:var(--bg-code);";
      c.textContent = "#" + i;
      grid.appendChild(c); cells.push(c);
    }
    data.forEach(function (r, ri) {
      r.blocks.forEach(function (b, bi) {
        const c = cells[b];
        c.style.background = PALETTE[ri] + "55";
        c.style.border = "1px solid " + PALETTE[ri];
        c.style.color = "#fff";
        c.style.fontWeight = "700";
        c.textContent = r.req.replace("请求 ", "") + "·" + bi;
      });
    });
    body.appendChild(grid);
    body.appendChild(el("div", "viz-legend", data.map(function (r, i) {
      return `<span class="key"><span class="sw" style="background:${PALETTE[i]}"></span>${esc(r.req)}：逻辑连续 → 物理散落 [${r.blocks.join(", ")}]</span>`;
    }).join("")));
    body.appendChild(el("div", "viz-note", `注意 请求 A 和 B 都引用了 <span class="hl">物理块 #2</span> —— 相同前缀（如同一个 system prompt）只存一份，引用计数管理生命周期。`));
  };

  /* ============================================================
     自动初始化
     ============================================================ */
  function init() {
    document.querySelectorAll(".viz[data-viz]").forEach(function (box) {
      const name = box.getAttribute("data-viz");
      const cfgTag = box.querySelector("script.viz-cfg");
      let cfg = {};
      if (cfgTag) { try { cfg = JSON.parse(cfgTag.textContent); } catch (e) { console.warn("viz cfg parse fail", name); } }
      const fn = REGISTRY[name];
      if (fn) { try { fn(box, cfg); } catch (e) { console.error("viz error:", name, e); } }
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
