/* vLLM 教程网页版 · 通用交互（导航 / 阅读进度 / 代码复制 / MathJax） */
(function () {
  "use strict";

  /* ---------- MathJax 离线渲染（$...$ 行内，$$...$$ 块级） ---------- */
  window.MathJax = {
    tex: {
      inlineMath: [["$", "$"], ["\\(", "\\)"]],
      displayMath: [["$$", "$$"], ["\\[", "\\]"]],
      processEscapes: true
    },
    svg: { fontCache: "global" },
    options: { skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"] },
    startup: { typeset: true }
  };

  /* ---------- 移动端侧边栏 ---------- */
  var menuBtn = document.querySelector(".menu-btn");
  var sidebar = document.querySelector(".sidebar");
  if (menuBtn && sidebar) {
    menuBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      sidebar.classList.toggle("open");
    });
    document.addEventListener("click", function (e) {
      if (sidebar.classList.contains("open") &&
          !sidebar.contains(e.target) && e.target !== menuBtn) {
        sidebar.classList.remove("open");
      }
    });
  }

  /* ---------- 阅读进度 ---------- */
  var bar = document.querySelector(".reading-progress");
  if (bar) {
    window.addEventListener("scroll", function () {
      var h = document.documentElement;
      var p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
      bar.style.width = (p * 100).toFixed(1) + "%";
    }, { passive: true });
  }

  /* ---------- 侧边栏当前章高亮 ---------- */
  var here = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".sidebar nav a").forEach(function (a) {
    var href = a.getAttribute("href").split("/").pop();
    if (href === here) a.classList.add("active");
  });

  /* ---------- 轻量语法高亮（python / bash / c / json 等，离线可用） ---------- */
  var KW_PY = /\b(def|class|return|import|from|as|if|elif|else|for|while|in|not|and|or|is|None|True|False|try|except|finally|with|lambda|yield|pass|break|continue|raise|global|nonlocal|assert|async|await|del|self)\b/g;
  function hlCode(src, lang) {
    var escd = src.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    var commentRe = lang === "bash" || lang === "sh" || lang === "yaml" || lang === "toml"
      ? /(#[^\n]*)/g
      : (lang === "c" || lang === "cpp" || lang === "cuda" ? /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)/g : /(#[^\n]*)/g);
    var parts = escd.split(commentRe);
    var isPy = lang === "python" || lang === "py";
    for (var i = 0; i < parts.length; i++) {
      if (i % 2 === 1) { parts[i] = '<span style="color:#6a7a8f;font-style:italic;">' + parts[i] + "</span>"; continue; }
      parts[i] = parts[i]
        .replace(/("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')/g, '<span style="color:#a5d6a7;">$1</span>')
        .replace(/\b(0x[0-9a-fA-F]+|\d+\.?\d*(?:e[+-]?\d+)?)\b/g, '<span style="color:#f7bd7f;">$1</span>');
      if (isPy || !lang) parts[i] = parts[i].replace(KW_PY, '<span style="color:#c792ea;font-weight:600;">$1</span>');
      if (lang === "bash" || lang === "sh" || lang === "shell") {
        parts[i] = parts[i].replace(/^(\s*)(pip|pip3|python|python3|uv|docker|curl|git|vllm|serve|source|cd|export|sudo|apt|apt-get|nvidia-smi|bash|sh|echo|conda|nvcc)\b/m, '$1<span style="color:#82aaff;font-weight:600;">$2</span>');
      }
      if (lang === "c" || lang === "cpp" || lang === "cuda") {
        parts[i] = parts[i].replace(/\b(void|int|float|double|char|const|static|inline|struct|enum|template|typename|using|namespace|public|private|__global__|__device__|__host__|T|kernel|auto|for|if|else|return|include|define|pragma)\b/g, '<span style="color:#c792ea;font-weight:600;">$1</span>');
      }
    }
    return parts.join("");
  }
  document.querySelectorAll("pre > code").forEach(function (code) {
    var pre = code.parentElement;
    if (code.children.length) return; // 已高亮
    var cls = (code.className.match(/language-([\w+-]+)/) || [])[1] || "";
    if (!cls) {
      var txt = (code.textContent || "").slice(0, 80);
      if (/^\s*(\$|#|pip |python |uv |docker |curl |git |export |source )/m.test(txt)) cls = "bash";
      else if (/def |class |import |print\(/.test(txt)) cls = "python";
      else if (/__global__|#include/.test(txt)) cls = "c";
    }
    code.innerHTML = hlCode(code.textContent, cls.toLowerCase());
  });

  /* ---------- 代码复制按钮 ---------- */
  document.querySelectorAll("pre").forEach(function (pre) {
    var head = pre.previousElementSibling;
    var btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "复制";
    btn.addEventListener("click", function () {
      var code = pre.querySelector("code");
      var text = (code || pre).innerText;
      (navigator.clipboard ? navigator.clipboard.writeText(text)
        : Promise.reject()).then(function () {
        btn.textContent = "已复制 ✓"; btn.classList.add("ok");
        setTimeout(function () { btn.textContent = "复制"; btn.classList.remove("ok"); }, 1600);
      }).catch(function () {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); btn.textContent = "已复制 ✓"; btn.classList.add("ok"); }
        catch (e) { btn.textContent = "复制失败"; }
        document.body.removeChild(ta);
        setTimeout(function () { btn.textContent = "复制"; btn.classList.remove("ok"); }, 1600);
      });
    });
    if (head && head.classList.contains("code-head")) {
      head.appendChild(btn);
    } else {
      var wrap = document.createElement("div");
      wrap.style.position = "absolute"; wrap.style.top = "8px"; wrap.style.right = "8px";
      wrap.appendChild(btn);
      pre.style.position = "relative";
      pre.appendChild(wrap);
    }
  });
})();
