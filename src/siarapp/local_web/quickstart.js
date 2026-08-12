(() => {
  "use strict";

  const MARKERS = {
    "$": "cmd",
    ">": "ask",
    "!": "flag",
    "+": "ok",
    ".": "dim",
    "-": "rule"
  };

  function parse(raw) {
    const head = {};
    const slides = [];
    let slide = null;
    let inTerm = false;

    for (const rawLine of String(raw).split("\n")) {
      const trimmed = rawLine.trim();

      if (trimmed === "--- slide ---") {
        slide = { title: "", rail: "", note: "", lines: [] };
        slides.push(slide);
        inTerm = false;
        continue;
      }

      if (!slide) {
        const m = trimmed.match(/^([a-z]+):\s*(.*)$/);
        if (m) head[m[1]] = m[2];
        continue;
      }

      if (!inTerm) {
        if (trimmed === "term:") { inTerm = true; continue; }
        const m = trimmed.match(/^(title|rail|note):\s*(.*)$/);
        if (m) slide[m[1]] = m[2];
        continue;
      }

      if (trimmed === "") { slide.lines.push({ k: "b" }); continue; }

      const marker = rawLine[0];
      const cls = Object.prototype.hasOwnProperty.call(MARKERS, marker) && rawLine[1] === " "
        ? MARKERS[marker]
        : null;
      const text = cls ? rawLine.slice(2) : rawLine;

      if (cls === "cmd") {
        slide.lines.push({ k: "c", t: text });
      } else if (cls === "ask") {
        const at = text.indexOf("::");
        if (at === -1) slide.lines.push({ k: "o", t: text });
        else slide.lines.push({ k: "a", label: text.slice(0, at).trimEnd() + " ", t: text.slice(at + 2).trim() });
      } else {
        slide.lines.push({ k: cls || "o", t: text });
      }
    }

    for (const s of slides) {
      while (s.lines.length && s.lines[s.lines.length - 1].k === "b") s.lines.pop();
      if (!s.rail) s.rail = s.title;
    }
    return { head, slides };
  }

  function markup(text) {
    const esc = String(text)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return esc
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/~([^~]+)~/g, "<code>$1</code>");
  }

  const src = typeof window.QUICKSTART_TEXT === "string" ? window.QUICKSTART_TEXT : "";
  const { head, slides } = parse(src);

  const term = document.getElementById("term");
  const railEl = document.getElementById("rail");
  const stepEl = document.getElementById("slide-step");
  const titleEl = document.getElementById("slide-title");
  const noteEl = document.getElementById("note");
  const counterEl = document.getElementById("counter");
  const prevBtn = document.getElementById("prev");
  const nextBtn = document.getElementById("next");

  if (!slides.length) {
    term.textContent = "page-text.js did not load, or has no slides in it.";
    return;
  }

  const PROMPT = (head.prompt || "user@vi:~$") + " ";

  document.getElementById("eyebrow").textContent = head.eyebrow || "";
  document.getElementById("headline").textContent = head.headline || "";
  document.getElementById("intro").innerHTML = markup(head.intro || "");
  document.getElementById("rail-title").textContent = head.railtitle || "Steps";
  document.getElementById("term-title").textContent = head.window || "user@vi: ~";
  document.getElementById("colophon").innerHTML = markup(head.footer || "");

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const CLASS = { o: "", dim: "dim", flag: "flag", ok: "ok", rule: "rule" };

  let index = 0;
  let seq = 0;
  let finish = null;

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function buildRail() {
    slides.forEach((s, i) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      const num = document.createElement("span");
      num.className = "num";
      num.textContent = String(i + 1).padStart(2, "0");
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = s.rail;
      b.append(num, label);
      b.addEventListener("click", () => show(i));
      li.appendChild(b);
      railEl.appendChild(li);
    });
  }

  function line(cls) {
    const el = document.createElement("span");
    el.className = "ln" + (cls ? " " + cls : "");
    term.appendChild(el);
    return el;
  }

  function span(parent, cls, text) {
    const s = document.createElement("span");
    if (cls) s.className = cls;
    s.textContent = text;
    parent.appendChild(s);
    return s;
  }

  function tail() {
    const el = line("");
    const cur = document.createElement("span");
    cur.className = "cursor";
    el.appendChild(cur);
  }

  async function type(parent, cls, text, mine) {
    const s = span(parent, cls, "");
    if (reduced) { s.textContent = text; return; }
    for (let i = 1; i <= text.length; i++) {
      if (mine !== seq) return;
      s.textContent = text.slice(0, i);
      await sleep(text.length > 60 ? 9 : 17);
    }
  }

  function paint(slide) {
    term.textContent = "";
    for (const l of slide.lines) {
      if (l.k === "b") { line(""); continue; }
      const el = line(CLASS[l.k] || "");
      if (l.k === "c") { span(el, "prompt", PROMPT); span(el, "cmd", l.t); }
      else if (l.k === "a") { span(el, "dim", l.label); span(el, "ask", l.t); }
      else el.textContent = l.t;
    }
    tail();
  }

  async function render(slide) {
    const mine = ++seq;
    term.textContent = "";

    if (reduced) { finish = null; paint(slide); return; }

    finish = () => { if (mine === seq) { seq++; paint(slide); finish = null; } };

    for (const l of slide.lines) {
      if (mine !== seq) return;
      if (l.k === "b") { line(""); await sleep(40); continue; }
      const el = line(CLASS[l.k] || "");
      if (l.k === "c") {
        span(el, "prompt", PROMPT);
        await type(el, "cmd", l.t, mine);
        await sleep(220);
      } else if (l.k === "a") {
        span(el, "dim", l.label);
        await sleep(120);
        await type(el, "ask", l.t, mine);
        await sleep(160);
      } else {
        el.classList.add("reveal");
        el.textContent = l.t;
        await sleep(60);
      }
    }
    if (mine !== seq) return;
    tail();
    finish = null;
  }

  function show(i) {
    index = Math.max(0, Math.min(slides.length - 1, i));
    const slide = slides[index];

    stepEl.textContent = "Step " + String(index + 1).padStart(2, "0") + " of " + slides.length;
    titleEl.textContent = slide.title;
    noteEl.innerHTML = markup(slide.note);
    counterEl.textContent = String(index + 1).padStart(2, "0") + " / " + slides.length;

    Array.from(railEl.children).forEach((li, n) => {
      if (n === index) li.setAttribute("aria-current", "step");
      else li.removeAttribute("aria-current");
    });

    prevBtn.disabled = index === 0;
    nextBtn.disabled = index === slides.length - 1;

    render(slide);
  }

  function step(delta) {
    if (finish) { finish(); return; }
    show(index + delta);
  }

  prevBtn.addEventListener("click", () => step(-1));
  nextBtn.addEventListener("click", () => step(1));

  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown") { e.preventDefault(); step(1); }
    else if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); step(-1); }
    else if (e.key === "Home") { e.preventDefault(); show(0); }
    else if (e.key === "End") { e.preventDefault(); show(slides.length - 1); }
  });

  buildRail();
  show(0);
})();
