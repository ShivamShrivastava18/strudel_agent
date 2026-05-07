// Strudel Agent — frontend logic
//
// Streams agent events from /api/generate (SSE), renders them in the left
// thinking panel, and routes the final code into the editor textarea.
// The strudel.cc iframe gets reloaded with `#<base64(code)>` whenever the
// user clicks Play.

const $ = (id) => document.getElementById(id);
const promptEl = $("prompt");
const goBtn = $("go");
const playBtn = $("play");
const openBtn = $("open-strudel");
const codeEl = $("code");
const thinkingEl = $("thinking");
const statusEl = $("status");
const iframeEl = $("strudel");
const validationEl = $("validation");
const configEl = $("config");

let generating = false;
let statusTimer = null;
let startedAt = 0;
let lastEventAt = 0;

// ---------- Config ----------
fetch("/api/config")
  .then((r) => r.json())
  .then((c) => {
    configEl.textContent = `${c.provider} · ${c.model} · max_iters=${c.max_iters}`;
  })
  .catch(() => (configEl.textContent = "config unavailable"));

// ---------- Event rendering ----------
function pushEvent(type, label, body) {
  const div = document.createElement("div");
  div.className = `event ${type}`;
  const lbl = document.createElement("div");
  lbl.className = "label";
  lbl.textContent = label;
  const pre = document.createElement("pre");
  pre.textContent = body;
  div.appendChild(lbl);
  div.appendChild(pre);
  thinkingEl.appendChild(div);
  thinkingEl.scrollTop = thinkingEl.scrollHeight;
}

function clearThinking() {
  thinkingEl.innerHTML = "";
}

function summarize(obj, maxLen = 280) {
  let s;
  try {
    s = typeof obj === "string" ? obj : JSON.stringify(obj);
  } catch {
    s = String(obj);
  }
  return s.length > maxLen ? s.slice(0, maxLen) + "…" : s;
}

// ---------- SSE generation ----------
async function generate(prompt) {
  if (generating) return;
  generating = true;
  startedAt = Date.now();
  lastEventAt = startedAt;
  goBtn.disabled = true;
  goBtn.textContent = "…";
  clearThinking();
  startStatusTicker();

  console.log("[agent] POST /api/generate", { prompt });

  // Use POST + ReadableStream parsing because EventSource doesn't support POST.
  let res;
  try {
    res = await fetch("/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify({ prompt }),
    });
  } catch (e) {
    pushEvent("error", "fetch error", String(e));
    finish();
    return;
  }
  console.log("[agent] response", res.status, res.headers.get("content-type"));
  if (!res.ok) {
    pushEvent("error", `HTTP ${res.status}`, await res.text());
    finish();
    return;
  }
  if (!res.body) {
    pushEvent("error", "no stream body", "Browser returned no readable response body.");
    finish();
    return;
  }

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    // Normalise CRLF → LF so framing works regardless of server choice.
    buf = buf.replace(/\r\n/g, "\n");
    // SSE frames are separated by \n\n
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      handleFrame(frame);
    }
  }
  // flush trailing frame if any
  if (buf.trim()) handleFrame(buf);
  finish();
}

function startStatusTicker() {
  clearInterval(statusTimer);
  statusTimer = setInterval(() => {
    if (!generating) return;
    const elapsed = Math.floor((Date.now() - startedAt) / 1000);
    const sinceEvent = Math.floor((Date.now() - lastEventAt) / 1000);
    statusEl.textContent =
      `running… ${elapsed}s` +
      (sinceEvent > 5 ? ` (waiting on LLM, ${sinceEvent}s since last event)` : "");
  }, 500);
}

function handleFrame(frame) {
  // Each frame is one or more lines like "data: {...}"
  const lines = frame.split("\n");
  for (const line of lines) {
    if (!line.startsWith("data:")) continue;
    const payload = line.slice(5).trim();
    if (!payload) continue;
    let ev;
    try {
      ev = JSON.parse(payload);
    } catch {
      continue;
    }
    handleEvent(ev);
  }
}

function handleEvent(ev) {
  console.log("[agent] event", ev);
  lastEventAt = Date.now();
  switch (ev.type) {
    case "starting":
      pushEvent("assistant", "connected", `${ev.provider} · ${ev.model}`);
      break;
    case "heartbeat":
      // silent — the status ticker shows elapsed time
      break;
    case "vibe": {
      const matches = (ev.matches || []).map((m) => `${m.name} (${m.type})`).join(", ");
      pushEvent("vibe", "vibe spec", `matched: ${matches || "none"}\n${summarize(ev.spec, 600)}`);
      break;
    }
    case "thinking":
      pushEvent("thinking", `step ${ev.step} thinking`, summarize(ev.content, 600));
      break;
    case "assistant":
      pushEvent("assistant", `step ${ev.step} assistant`, summarize(ev.content, 600));
      break;
    case "tool_call":
      pushEvent("tool_call", `step ${ev.step} → ${ev.name}`, summarize(ev.arguments));
      break;
    case "tool_result":
      pushEvent("tool_result", `step ${ev.step} ← ${ev.name}`, summarize(ev.result));
      break;
    case "final":
      pushEvent("final", "final answer", ev.explanation || "");
      codeEl.value = ev.code || "";
      validate(ev.code || "");
      // auto-load into iframe
      loadIntoStrudel(ev.code || "");
      break;
    case "error":
      pushEvent("error", "error", ev.message || "");
      break;
    case "done":
      // handled in finish()
      break;
    default:
      pushEvent("assistant", ev.type, summarize(ev));
  }
}

function finish() {
  generating = false;
  goBtn.disabled = false;
  goBtn.textContent = "Generate";
  clearInterval(statusTimer);
  statusTimer = null;
  const elapsed = startedAt ? Math.floor((Date.now() - startedAt) / 1000) : 0;
  statusEl.textContent = elapsed ? `done in ${elapsed}s` : "";
}

// ---------- Validation ----------
async function validate(code) {
  if (!code || !code.trim()) {
    validationEl.textContent = "";
    validationEl.className = "muted";
    return;
  }
  try {
    const r = await fetch("/api/validate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ code }),
    });
    const v = await r.json();
    if (v.ok) {
      validationEl.textContent = "✓ valid";
      validationEl.className = "ok";
    } else {
      const issues = (v.issues || []).map((i) => `${i.kind}: ${i.name || ""}`).join("; ");
      validationEl.textContent = "✗ " + (issues || "issues");
      validationEl.className = "err";
    }
  } catch {
    validationEl.textContent = "validate failed";
    validationEl.className = "err";
  }
}

// debounce live-edit validation
let valTimer = null;
codeEl.addEventListener("input", () => {
  clearTimeout(valTimer);
  valTimer = setTimeout(() => validate(codeEl.value), 350);
});

// ---------- Strudel iframe ----------
function strudelUrl(code) {
  // strudel.cc reads the URL fragment and base64-decodes it
  const b64 = btoa(unescape(encodeURIComponent(code || "")));
  return `https://strudel.cc/#${b64}`;
}

function loadIntoStrudel(code) {
  iframeEl.src = strudelUrl(code);
}

playBtn.addEventListener("click", () => loadIntoStrudel(codeEl.value));
openBtn.addEventListener("click", () => window.open(strudelUrl(codeEl.value), "_blank"));

// ---------- Submit ----------
goBtn.addEventListener("click", () => {
  const p = promptEl.value.trim();
  if (p) generate(p);
});
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") goBtn.click();
});
