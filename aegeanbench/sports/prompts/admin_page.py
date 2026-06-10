"""
Single-file HTML admin page for editing the global prompt addendum.

Served at GET /admin. The page hits the JSON admin endpoints
(/api/v1/admin/prompts/global) with a token the user pastes in.
"""

ADMIN_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>AegeanBench · Prompt 控制台</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 24px;
    background: #0f1419; color: #e6edf3;
    max-width: 960px; margin-left: auto; margin-right: auto;
  }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #8b949e; font-size: 13px; margin-bottom: 24px; }
  label { display: block; margin: 16px 0 6px; font-size: 13px; color: #c9d1d9; }
  input[type=password], textarea {
    width: 100%; padding: 10px 12px; border-radius: 6px;
    background: #161b22; color: #e6edf3; border: 1px solid #30363d;
    font-family: ui-monospace, SFMono-Regular, monospace; font-size: 13px;
  }
  textarea { min-height: 240px; resize: vertical; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
  button {
    padding: 10px 18px; border-radius: 6px; border: none; cursor: pointer;
    font-size: 14px; font-weight: 500;
  }
  .primary { background: #2da44e; color: #fff; }
  .primary:hover { background: #2c974b; }
  .secondary { background: #21262d; color: #e6edf3; border: 1px solid #30363d; }
  .secondary:hover { background: #30363d; }
  .danger { background: #da3633; color: #fff; }
  .danger:hover { background: #b62324; }
  .toast { padding: 10px 14px; border-radius: 6px; margin-top: 14px; font-size: 13px; }
  .toast.ok  { background: #1f4d26; color: #c9f1d3; }
  .toast.err { background: #5d1d1d; color: #ffcfcf; }
  .meta { color: #8b949e; font-size: 12px; margin-top: 8px; }
  h2 { font-size: 15px; color: #c9d1d9; margin: 32px 0 8px; }
  .history-item {
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px; margin-bottom: 8px;
  }
  .history-item pre {
    background: #0d1117; padding: 8px 10px; border-radius: 4px;
    margin: 6px 0 0; white-space: pre-wrap; font-size: 12px;
    max-height: 120px; overflow: auto;
  }
  .history-meta { font-size: 11px; color: #8b949e; }
  .empty { color: #6e7681; font-style: italic; font-size: 13px; }
</style>
</head>
<body>
<h1>AegeanBench · Prompt 控制台</h1>
<div class="sub">编辑全局指令，所有 agent 的回答都会附带这段内容。改完点保存即时生效。</div>

<label for="token">管理员 Token <span style="color:#6e7681;">(从运维拿)</span></label>
<input id="token" type="password" placeholder="X-Admin-Token" />

<label for="prompt">全局指令</label>
<textarea id="prompt" placeholder="例如：所有回答严格按以下三行格式：&#10;论点：<一句话给出你的判断>&#10;证据：<一句话引用具体数据>&#10;保留：<一句话指出缺什么或不确定什么>"></textarea>
<div class="meta" id="current-meta">加载中…</div>

<div class="row">
  <button class="primary" onclick="save()">保存并立即生效</button>
  <button class="secondary" onclick="load()">刷新</button>
  <button class="danger" onclick="rollback()">回滚到上一版</button>
</div>
<div id="toast"></div>

<h2>历史版本（最近 5 条）</h2>
<div id="history"></div>

<script>
const API = "/api/v1/admin/prompts/global";

function toast(msg, kind) {
  const t = document.getElementById("toast");
  t.className = "toast " + kind;
  t.textContent = msg;
  setTimeout(() => { t.className = ""; t.textContent = ""; }, 4000);
}

function renderState(state) {
  const cur = state.current;
  if (cur) {
    document.getElementById("prompt").value = cur.prompt || "";
    document.getElementById("current-meta").textContent =
      "当前版本 · " + (cur.updated_at || "?") + " · 修改人 " + (cur.updated_by || "?");
  } else {
    document.getElementById("prompt").value = "";
    document.getElementById("current-meta").textContent = "尚未配置全局指令";
  }
  const h = document.getElementById("history");
  const items = state.history || [];
  if (items.length === 0) {
    h.innerHTML = '<div class="empty">没有历史版本</div>';
    return;
  }
  h.innerHTML = items.map(it => `
    <div class="history-item">
      <div class="history-meta">${it.updated_at || "?"} · ${it.updated_by || "?"}</div>
      <pre>${(it.prompt || "").replace(/[<&]/g, c => c === "<" ? "&lt;" : "&amp;")}</pre>
    </div>
  `).join("");
}

async function load() {
  try {
    const r = await fetch(API);
    if (!r.ok) throw new Error("GET failed " + r.status);
    renderState(await r.json());
  } catch (e) {
    toast("加载失败：" + e.message, "err");
  }
}

async function save() {
  const token = document.getElementById("token").value.trim();
  const prompt = document.getElementById("prompt").value;
  if (!token) { toast("请填入管理员 Token", "err"); return; }
  try {
    const r = await fetch(API, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Admin-Token": token,
      },
      body: JSON.stringify({ prompt, updated_by: "admin@web" }),
    });
    if (r.status === 401) { toast("Token 错误", "err"); return; }
    if (!r.ok) { toast("保存失败：HTTP " + r.status, "err"); return; }
    renderState(await r.json());
    toast("已保存并立即生效", "ok");
  } catch (e) {
    toast("保存失败：" + e.message, "err");
  }
}

async function rollback() {
  const token = document.getElementById("token").value.trim();
  if (!token) { toast("请填入管理员 Token", "err"); return; }
  if (!confirm("确定要回滚到上一版吗？")) return;
  try {
    const r = await fetch(API + "/rollback", {
      method: "POST",
      headers: { "X-Admin-Token": token },
    });
    if (r.status === 401) { toast("Token 错误", "err"); return; }
    if (!r.ok) { toast("回滚失败：HTTP " + r.status, "err"); return; }
    renderState(await r.json());
    toast("已回滚", "ok");
  } catch (e) {
    toast("回滚失败：" + e.message, "err");
  }
}

load();
</script>
</body>
</html>
"""
