"""RAG Web UI — proxies API calls to avoid CORS issues."""
import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response

app = FastAPI()

# Internal API URL (container-to-container, never exposed to browser)
_API = os.environ.get("API_BASE_URL", "http://localhost:8000")


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    """Transparent proxy — forwards all /api/* requests to the backend API."""
    url = f"{_API}/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.request(
            method=request.method,
            url=url,
            content=body,
            headers=headers,
            params=dict(request.query_params),
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>RAG Query UI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f3f4f6;color:#111;height:100vh;display:flex;flex-direction:column}

/* ── Login overlay ── */
#login-overlay{position:fixed;inset:0;background:rgba(15,23,42,.85);display:flex;align-items:center;justify-content:center;z-index:100}
#login-box{background:#fff;border-radius:16px;padding:40px 48px;width:360px;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,.3)}
#login-box h1{font-size:1.5rem;margin-bottom:6px}
#login-box p{color:#6b7280;font-size:.9rem;margin-bottom:28px}
.login-btn{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;padding:11px 16px;border:1px solid #d1d5db;border-radius:8px;background:#fff;cursor:pointer;font-size:.95rem;font-weight:500;margin-bottom:12px;text-decoration:none;color:#111;transition:background .15s}
.login-btn:hover{background:#f9fafb}
.login-btn.google{border-color:#4285f4;color:#4285f4}
.login-btn.microsoft{border-color:#0078d4;color:#0078d4}
.login-btn svg{flex-shrink:0}

/* ── Header ── */
header{background:#1e3a8a;color:#fff;padding:12px 20px;display:flex;align-items:center;gap:10px;flex-shrink:0}
header span.title{font-size:1.1rem;font-weight:600;flex:1}
#user-info{display:flex;align-items:center;gap:8px;font-size:.85rem}
#user-avatar{width:28px;height:28px;border-radius:50%;border:2px solid rgba(255,255,255,.4)}
#logout-btn{background:rgba(255,255,255,.15);border:none;color:#fff;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:.8rem}
#logout-btn:hover{background:rgba(255,255,255,.25)}
#status-badge{font-size:.75rem;background:rgba(255,255,255,.12);padding:3px 10px;border-radius:20px;cursor:pointer}
#status-badge.ok{background:rgba(134,239,172,.25)}
#status-badge.err{background:rgba(252,165,165,.25)}

/* ── Layout ── */
main{display:flex;flex:1;overflow:hidden}
#chat-col{display:flex;flex-direction:column;flex:1;padding:14px;gap:10px;overflow:hidden}
#messages{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding-right:4px}
.msg{max-width:82%;padding:10px 14px;border-radius:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word;font-size:.9rem}
.user{align-self:flex-end;background:#1e3a8a;color:#fff;border-bottom-right-radius:3px}
.bot{align-self:flex-start;background:#fff;border:1px solid #e5e7eb;border-bottom-left-radius:3px}
.thinking{color:#9ca3af;font-style:italic}
#input-row{display:flex;gap:8px}
#question{flex:1;padding:9px 13px;border:1px solid #d1d5db;border-radius:8px;font-size:.9rem;resize:none;height:44px}
#question:focus{outline:none;border-color:#1e3a8a}
button{padding:9px 16px;border:none;border-radius:8px;cursor:pointer;font-size:.88rem;font-weight:500}
#send-btn{background:#1e3a8a;color:#fff}
#clear-btn{background:#e5e7eb;color:#374151}

/* ── Sidebar ── */
#sidebar{width:310px;background:#fff;border-left:1px solid #e5e7eb;display:flex;flex-direction:column;overflow:hidden}
.tab-bar{display:flex;border-bottom:1px solid #e5e7eb;flex-shrink:0}
.tab{flex:1;padding:9px 4px;text-align:center;font-size:.8rem;font-weight:500;cursor:pointer;color:#6b7280;border-bottom:2px solid transparent}
.tab.active{color:#1e3a8a;border-bottom-color:#1e3a8a}
.tab-panel{display:none;flex-direction:column;gap:12px;padding:14px;overflow-y:auto;flex:1}
.tab-panel.active{display:flex}
.section-title{font-weight:600;font-size:.78rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em}
label{font-size:.85rem;color:#374151;display:flex;flex-direction:column;gap:3px}
input[type=range]{width:100%}
.range-val{font-size:.78rem;color:#6b7280;text-align:right}
input[type=checkbox]{accent-color:#1e3a8a}
.check-row{flex-direction:row;align-items:center;gap:8px}
#sources-box{font-size:.8rem;color:#374151;line-height:1.6;background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;padding:10px;min-height:60px;white-space:pre-wrap}
.field{display:flex;flex-direction:column;gap:3px}
.field label{font-size:.82rem;color:#6b7280;font-weight:500}
.field input{padding:7px 10px;border:1px solid #d1d5db;border-radius:6px;font-size:.85rem;width:100%}
.ingest-btn{width:100%;padding:9px;border-radius:7px;font-size:.88rem;font-weight:500;cursor:pointer;border:none;color:#fff;margin-top:4px}
.ingest-btn.local{background:#059669}
.ingest-btn.confluence{background:#0052cc}
.ingest-btn.sharepoint{background:#0078d4}
.ingest-btn:disabled{opacity:.5;cursor:not-allowed}
.ingest-status{font-size:.8rem;min-height:18px;margin-top:2px}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-thumb{background:#d1d5db;border-radius:3px}
</style>
</head>
<body>

<!-- Login overlay -->
<div id="login-overlay" style="display:none">
  <div id="login-box">
    <h1>📚 RAG Pipeline</h1>
    <p>Sign in to access the knowledge base</p>
    <a class="login-btn google" id="google-login-btn" href="#">
      <svg width="18" height="18" viewBox="0 0 48 48"><path fill="#4285F4" d="M44.5 20H24v8.5h11.8C34.7 33.9 30.1 37 24 37c-7.2 0-13-5.8-13-13s5.8-13 13-13c3.1 0 5.9 1.1 8.1 2.9l6.4-6.4C34.6 4.1 29.6 2 24 2 11.8 2 2 11.8 2 24s9.8 22 22 22c11 0 21-8 21-22 0-1.3-.2-2.7-.5-4z"/></svg>
      Continue with Google
    </a>
    <a class="login-btn microsoft" id="ms-login-btn" href="#">
      <svg width="18" height="18" viewBox="0 0 21 21"><rect x="1" y="1" width="9" height="9" fill="#f25022"/><rect x="11" y="1" width="9" height="9" fill="#7fba00"/><rect x="1" y="11" width="9" height="9" fill="#00a4ef"/><rect x="11" y="11" width="9" height="9" fill="#ffb900"/></svg>
      Continue with Microsoft
    </a>
    <div style="margin-top:16px;font-size:.78rem;color:#9ca3af">
      Or <a href="#" onclick="skipLogin()" style="color:#1e3a8a">continue without login</a>
    </div>
  </div>
</div>

<header>
  <span class="title">📚 RAG Pipeline</span>
  <div id="user-info" style="display:none">
    <img id="user-avatar" src="" alt=""/>
    <span id="user-name"></span>
    <button id="logout-btn" onclick="logout()">Logout</button>
  </div>
  <span id="status-badge" onclick="checkStatus()">checking…</span>
</header>

<main>
  <!-- Chat -->
  <section id="chat-col">
    <div id="messages"></div>
    <div id="input-row">
      <textarea id="question" placeholder="Ask a question about your documents…"></textarea>
      <button id="send-btn" onclick="sendQuery()">Send</button>
      <button id="clear-btn" onclick="clearChat()">Clear</button>
    </div>
  </section>

  <!-- Sidebar -->
  <aside id="sidebar">
    <div class="tab-bar">
      <div class="tab active" onclick="switchTab('search')">🔍 Search</div>
      <div class="tab" onclick="switchTab('local')">📁 Local</div>
      <div class="tab" onclick="switchTab('confluence')">🔷 Confluence</div>
      <div class="tab" onclick="switchTab('sharepoint')">🔵 SharePoint</div>
    </div>

    <!-- Search settings -->
    <div id="tab-search" class="tab-panel active">
      <div class="section-title">Retrieval Settings</div>
      <label>Top-K documents
        <input type="range" id="top-k" min="1" max="10" value="5" oninput="document.getElementById('top-k-val').textContent=this.value">
        <span class="range-val" id="top-k-val">5</span>
      </label>
      <label>Min similarity score
        <input type="range" id="min-score" min="0" max="1" step="0.05" value="0.2" oninput="document.getElementById('min-score-val').textContent=parseFloat(this.value).toFixed(2)">
        <span class="range-val" id="min-score-val">0.20</span>
      </label>
      <label class="check-row"><input type="checkbox" id="show-sources" checked> Show sources</label>
      <label class="check-row"><input type="checkbox" id="summarize"> Summarize answer</label>
      <div class="section-title" style="margin-top:4px">Retrieved Sources</div>
      <div id="sources-box">Sources will appear here after a query.</div>
    </div>

    <!-- Local ingest -->
    <div id="tab-local" class="tab-panel">
      <div class="section-title">Ingest Local Documents</div>
      <p style="font-size:.82rem;color:#6b7280">Index files from a folder on the server (PDF, TXT, XLSX, DOCX, CSV).</p>
      <div class="field"><label>Data directory</label><input id="local-dir" value="data"/></div>
      <button class="ingest-btn local" onclick="runLocalIngest()">Ingest Local Files</button>
      <div class="ingest-status" id="local-status"></div>
    </div>

    <!-- Confluence ingest -->
    <div id="tab-confluence" class="tab-panel">
      <div class="section-title">Ingest from Confluence</div>
      <p style="font-size:.82rem;color:#6b7280">Loads all spaces from your Atlassian Confluence instance.</p>
      <div class="field"><label>Base URL</label><input id="cf-url" placeholder="https://yourproject.atlassian.net"/></div>
      <div class="field"><label>Wiki URL</label><input id="cf-url-ext" placeholder="https://yourproject.atlassian.net/wiki"/></div>
      <div class="field"><label>Username (email)</label><input id="cf-user" placeholder="you@company.com"/></div>
      <div class="field"><label>API Token</label><input id="cf-token" type="password" placeholder="Atlassian API token"/></div>
      <button class="ingest-btn confluence" onclick="runConfluenceIngest()">Ingest Confluence</button>
      <div class="ingest-status" id="cf-status"></div>
    </div>

    <!-- SharePoint ingest -->
    <div id="tab-sharepoint" class="tab-panel">
      <div class="section-title">Ingest from SharePoint</div>
      <p style="font-size:.82rem;color:#6b7280">Connects via Microsoft Graph API. Requires an Azure AD app with <code>Sites.Read.All</code>.</p>
      <div class="field"><label>Tenant ID</label><input id="sp-tenant" placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"/></div>
      <div class="field"><label>Client ID</label><input id="sp-client" placeholder="Azure AD app client ID"/></div>
      <div class="field"><label>Client Secret</label><input id="sp-secret" type="password" placeholder="Azure AD app secret"/></div>
      <div class="field"><label>Site ID</label><input id="sp-site" placeholder="SharePoint site ID"/></div>
      <button class="ingest-btn sharepoint" onclick="runSharePointIngest()">Ingest SharePoint</button>
      <div class="ingest-status" id="sp-status"></div>
    </div>
  </aside>
</main>

<script>
const API = '/api';

// ── Auth ──────────────────────────────────────────────────────────────────
async function checkAuth() {
  try {
    const r = await fetch(`${API}/auth/me`, {credentials:'include'});
    const d = await r.json();
    if (d.authenticated) {
      document.getElementById('login-overlay').style.display = 'none';
      const ui = document.getElementById('user-info');
      ui.style.display = 'flex';
      document.getElementById('user-name').textContent = d.name || d.email;
      const av = document.getElementById('user-avatar');
      if (d.picture) { av.src = d.picture; av.style.display='block'; }
      else av.style.display='none';
    } else {
      document.getElementById('login-overlay').style.display = 'flex';
    }
  } catch(e) {
    // API unreachable — allow access anyway
    document.getElementById('login-overlay').style.display = 'none';
  }
}

function skipLogin() {
  document.getElementById('login-overlay').style.display = 'none';
}

// ── Status ────────────────────────────────────────────────────────────────
async function checkStatus() {
  const el = document.getElementById('status-badge');
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    el.textContent = `🟢 ${d.vector_store_chunks} chunks`;
    el.className = 'ok';
  } catch(e) {
    el.textContent = '🔴 API unreachable';
    el.className = 'err';
  }
}

// ── Chat ──────────────────────────────────────────────────────────────────
function appendMsg(role, text) {
  const box = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

async function sendQuery() {
  const q = document.getElementById('question').value.trim();
  if (!q) return;
  document.getElementById('question').value = '';
  appendMsg('user', q);
  const thinking = appendMsg('bot thinking', '…');
  try {
    const res = await fetch(`${API}/query`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        question: q,
        top_k: parseInt(document.getElementById('top-k').value),
        min_score: parseFloat(document.getElementById('min-score').value),
        summarize: document.getElementById('summarize').checked
      })
    });
    const data = await res.json();
    thinking.className = 'msg bot';
    thinking.textContent = data.answer;
    if (document.getElementById('show-sources').checked && data.sources?.length) {
      document.getElementById('sources-box').textContent = data.sources.map((s,i) =>
        `[${i+1}] ${s.source} — page ${s.page} — score ${(s.score*100).toFixed(1)}%\n    ${s.preview}`
      ).join('\n\n');
    } else {
      document.getElementById('sources-box').textContent = 'No sources shown.';
    }
    checkStatus();
  } catch(e) {
    thinking.className = 'msg bot';
    thinking.textContent = `Error: ${e.message}`;
  }
}

function clearChat() {
  document.getElementById('messages').innerHTML = '';
  document.getElementById('sources-box').textContent = 'Sources will appear here after a query.';
}

// ── Tabs ──────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    const names = ['search','local','confluence','sharepoint'];
    t.classList.toggle('active', names[i] === name);
  });
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.classList.toggle('active', p.id === `tab-${name}`);
  });
}

// ── Ingest helpers ────────────────────────────────────────────────────────
async function _ingest(url, body, statusId, btnEl) {
  const statusEl = document.getElementById(statusId);
  btnEl.disabled = true;
  statusEl.textContent = 'Ingesting…';
  statusEl.style.color = '#6b7280';
  try {
    const res = await fetch(url, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || res.statusText);
    statusEl.textContent = d.status === 'ok'
      ? `✅ ${d.documents_loaded} docs → ${d.chunks} chunks`
      : `⚠️ ${d.status}`;
    statusEl.style.color = '#059669';
    checkStatus();
  } catch(e) {
    statusEl.textContent = `❌ ${e.message}`;
    statusEl.style.color = '#dc2626';
  }
  btnEl.disabled = false;
}

function runLocalIngest() {
  _ingest(`${API}/ingest`,
    {data_dir: document.getElementById('local-dir').value || 'data'},
    'local-status', event.target);
}

function runConfluenceIngest() {
  _ingest(`${API}/ingest/confluence`, {
    url:          document.getElementById('cf-url').value,
    url_extended: document.getElementById('cf-url-ext').value,
    username:     document.getElementById('cf-user').value,
    token:        document.getElementById('cf-token').value,
  }, 'cf-status', event.target);
}

function runSharePointIngest() {
  _ingest(`${API}/ingest/sharepoint`, {
    tenant_id:     document.getElementById('sp-tenant').value,
    client_id:     document.getElementById('sp-client').value,
    client_secret: document.getElementById('sp-secret').value,
    site_id:       document.getElementById('sp-site').value,
  }, 'sp-status', event.target);
}

// ── Keyboard ──────────────────────────────────────────────────────────────
document.getElementById('question').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
});

// ── Init ──────────────────────────────────────────────────────────────────
// Set login/logout URLs dynamically based on current host
document.getElementById('google-login-btn').href = `${API}/auth/google/login`;
document.getElementById('ms-login-btn').href = `${API}/auth/microsoft/login`;

function logout() {
  window.location.href = `${API}/auth/logout`;
}

checkAuth();
checkStatus();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
