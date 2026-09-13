const API_BASE = 'http://127.0.0.1:8000';
const $ = (s, r=document) => r.querySelector(s);

const thread = $('#thread');
const empty = $('#empty');
const composer = $('#composer');
const input = $('#prompt');
const sendBtn = $('#send');
const hint = $('#hint');
const trace = $('#trace');
const tracePath = $('#tracePath');
const modelPill = $('#modelPill');
const modelNameEl = $('#modelName');
const modelMenu = $('#modelMenu');
const historyList = $('#historyList');
const ttftEl = $('#ttft');
const statusDot = $('#statusDot');
const statusText = $('#statusText');
const centerStatusDot = $('#centerStatusDot');
const centerStatusText = $('#centerStatusText');
const centerStatus = $('#centerStatus');
const attachBtn = $('#attachBtn');
const fileInput = $('#fileInput');
const docTray = $('#docTray');
const routePill = $('#routePill');
const toastStack = $('#toastStack');
const filesSidebar = $('#filesSidebar');
const filesList = $('#filesList');
const filesCount = $('#filesCount');
const filesToggle = $('#filesToggle');

const ACCEPT_EXTS = ['.txt', '.md', '.pdf'];
const MAX_FILE_MB = 25;
const DOC_POLL_MS = 3000;
const DOC_POLL_MAX_MS = 5 * 60 * 1000;
const GATE_HINT = 'Indexing docs… chat paused — ask after Indexed';

let selectedModel = null;
let isStreaming = false;
let currentConversationId = null;
let conversations = [];
let openHistoryMenuId = null;
let renamingId = null;
let pendingHistoryDeleteId = null;
let stagedFiles = new Map();
let serverDocs = [];
let uploadingFiles = new Map();
let docPollTimer = null;
let docPollStartedAt = 0;
let prevIndexedNames = new Set();
let composerBlocked = false;
let filesEverIndexed = false;

const historyDeleteModal = $('#historyDeleteModal');
const historyDeleteCancel = $('#historyDeleteCancel');
const historyDeleteConfirm = $('#historyDeleteConfirm');

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

function mdSimple(src) {
  const fences = [];
  let html = src.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
    const idx = fences.length;
    fences.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    return `\x00FENCE${idx}\x00`;
  });
  html = html.replace(/`([^`]+)`/g, (_, c) => `<code>${escapeHtml(c)}</code>`);
  html = escapeHtml(html);
  html = html.replace(/\x00FENCE(\d+)\x00/g, (_, i) => fences[Number(i)]);
  html = html.replace(/\*\*([^\n*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__([^\n_]+)__/g, '<strong>$1</strong>');
  html = html.replace(/\*([^\n*]+)\*/g, '<em>$1</em>');
  const lines = html.split('\n');
  let out = '';
  let inList = false;
  let listType = null;
  function closeList(){ if(inList){ out += `</${listType}>`; inList=false; listType=null; } }
  for (let line of lines) {
    if (line.includes('<pre>')) { closeList(); out += line; continue; }
    if (/^\s*$/.test(line)) { closeList(); continue; }
    const ulMatch = line.match(/^\s*[-*]\s+(.*)$/);
    const olMatch = line.match(/^\s*\d+\.\s+(.*)$/);
    if (ulMatch) {
      if (!inList || listType !== 'ul') { closeList(); out += '<ul>'; inList=true; listType='ul'; }
      out += `<li>${ulMatch[1]}</li>`;
    } else if (olMatch) {
      if (!inList || listType !== 'ol') { closeList(); out += '<ol>'; inList=true; listType='ol'; }
      out += `<li>${olMatch[1]}</li>`;
    } else {
      closeList();
      if (line.startsWith('<pre>')) out += line;
      else out += `<p>${line}</p>`;
    }
  }
  closeList();
  return out;
}

function setStatus(online, text) {
  statusDot.className = 'status-dot' + (online ? '' : ' off');
  statusText.textContent = text;
  if (centerStatusDot) centerStatusDot.className = 'status-dot' + (online ? '' : ' off');
  if (centerStatusText) centerStatusText.textContent = text;
  if (centerStatus) centerStatus.style.display = empty.style.display !== 'none' ? 'flex' : 'none';
}
function setModelPillDisabled(disabled) {
  modelPill.disabled = disabled;
  modelPill.setAttribute('aria-disabled', disabled ? 'true' : 'false');
  modelPill.style.opacity = disabled ? '0.45' : '';
  modelPill.style.pointerEvents = disabled ? 'none' : '';
  if (disabled) {
    modelPill.removeAttribute('popovertarget');
    modelMenu.style.display = 'none';
    try { modelMenu.hidePopover(); } catch {}
  } else {
    modelPill.setAttribute('popovertarget', 'modelMenu');
    modelMenu.style.display = '';
  }
}

function autoResize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
}

function updateHintVisibility() {
  const hasMessages = !!thread.querySelector('.msg');
  if (!hint) return;
  hint.style.display = hasMessages ? 'none' : '';
  if (hasMessages) hint.setAttribute('hidden', '');
  else hint.removeAttribute('hidden');
}

function showEmpty(show) {
  empty.style.display = show ? 'block' : 'none';
  trace.style.display = show ? 'block' : 'none';
  if (centerStatus) centerStatus.style.display = show ? 'flex' : 'none';
  updateHintVisibility();
}

function shortFileName(name, maxLen = 22) {
  const t = (name || '').trim();
  return t.length > maxLen ? t.slice(0, maxLen) + '…' : t;
}

function validUpload(file) {
  const name = file.name || '';
  const dot = name.lastIndexOf('.');
  const ext = dot >= 0 ? name.slice(dot).toLowerCase() : '';
  if (!ACCEPT_EXTS.includes(ext)) return { ok: false, reason: `Only ${ACCEPT_EXTS.join(', ')} supported` };
  if (file.size > MAX_FILE_MB * 1024 * 1024) return { ok: false, reason: `Exceeds ${MAX_FILE_MB}MB` };
  if (!name.trim()) return { ok: false, reason: 'Filename missing' };
  return { ok: true };
}

function stageLabel(stage) {
  if (!stage) return '';
  if (stage === 'deciding') return 'Deciding…';
  if (stage === 'searching') return 'Searching docs…';
  if (stage === 'uploading') return 'Uploading…';
  if (stage === 'indexing') return 'Indexing docs…';
  if (stage.startsWith('reading')) return `Reading ${stage.replace('reading', '').trim() || 'docs'}…`;
  if (stage === 'answering') return 'Answering…';
  return stage;
}

function setRoutePill(route) {
  if (!routePill) return;
  if (!route) { routePill.hidden = true; routePill.textContent = ''; return; }
  const r = String(route).toUpperCase();
  routePill.hidden = false;
  routePill.textContent = r === 'RAG' ? 'RAG · docs' : r;
  routePill.classList.toggle('rag', r === 'RAG');
}

function toast(msg, type='info', ttl) {
  if (!toastStack) return;
  const el = document.createElement('div');
  el.className = `toast-item ${type}`;
  const iconMap = { info: 'i', success: '✓', warning: '!', error: '×' };
  const icon = document.createElement('span');
  icon.className = 'toast-icon';
  icon.textContent = iconMap[type] || 'i';
  const text = document.createElement('span');
  text.textContent = msg;
  text.style.flex = '1';
  text.style.minWidth = '0';
  const close = document.createElement('button');
  close.className = 'toast-close';
  close.type = 'button';
  close.textContent = '×';
  close.addEventListener('click', () => dismiss());
  el.append(icon, text, close);
  toastStack.appendChild(el);
  const duration = ttl ?? (type === 'error' ? 6000 : type === 'warning' ? 5000 : 4000);
  let t = setTimeout(dismiss, duration);
  function dismiss() {
    clearTimeout(t);
    el.style.animation = 'toastOut 140ms ease forwards';
    setTimeout(() => el.remove(), 150);
  }
  el.addEventListener('mouseenter', () => clearTimeout(t));
  el.addEventListener('mouseleave', () => t = setTimeout(dismiss, 1200));
}

function isIndexing() {
  if (uploadingFiles.size > 0) return true;
  return serverDocs.some(d => d.status === 'pending' || d.status === 'indexing');
}

function setComposerBlocked(blocked, reason='') {
  composerBlocked = blocked;
  if (composer) {
    if (blocked) composer.setAttribute('data-disabled', '');
    else composer.removeAttribute('data-disabled');
  }
  if (input) input.disabled = blocked;
  if (sendBtn) sendBtn.disabled = blocked || isStreaming;
  if (attachBtn) attachBtn.disabled = blocked;
  if (!hint) return;
  if (blocked) {
    hint.textContent = reason || GATE_HINT;
    hint.style.display = '';
    hint.removeAttribute('hidden');
  } else {
    hint.textContent = 'Enter to send • Shift+Enter for newline';
    updateHintVisibility();
  }
}

function hasIndexedDocs() {
  return serverDocs.some(d => (d.status || 'indexed') === 'indexed');
}

function renderFilesSidebar() {
  const indexed = serverDocs.filter(d => (d.status || 'indexed') === 'indexed');
  if (filesCount) filesCount.textContent = String(indexed.length);
  if (filesList) {
    filesList.innerHTML = '';
    if (!indexed.length) {
      filesList.innerHTML = `<div class="empty-hint">No files yet.</div>`;
    } else {
      indexed.forEach(doc => {
        const div = document.createElement('div');
        div.className = 'files-item';
        const name = document.createElement('span');
        name.className = 'files-name';
        name.textContent = doc.filename;
        name.title = (doc.summary || '').trim() || doc.filename;
        const x = document.createElement('button');
        x.type = 'button';
        x.className = 'doc-x';
        x.setAttribute('aria-label', `Remove ${doc.filename}`);
        x.textContent = '×';
        x.addEventListener('click', () => deleteServerDoc(doc.id));
        div.append(name, x);
        filesList.appendChild(div);
      });
    }
  }
  const show = indexed.length > 0;
  if (filesSidebar) filesSidebar.hidden = !show;
  if (filesToggle) filesToggle.hidden = !show;
  if (show && !filesEverIndexed) {
    filesEverIndexed = true;
    document.body.setAttribute('data-files-open', '');
  }
  if (!show) document.body.removeAttribute('data-files-open');
}

function clearThread() {
  thread.querySelectorAll('.msg').forEach(el => el.remove());
}

function addMessage(role, content, opts={}) {
  showEmpty(false);
  const wrap = document.createElement('div');
  wrap.className = 'msg';
  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar ' + role;
  avatar.textContent = role === 'user' ? 'You' : 'AI';
  const body = document.createElement('div');
  body.className = 'msg-body';
  const roleEl = document.createElement('div');
  roleEl.className = 'msg-role mono';
  roleEl.textContent = role === 'user' ? 'You' : 'Assistant';
  const card = document.createElement('div');
  card.className = 'msg-card ' + role;
  let statusEl = null;
  if (role === 'assistant' && opts.streaming) {
    statusEl = document.createElement('div');
    statusEl.className = 'rag-status';
    statusEl.hidden = true;
    body.append(roleEl, statusEl, card);
  } else {
    body.append(roleEl, card);
  }
  wrap.append(avatar, body);
  thread.appendChild(wrap);
  if (role === 'user') {
    if (opts.files && opts.files.length) {
      const filesEl = document.createElement('div');
      filesEl.className = 'msg-files';
      opts.files.forEach(name => {
        const chip = document.createElement('span');
        chip.className = 'doc-chip staged';
        chip.title = name;
        chip.innerHTML = `<span class="doc-name"></span>`;
        chip.querySelector('.doc-name').textContent = shortFileName(name);
        filesEl.appendChild(chip);
      });
      card.appendChild(filesEl);
    }
    const textEl = document.createElement('div');
    textEl.textContent = content;
    card.appendChild(textEl);
  } else {
    card.innerHTML = content ? mdSimple(content) : '';
    if (opts.streaming) {
      const cur = document.createElement('span');
      cur.className = 'cursor';
      card.appendChild(cur);
    }
  }
  thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
  updateHintVisibility();
  return { wrap, card, body, statusEl };
}

function setRagStatus(statusEl, stage) {
  if (!statusEl) return;
  if (!stage) { statusEl.hidden = true; statusEl.textContent = ''; return; }
  statusEl.hidden = false;
  statusEl.innerHTML = '';
  const spin = document.createElement('span');
  spin.className = 'doc-spin';
  const label = document.createElement('span');
  label.textContent = stageLabel(stage);
  statusEl.append(spin, label);
  thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
}

function updateAssistantCard(card, content, done=false) {
  card.innerHTML = mdSimple(content);
  if (!done) {
    const cur = document.createElement('span');
    cur.className = 'cursor';
    card.appendChild(cur);
  }
  thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
}

function renderTray() {
  if (!docTray) return;
  docTray.innerHTML = '';
  const hasStaged = stagedFiles.size > 0;
  const hasUploading = uploadingFiles.size > 0;
  const hasServer = serverDocs.length > 0;
  if (!hasStaged && !hasUploading && !hasServer) {
    docTray.hidden = true;
    return;
  }
  docTray.hidden = false;

  uploadingFiles.forEach((info, name) => {
    const chip = document.createElement('span');
    const isUpdate = serverDocs.some(d => d.filename === name && d.status === 'indexed');
    chip.className = 'doc-chip ' + (isUpdate ? 'updating' : 'indexing');
    chip.title = isUpdate ? `Updating ${name}…` : `${name} — indexing…`;
    const spin = document.createElement('span');
    spin.className = 'doc-spin';
    const label = document.createElement('span');
    label.className = 'doc-name';
    label.textContent = isUpdate ? `Updating ${shortFileName(name)}…` : `${shortFileName(name)}…`;
    chip.append(spin, label);
    if (info.queueDepth > 0) {
      const q = document.createElement('span');
      q.className = 'mono';
      q.style.fontSize = '10px';
      q.textContent = `+${info.queueDepth}`;
      chip.appendChild(q);
    }
    docTray.appendChild(chip);
  });

  stagedFiles.forEach((file, name) => {
    const chip = document.createElement('span');
    chip.className = 'doc-chip staged';
    chip.title = name;
    const label = document.createElement('span');
    label.className = 'doc-name';
    label.textContent = shortFileName(name);
    const x = document.createElement('button');
    x.type = 'button';
    x.className = 'doc-x';
    x.setAttribute('aria-label', `Remove ${name}`);
    x.textContent = '×';
    x.addEventListener('click', () => { stagedFiles.delete(name); renderTray(); });
    chip.append(label, x);
    docTray.appendChild(chip);
  });

  serverDocs.forEach(doc => {
    if (uploadingFiles.has(doc.filename) && doc.status === 'indexed') return;
    const chip = document.createElement('span');
    chip.className = 'doc-chip ' + (doc.status || 'indexed');
    chip.title = (doc.summary || '').trim() || doc.filename;
    if (doc.status === 'pending' || doc.status === 'indexing') {
      const spin = document.createElement('span');
      spin.className = 'doc-spin';
      const label = document.createElement('span');
      label.className = 'doc-name';
      label.textContent = `${shortFileName(doc.filename)}…`;
      chip.append(spin, label);
    } else if (doc.status === 'failed') {
      const label = document.createElement('span');
      label.className = 'doc-name';
      label.textContent = shortFileName(doc.filename);
      const retry = document.createElement('button');
      retry.type = 'button';
      retry.className = 'doc-retry';
      retry.setAttribute('aria-label', `Retry ${doc.filename}`);
      retry.title = 'Retry upload';
      retry.textContent = '↻';
      retry.addEventListener('click', () => retryFailedDoc(doc));
      const x = document.createElement('button');
      x.type = 'button';
      x.className = 'doc-x';
      x.setAttribute('aria-label', `Remove ${doc.filename}`);
      x.textContent = '×';
      x.addEventListener('click', () => deleteServerDoc(doc.id));
      chip.append(label, retry, x);
    } else {
      const label = document.createElement('span');
      label.className = 'doc-name';
      label.textContent = shortFileName(doc.filename);
      const x = document.createElement('button');
      x.type = 'button';
      x.className = 'doc-x';
      x.setAttribute('aria-label', `Remove ${doc.filename}`);
      x.textContent = '×';
      x.addEventListener('click', () => deleteServerDoc(doc.id));
      chip.append(label, x);
    }
    docTray.appendChild(chip);
  });
  const blocked = isIndexing();
  setComposerBlocked(blocked, blocked ? GATE_HINT : '');
  renderFilesSidebar();
}

async function ensureConversationId() {
  if (currentConversationId) return currentConversationId;
  const r = await fetch(`${API_BASE}/api/conversations`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({})
  });
  if (!r.ok) throw new Error(`Could not create conversation: HTTP ${r.status}`);
  const conv = await r.json();
  currentConversationId = conv.id;
  prevIndexedNames = new Set();
  await loadConversations();
  renderHistory();
  return currentConversationId;
}

async function loadServerDocs(conversationId) {
  if (!conversationId) {
    serverDocs = [];
    prevIndexedNames = new Set();
    renderTray();
    return;
  }
  try {
    const r = await fetch(`${API_BASE}/api/documents?conversation_id=${encodeURIComponent(conversationId)}`);
    if (!r.ok) throw new Error('failed');
    serverDocs = await r.json();
    prevIndexedNames = new Set(serverDocs.filter(d => (d.status || 'indexed') === 'indexed').map(d => d.filename));
  } catch {
    serverDocs = [];
    prevIndexedNames = new Set();
  }
  renderTray();
}

function startDocPoll() {
  stopDocPoll();
  docPollStartedAt = Date.now();
  docPollTimer = setInterval(pollDocs, DOC_POLL_MS);
}

function stopDocPoll() {
  if (docPollTimer) clearInterval(docPollTimer);
  docPollTimer = null;
}

async function pollDocs() {
  if (!currentConversationId) { stopDocPoll(); return; }
  if (Date.now() - docPollStartedAt > DOC_POLL_MAX_MS) {
    uploadingFiles.clear();
    stopDocPoll();
    renderTray();
    toast('Indexing timed out after 5 min', 'warning');
    return;
  }
  try {
    const r = await fetch(`${API_BASE}/api/documents?conversation_id=${encodeURIComponent(currentConversationId)}`);
    if (!r.ok) return;
    const docs = await r.json();
    serverDocs = docs;
    uploadingFiles.forEach((info, name) => {
      const match = docs.find(d => d.filename === name);
      if (!match) return;
      if (match.id !== info.oldId) uploadingFiles.delete(name);
    });
    const currIndexed = new Set(docs.filter(d => (d.status || 'indexed') === 'indexed').map(d => d.filename));
    docs.filter(d => (d.status || 'indexed') === 'indexed' && !prevIndexedNames.has(d.filename)).forEach(d => {
      toast(`${d.filename} indexed`, 'success');
    });
    docs.filter(d => d.status === 'failed' && !prevIndexedNames.has(d.filename)).forEach(d => {
      toast(`${d.filename} failed — retry`, 'error');
    });
    prevIndexedNames = currIndexed;
    const stillIndexing = docs.some(d => d.status === 'pending' || d.status === 'indexing');
    const hasFailed = docs.some(d => d.status === 'failed');
    if (!uploadingFiles.size && !stillIndexing) stopDocPoll();
    renderTray();
    if (hasFailed && !uploadingFiles.size && !stillIndexing) setComposerBlocked(false);
  } catch {}
}

async function uploadStaged(conversationId) {
  if (!stagedFiles.size) return;
  const entries = [...stagedFiles.entries()];
  stagedFiles.clear();
  const byName = new Map(serverDocs.map(d => [d.filename, d.id]));
  for (const [name, file] of entries) {
    const oldId = byName.get(name) || null;
    uploadingFiles.set(name, { file, queueDepth: 0, startTime: Date.now(), oldId });
  }
  renderTray();
  startDocPoll();
  for (const [name, file] of entries) {
    try {
      const form = new FormData();
      form.append('file', file, file.name);
      form.append('conversation_id', conversationId);
      const r = await fetch(`${API_BASE}/api/ingest`, { method: 'POST', body: form });
      if (!r.ok) {
        const info = uploadingFiles.get(name);
        if (info) {
          uploadingFiles.delete(name);
          serverDocs = [{ id: `local-failed-${Date.now()}`, filename: name, status: 'failed', summary: `Upload failed: HTTP ${r.status}`, conversation_id: conversationId }, ...serverDocs];
          toast(`Upload failed: ${name} (HTTP ${r.status})`, 'error');
        }
        continue;
      }
      const data = await r.json().catch(() => ({}));
      const info = uploadingFiles.get(name);
      if (info && typeof data.queue_depth === 'number') {
        info.queueDepth = data.queue_depth;
        if (data.queue_depth > 0) toast(`+${data.queue_depth} ahead in queue`, 'info');
      }
      renderTray();
    } catch {
      uploadingFiles.delete(name);
      toast(`Upload failed: ${name}`, 'error');
    }
  }
  renderTray();
  pollDocs();
}

async function uploadFilesNow(files) {
  await ensureConversationId();
  files.forEach(f => stagedFiles.set(f.name, f));
  renderTray();
  await uploadStaged(currentConversationId);
}

async function deleteServerDoc(documentId) {
  try {
    const r = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const gone = serverDocs.find(d => String(d.id) === String(documentId));
    serverDocs = serverDocs.filter(d => String(d.id) !== String(documentId));
    prevIndexedNames = new Set(serverDocs.filter(d => (d.status || 'indexed') === 'indexed').map(d => d.filename));
    if (gone) toast(`Deleted ${gone.filename}`, 'success');
    renderTray();
  } catch (e) {
    console.error(e);
  }
}

async function retryFailedDoc(doc) {
  const cached = uploadingFiles.get(doc.filename);
  const file = cached ? cached.file : null;
  try {
    await fetch(`${API_BASE}/api/documents/${encodeURIComponent(doc.id)}`, { method: 'DELETE' });
  } catch {}
  serverDocs = serverDocs.filter(d => String(d.id) !== String(doc.id));
  if (file && currentConversationId) {
    stagedFiles.set(file.name, file);
    renderTray();
    uploadStaged(currentConversationId);
  } else {
    renderTray();
    if (fileInput) fileInput.click();
  }
}

async function checkModelHealth() {
  try {
    const r = await fetch(`${API_BASE}/api/health`, { headers: { 'accept': 'application/json' } });
    if (!r.ok) return false;
    const data = await r.json();
    // dynamic: idle = GGUF exists but not yet mmap-loaded — lazy loads on first chat, still healthy
    return (data.status === 'ready' || data.status === 'idle') && !!data.exists;
  } catch { return false; }
}

async function fetchModels(forcedHealthy = null) {
  let healthy = forcedHealthy;
  try {
    const r = await fetch(`${API_BASE}/api/models`, { headers: { 'accept': 'application/json' } });
    if (!r.ok) throw new Error('no models');
    const data = await r.json();
    const models = (data.models || []).map(m => typeof m === 'string' ? { name: m, path: m } : m);
    if (healthy === null) healthy = await checkModelHealth();
    if (models.length) {
      setStatus(healthy, healthy ? 'MODEL READY' : 'MODEL OFFLINE');
      renderModelMenu(models);
      if (!selectedModel) selectedModel = models[0];
      const first = parseModel(selectedModel.name || selectedModel);
      modelNameEl.textContent = first.base;
      const pq = document.getElementById('modelQuant');
      if (pq) { pq.textContent = first.quant; pq.hidden = !first.quant; }
      hint.textContent = 'Enter to send \u2022 Shift+Enter for newline';
      updateHintVisibility();
      setModelPillDisabled(!healthy);
      if (!healthy) { const f2 = parseModel(selectedModel.name || selectedModel); modelNameEl.textContent = f2.base; }
      return healthy;
    }
    setStatus(healthy, healthy ? 'no models — add GGUF to ' + (data.path || '~/.yourstrulyai/models') : 'MODEL OFFLINE');
    hint.textContent = 'Enter to send \u2022 Shift+Enter for newline';
    updateHintVisibility();
    setModelPillDisabled(true);
    return healthy;
  } catch (e) {
    setStatus(false, 'MODEL OFFLINE');
    hint.textContent = '';
    hint.style.display = 'none';
    hint.setAttribute('hidden', '');
    setModelPillDisabled(true);
    modelNameEl.textContent = 'offline';
    modelMenu.innerHTML = `<div class="mono" style="padding:8px 10px; color:var(--muted-foreground)">Model offline — check ~/.yourstrulyai/models</div>`;
    return false;
  }
}

function parseModel(raw) {
  if (!raw) return { base: '', quant: '' };
  const baseRaw = raw.replace(/\.gguf$/i, '').trim();
  const m = baseRaw.match(/^(.*?)[-_](Q\d.*)$/i);
  if (m) return { base: m[1], quant: m[2] };
  return { base: baseRaw.replace(/_/g, ' '), quant: '' };
}
function displayName(raw) {
  const p = parseModel(raw);
  return p.quant ? `${p.base} (${p.quant})` : p.base;
}
function renderModelMenu(models) {
  modelMenu.innerHTML = '';
  models.forEach(m => {
    const name = m.name || m;
    const path = m.path || m;
    const isSame = selectedModel && (selectedModel.path === path || selectedModel === m);
    const div = document.createElement('button');
    div.setAttribute('role', 'menuitem');
    div.className = 'model-option ghost' + (isSame ? ' active' : '');
    const p = parseModel(name);
    div.innerHTML = `<span class="model-name" title="${escapeHtml(name)}">${escapeHtml(p.base)}</span>${p.quant ? `<small class="model-quant">${escapeHtml(p.quant)}</small>` : ''}`;
    div.addEventListener('click', () => {
      selectedModel = m;
      modelNameEl.textContent = p.base;
      const pillQuant = document.getElementById('modelQuant');
      if (pillQuant) pillQuant.textContent = p.quant;
      if (pillQuant) pillQuant.hidden = !p.quant;
      [...modelMenu.children].forEach(c => c.classList.remove('active'));
      div.classList.add('active');
      try { modelMenu.hidePopover(); } catch {}
    });
    modelMenu.appendChild(div);
  });
}

function truncateTitle(title, maxLen = 32) {
  const t = title.trim();
  return t.length > maxLen ? t.slice(0, maxLen) + '…' : t;
}

function closeHistoryMenu() {
  openHistoryMenuId = null;
  document.querySelectorAll('.history-item.menu-open').forEach(el => el.classList.remove('menu-open'));
  document.querySelectorAll('.history-menu').forEach(el => el.hidden = true);
}

function startHistoryRename(id) {
  if (isStreaming) return;
  renamingId = id;
  openHistoryMenuId = null;
  renderHistory();
  const input = historyList.querySelector(`[data-rename-input="${id}"]`);
  if (input) { input.focus(); input.select(); }
}

function cancelHistoryRename() {
  renamingId = null;
  renderHistory();
}

async function saveHistoryRename(id) {
  const input = historyList.querySelector(`[data-rename-input="${id}"]`);
  if (!input) return;
  const raw = input.value.trim();
  if (!raw) {
    input.style.borderColor = 'var(--danger)';
    input.focus();
    return;
  }
  if (raw.length > 250) {
    input.style.borderColor = 'var(--danger)';
    input.focus();
    return;
  }
  const prev = conversations.find(x => x.id === id);
  const prevTitle = prev ? prev.title : '';
  if (raw === prevTitle) { cancelHistoryRename(); return; }
  input.disabled = true;
  try {
    const r = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ title: raw })
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || `HTTP ${r.status}`);
    }
    const updated = await r.json();
    const idx = conversations.findIndex(x => x.id === id);
    if (idx !== -1) conversations[idx].title = updated.title;
    if (conversations[idx]) conversations[idx].updated_at = updated.updated_at;
    renamingId = null;
    await loadConversations();
  } catch (e) {
    input.disabled = false;
    input.style.borderColor = 'var(--danger)';
    input.focus();
    console.error(e);
  }
}

async function deleteHistoryConversation(id) {
  if (isStreaming) return;
  try {
    const r = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || `HTTP ${r.status}`);
    }
    conversations = conversations.filter(x => x.id !== id);
    if (currentConversationId === id) {
      currentConversationId = null;
      serverDocs = [];
      stagedFiles.clear();
      uploadingFiles.clear();
      prevIndexedNames = new Set();
      filesEverIndexed = false;
      stopDocPoll();
      renderTray();
      clearThread();
      showEmpty(true);
      ttftEl.textContent = '';
      setRoutePill(null);
      setComposerBlocked(false);
    }
    renderHistory();
    await loadConversations();
  } catch (e) {
    console.error(e);
  } finally {
    pendingHistoryDeleteId = null;
  }
}

function renderHistory() {
  if (!conversations.length) {
    historyList.innerHTML = `<div class="empty-hint">No conversations yet.</div>`;
    return;
  }
  historyList.innerHTML = '';
  conversations.forEach(c => {
    const isActive = c.id === currentConversationId;
    const isRenaming = renamingId === c.id;
    const isMenuOpen = openHistoryMenuId === c.id;
    const div = document.createElement('div');
    div.className = 'history-item' + (isActive ? ' active' : '') + (isMenuOpen ? ' menu-open' : '');
    div.dataset.id = c.id;

    if (isRenaming) {
      const wrap = document.createElement('div');
      wrap.className = 'history-rename-wrap';
      wrap.addEventListener('click', e => e.stopPropagation());
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'history-rename-input';
      input.value = c.title;
      input.maxLength = 250;
      input.setAttribute('data-rename-input', c.id);
      input.setAttribute('aria-label', 'Rename conversation');
      input.placeholder = 'Enter to save, Esc to cancel';
      input.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); saveHistoryRename(c.id); }
        if (e.key === 'Escape') { e.preventDefault(); cancelHistoryRename(); }
      });
      // blur alone does not auto-save; document click handler below will save on outside click
      wrap.appendChild(input);
      div.appendChild(wrap);
      div.addEventListener('click', e => e.stopPropagation());
    } else {
      const short = truncateTitle(c.title, 32);
      const titleEl = document.createElement('span');
      titleEl.className = 'history-title';
      titleEl.title = c.title;
      titleEl.textContent = short;
      // double-click to rename
      titleEl.addEventListener('dblclick', e => { e.stopPropagation(); if (!isStreaming) startHistoryRename(c.id); });
      div.appendChild(titleEl);

      const menuBtn = document.createElement('button');
      menuBtn.type = 'button';
      menuBtn.className = 'history-menu-btn';
      menuBtn.setAttribute('aria-label', 'Conversation menu');
      menuBtn.setAttribute('aria-haspopup', 'true');
      menuBtn.setAttribute('aria-expanded', isMenuOpen ? 'true' : 'false');
      menuBtn.textContent = '⋮';
      menuBtn.disabled = isStreaming;
      menuBtn.addEventListener('click', e => {
        e.stopPropagation();
        if (isStreaming) return;
        if (openHistoryMenuId === c.id) closeHistoryMenu();
        else { closeHistoryMenu(); openHistoryMenuId = c.id; renderHistory(); }
      });
      div.appendChild(menuBtn);

      const menu = document.createElement('div');
      menu.className = 'history-menu';
      menu.hidden = !isMenuOpen;
      menu.setAttribute('role', 'menu');
      const renameBtn = document.createElement('button');
      renameBtn.type = 'button';
      renameBtn.className = 'history-menu-item';
      renameBtn.setAttribute('role', 'menuitem');
      renameBtn.textContent = 'Rename';
      renameBtn.addEventListener('click', e => {
        e.stopPropagation();
        closeHistoryMenu();
        startHistoryRename(c.id);
      });
      const delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.className = 'history-menu-item danger';
      delBtn.setAttribute('role', 'menuitem');
      delBtn.textContent = 'Delete';
      delBtn.addEventListener('click', e => {
        e.stopPropagation();
        closeHistoryMenu();
        pendingHistoryDeleteId = c.id;
        if (historyDeleteModal && typeof historyDeleteModal.showModal === 'function') {
          try { historyDeleteModal.showModal(); } catch {}
        } else if (historyDeleteModal) historyDeleteModal.setAttribute('open','');
      });
      menu.append(renameBtn, delBtn);
      div.appendChild(menu);

      div.addEventListener('click', () => {
        if (isStreaming) return;
        if (renamingId) return;
        if (openHistoryMenuId) { closeHistoryMenu(); }
        if (currentConversationId === c.id) return;
        currentConversationId = c.id;
        stagedFiles.clear();
        uploadingFiles.clear();
        prevIndexedNames = new Set();
        filesEverIndexed = false;
        stopDocPoll();
        setRoutePill(null);
        setComposerBlocked(false);
        ttftEl.textContent = '';
        renderHistory();
        loadMessages(c.id);
      });
    }
    historyList.appendChild(div);
  });
  // dynamic placement: flip to open-up if not enough space below
  if (openHistoryMenuId) {
    const openItem = historyList.querySelector(`.history-item[data-id="${openHistoryMenuId}"]`);
    const openMenu = openItem ? openItem.querySelector('.history-menu') : null;
    if (openItem && openMenu && !openMenu.hidden) {
      const rect = openItem.getBoundingClientRect();
      const spaceBelow = window.innerHeight - rect.bottom;
      const need = 84;
      if (spaceBelow < need + 12) openMenu.classList.add('open-up');
      else openMenu.classList.remove('open-up');
    }
  }
}

async function loadConversations() {
  try {
    const r = await fetch(`${API_BASE}/api/conversations`);
    if (!r.ok) throw new Error('failed');
    const data = await r.json();
    conversations = Array.isArray(data) ? data : [];
    renderHistory();
  } catch {
    // keep empty hint, don't block
    if (!conversations.length) renderHistory();
  }
}

async function loadMessages(conversationId) {
  try {
    const r = await fetch(`${API_BASE}/api/conversations/${conversationId}/messages`);
    if (!r.ok) throw new Error('failed');
    const msgs = await r.json();
    clearThread();
    setRoutePill(null);
    ttftEl.textContent = '';
    loadServerDocs(conversationId);
    if (!msgs.length) {
      showEmpty(true);
      return;
    }
    showEmpty(false);
    msgs.forEach(m => addMessage(m.role, m.content));
    ttftEl.textContent = '';
  } catch {
    // fallback: show empty
    clearThread();
    showEmpty(true);
  }
}

async function send() {
  const query = input.value.trim();
  const hasStaged = stagedFiles.size > 0;
  if ((!query && !hasStaged) || isStreaming) return;
  if (composerBlocked || isIndexing()) {
    toast('Indexing docs… chat paused — ask after Indexed', 'warning');
    setComposerBlocked(true, GATE_HINT);
    return;
  }
  const isFirst = !currentConversationId && thread.querySelectorAll('.msg').length === 0;
  if (isFirst) {
    tracePath.classList.remove('animate');
    void tracePath.getBoundingClientRect();
    tracePath.classList.add('animate');
  }
  isStreaming = true;
  closeHistoryMenu();
  sendBtn.disabled = true;
  ttftEl.textContent = '';
  setRoutePill(null);

  if (!query && hasStaged) {
    try {
      await ensureConversationId();
      const cid = currentConversationId;
      input.value = '';
      autoResize();
      await uploadStaged(cid);
      await loadConversations();
      renderHistory();
    } catch (e) {
      console.error(e);
    } finally {
      isStreaming = false;
      sendBtn.disabled = false;
      input.focus();
    }
    return;
  }

  const attachedNames = [...stagedFiles.keys()];
  addMessage('user', query, { files: attachedNames });
  input.value = '';
  autoResize();

  const { card, statusEl } = addMessage('assistant', '', { streaming: true });
  let acc = '';
  let start = performance.now();
  let ttftDone = false;

  try {
    await ensureConversationId();
    if (attachedNames.length) await uploadStaged(currentConversationId);

    const modelPath = selectedModel ? (selectedModel.path || selectedModel) : undefined;
    const payload = { query, model: modelPath || undefined };
    if (currentConversationId) payload.conversation_id = currentConversationId;
    const resp = await fetch(`${API_BASE}/api/chat/stream`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const newId = resp.headers.get('X-Conversation-Id') || resp.headers.get('x-conversation-id');
    if (newId) currentConversationId = newId;
    const routeHeader = resp.headers.get('X-Route') || resp.headers.get('x-route');
    if (routeHeader) setRoutePill(routeHeader);
    if (!resp.ok) {
      const t = await resp.text();
      if (newId) {
        await loadConversations();
        renderHistory();
      }
      throw new Error(`HTTP ${resp.status}: ${t}`);
    }
    const ttftHeader = resp.headers.get('X-TTFT') || resp.headers.get('x-ttft');
    if (ttftHeader) ttftEl.textContent = `TTFT ${Number(ttftHeader).toFixed(0)} ms`;

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const chunk of parts) {
        const lines = chunk.split('\n').map(l => l.trim()).filter(Boolean);
        let event = null;
        let data = null;
        for (const ln of lines) {
          if (ln.startsWith('event:')) event = ln.slice(6).trim();
          else if (ln.startsWith('data:')) data = ln.slice(5).trim();
        }
        if (data === null || data === '') continue;
        if (event === 'stage') {
          try {
            const j = JSON.parse(data);
            if (j.stage) setRagStatus(statusEl, j.stage);
          } catch {}
          continue;
        }
        if (data === '[DONE]') {
          if (statusEl) { statusEl.hidden = true; statusEl.textContent = ''; }
          updateAssistantCard(card, acc, true);
          isStreaming = false;
          sendBtn.disabled = false;
          if (!ttftDone && !ttftHeader) {
            const ms = performance.now() - start;
            ttftEl.textContent = `TTFT ${ms.toFixed(0)} ms`;
          }
          await loadConversations();
          renderHistory();
          return;
        }
        try {
          const j = JSON.parse(data);
          if (j.stage && !j.content) { setRagStatus(statusEl, j.stage); continue; }
          const delta = j.content || '';
          if (delta) {
            if (!ttftDone) {
              ttftDone = true;
              const ms = performance.now() - start;
              if (!ttftHeader) ttftEl.textContent = `TTFT ${ms.toFixed(0)} ms`;
            }
            acc += delta;
            updateAssistantCard(card, acc, false);
          }
        } catch {}
      }
    }
    if (statusEl) { statusEl.hidden = true; statusEl.textContent = ''; }
    updateAssistantCard(card, acc || 'No response.', true);
    await loadConversations();
  } catch (e) {
    if (statusEl) { statusEl.hidden = true; statusEl.textContent = ''; }
    const m = e && e.message || '';
    const isBusy = m.includes('429') || m.includes('System Busy');
    const msg = isBusy ? '**System Busy — model is generating.** Please wait and try again.' : '**Could not reach backend / model not loaded.** Check ~/.yourstrulyai/models and backend logs.';
    updateAssistantCard(card, msg, true);
  } finally {
    isStreaming = false;
    if (isIndexing()) setComposerBlocked(true, GATE_HINT);
    else { setComposerBlocked(false); sendBtn.disabled = false; }
    input.focus();
    await loadConversations();
    renderHistory();
  }
}

// History menu + rename + delete — global handlers
document.addEventListener('click', (e) => {
  const insideItem = e.target.closest('.history-item');
  if (!insideItem) {
    closeHistoryMenu();
    if (renamingId) {
      const input = historyList.querySelector(`[data-rename-input="${renamingId}"]`);
      const raw = input ? input.value.trim() : '';
      const orig = conversations.find(x => x.id === renamingId);
      const origTitle = orig ? orig.title : '';
      if (raw && raw !== origTitle && raw.length <= 250) saveHistoryRename(renamingId);
      else cancelHistoryRename();
    }
    return;
  }
  // click inside another item while renaming — treat as outside save
  if (renamingId && insideItem.dataset.id !== renamingId) {
    const input = historyList.querySelector(`[data-rename-input="${renamingId}"]`);
    const raw = input ? input.value.trim() : '';
    const orig = conversations.find(x => x.id === renamingId);
    const origTitle = orig ? orig.title : '';
    if (raw && raw !== origTitle && raw.length <= 250) saveHistoryRename(renamingId);
    else cancelHistoryRename();
  }
});
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (renamingId) { e.preventDefault(); cancelHistoryRename(); }
  else if (openHistoryMenuId) { e.preventDefault(); closeHistoryMenu(); renderHistory(); }
});
if (historyDeleteCancel) historyDeleteCancel.addEventListener('click', () => {
  pendingHistoryDeleteId = null;
  if (historyDeleteModal && typeof historyDeleteModal.close === 'function') try { historyDeleteModal.close(); } catch {}
  if (historyDeleteModal) historyDeleteModal.removeAttribute('open');
});
if (historyDeleteConfirm) historyDeleteConfirm.addEventListener('click', async (e) => {
  e.preventDefault();
  const id = pendingHistoryDeleteId;
  if (!id) return;
  if (historyDeleteModal && typeof historyDeleteModal.close === 'function') try { historyDeleteModal.close(); } catch {}
  if (historyDeleteModal) historyDeleteModal.removeAttribute('open');
  await deleteHistoryConversation(id);
});
if (historyDeleteModal) {
  historyDeleteModal.addEventListener('click', (e) => {
    if (e.target === historyDeleteModal) {
      pendingHistoryDeleteId = null;
      try { historyDeleteModal.close(); } catch {}
      historyDeleteModal.removeAttribute('open');
    }
  });
  historyDeleteModal.addEventListener('cancel', (e) => {
    e.preventDefault();
    pendingHistoryDeleteId = null;
    try { historyDeleteModal.close(); } catch {}
  });
}
// Events
input.addEventListener('input', autoResize);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
composer.addEventListener('submit', (e) => { e.preventDefault(); send(); });
sendBtn.addEventListener('click', (e) => { e.preventDefault(); send(); });
if (filesToggle) {
  filesToggle.addEventListener('click', (e) => {
    e.preventDefault();
    if (!hasIndexedDocs()) return;
    if (document.body.hasAttribute('data-files-open')) document.body.removeAttribute('data-files-open');
    else document.body.setAttribute('data-files-open', '');
  });
}
if (attachBtn && fileInput) {
  attachBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', async () => {
    const files = [...(fileInput.files || [])];
    fileInput.value = '';
    if (!files.length) return;
    const valids = [];
    files.forEach(f => {
      const check = validUpload(f);
      if (!check.ok) toast(`${f.name}: ${check.reason}`, 'error');
      else valids.push(f);
    });
    if (!valids.length) { input.focus(); return; }
    try {
      await uploadFilesNow(valids);
    } catch (e) {
      console.error(e);
      toast('Upload failed — retry', 'error');
    }
    input.focus();
  });
}
$('#newChat').addEventListener('click', async () => {
  currentConversationId = null;
  serverDocs = [];
  stagedFiles.clear();
  uploadingFiles.clear();
  prevIndexedNames = new Set();
  filesEverIndexed = false;
  stopDocPoll();
  renderTray();
  clearThread();
  showEmpty(true);
  ttftEl.textContent = '';
  setRoutePill(null);
  setComposerBlocked(false);
  closeHistoryMenu();
  cancelHistoryRename();
  renderHistory();
  input.focus();
});

document.querySelectorAll('[data-starter]').forEach(btn => {
  btn.addEventListener('click', () => {
    input.value = btn.dataset.starter;
    autoResize();
    input.focus();
  });
});

if (!HTMLElement.prototype.hasOwnProperty('popover')) {
  modelPill.addEventListener('click', () => {
    modelMenu.style.display = modelMenu.style.display === 'block' ? 'none' : 'block';
    modelMenu.style.position = 'absolute';
  });
  document.addEventListener('click', (e) => {
    if (!modelPill.contains(e.target) && !modelMenu.contains(e.target)) modelMenu.style.display = 'none';
  });
}

let lastHealthy = null;

async function pollHealth() {
  if (document.hidden) return;
  const healthy = await checkModelHealth();
  if (healthy === lastHealthy) return;
  lastHealthy = healthy;
  await fetchModels(healthy);
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) pollHealth();
});
window.addEventListener('focus', pollHealth);

// Init — single health check via fetchModels (no duplicate)
(async () => {
  await loadConversations();
  renderHistory();
  renderTray();
  const h = await fetchModels();
  lastHealthy = h;
  autoResize();
  if (!currentConversationId) showEmpty(true);
  else loadServerDocs(currentConversationId);
  updateHintVisibility();
  setInterval(pollHealth, 10000);
})();
