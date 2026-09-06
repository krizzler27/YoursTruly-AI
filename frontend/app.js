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

let selectedModel = null;
let isStreaming = false;
let currentConversationId = null;
let conversations = [];
let openHistoryMenuId = null;
let renamingId = null;
let pendingHistoryDeleteId = null;

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

function showEmpty(show) {
  empty.style.display = show ? 'block' : 'none';
  trace.style.display = show ? 'block' : 'none';
  if (centerStatus) centerStatus.style.display = show ? 'flex' : 'none';
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
  body.append(roleEl, card);
  wrap.append(avatar, body);
  thread.appendChild(wrap);
  if (role === 'user') {
    card.textContent = content;
  } else {
    card.innerHTML = content ? mdSimple(content) : '';
    if (opts.streaming) {
      const cur = document.createElement('span');
      cur.className = 'cursor';
      card.appendChild(cur);
    }
  }
  thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
  return { wrap, card };
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
      hint.style.display = '';
      hint.removeAttribute('hidden');
      setModelPillDisabled(!healthy);
      if (!healthy) { const f2 = parseModel(selectedModel.name || selectedModel); modelNameEl.textContent = f2.base; }
      return healthy;
    }
    setStatus(healthy, healthy ? 'no models — add GGUF to ' + (data.path || '~/.yourstrulyai/models') : 'MODEL OFFLINE');
    hint.textContent = 'Enter to send \u2022 Shift+Enter for newline';
    hint.style.display = '';
    hint.removeAttribute('hidden');
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
      clearThread();
      showEmpty(true);
      ttftEl.textContent = '';
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
      input.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); saveHistoryRename(c.id); }
        if (e.key === 'Escape') { e.preventDefault(); cancelHistoryRename(); }
      });
      input.addEventListener('blur', () => {
        setTimeout(() => {
          if (document.activeElement && document.activeElement.closest && document.activeElement.closest('.history-rename-actions')) return;
          if (renamingId === c.id) cancelHistoryRename();
        }, 150);
      });
      const actions = document.createElement('div');
      actions.className = 'history-rename-actions';
      const saveBtn = document.createElement('button');
      saveBtn.type = 'button';
      saveBtn.className = 'save';
      saveBtn.textContent = '✓';
      saveBtn.title = 'Save';
      saveBtn.addEventListener('click', e => { e.stopPropagation(); saveHistoryRename(c.id); });
      const cancelBtn = document.createElement('button');
      cancelBtn.type = 'button';
      cancelBtn.textContent = '×';
      cancelBtn.title = 'Cancel';
      cancelBtn.addEventListener('click', e => { e.stopPropagation(); cancelHistoryRename(); });
      actions.append(saveBtn, cancelBtn);
      wrap.append(input, actions);
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
  if (!query || isStreaming) return;
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
  addMessage('user', query);
  input.value = '';
  autoResize();

  const { card } = addMessage('assistant', '', { streaming: true });
  let acc = '';
  let start = performance.now();
  let ttftDone = false;

  try {
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
    if (!resp.ok) {
      const t = await resp.text();
      // keep conversation id even on 500 so next turn stays in same thread
      if (newId) {
        await loadConversations();
        renderHistory();
      }
      throw new Error(`HTTP ${resp.status}: ${t}`);
    }
    const ttftHeader = resp.headers.get('X-TTFT');
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
        const line = chunk.trim();
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') {
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
    updateAssistantCard(card, acc || 'No response.', true);
    await loadConversations();
  } catch (e) {
    const m = e && e.message || '';
    const isBusy = m.includes('429') || m.includes('System Busy');
    const msg = isBusy ? '**System Busy — model is generating.** Please wait and try again.' : '**Could not reach backend / model not loaded.** Check ~/.yourstrulyai/models and backend logs.';
    updateAssistantCard(card, msg, true);
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    input.focus();
    await loadConversations();
    renderHistory();
  }
}

// History menu + rename + delete — global handlers
document.addEventListener('click', (e) => {
  if (!e.target.closest('.history-item')) closeHistoryMenu();
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
$('#newChat').addEventListener('click', async () => {
  currentConversationId = null;
  clearThread();
  showEmpty(true);
  ttftEl.textContent = '';
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
  const h = await fetchModels();
  lastHealthy = h;
  autoResize();
  if (!currentConversationId) showEmpty(true);
  setInterval(pollHealth, 10000);
})();
