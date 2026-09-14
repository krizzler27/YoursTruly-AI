import { API_BASE, ACCEPT_EXTS, MAX_FILE_MB, DOC_POLL_MS, DOC_POLL_MAX_MS, GATE_HINT } from './js/shared/config.js';
import { $, escapeHtml, mdSimple, shortFileName, truncateTitle, parseModel, displayName } from './js/shared/utils.js';
import { ICONS } from './js/shared/icons.js';
import { createTicker } from './js/shared/ticker.js';
import { createToast } from './js/shared/toast.js';
import { store } from './js/chat/store.js';
import * as chatApi from './js/chat/api.js';

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
const centerStatusDot = $('#centerStatusDot');
const centerStatusText = $('#centerStatusText');
const centerStatus = $('#centerStatus');
const attachBtn = $('#attachBtn');
const fileInput = $('#fileInput');
const docTray = $('#docTray');
const routePill = $('#routePill');
const toastStack = $('#toastStack');
const contextBar = $('#contextBar');
const contextFiles = $('#contextFiles');
const contextToggle = $('#contextToggle');

const ticker = createTicker(thread);
function setRagStatus(el, stage) { ticker.set(el, stage); }
function stopTicker(el) { ticker.stop(el); }

const historyDeleteModal = $('#historyDeleteModal');
const historyDeleteCancel = $('#historyDeleteCancel');
const historyDeleteConfirm = $('#historyDeleteConfirm');

const toast = createToast(toastStack);

function setHeroStatus(online, text) {
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

function validUpload(file) {
  const name = file.name || '';
  const dot = name.lastIndexOf('.');
  const ext = dot >= 0 ? name.slice(dot).toLowerCase() : '';
  if (!ACCEPT_EXTS.includes(ext)) return { ok: false, reason: `Only ${ACCEPT_EXTS.join(', ')} supported` };
  if (file.size > MAX_FILE_MB * 1024 * 1024) return { ok: false, reason: `Exceeds ${MAX_FILE_MB}MB` };
  if (!name.trim()) return { ok: false, reason: 'Filename missing' };
  return { ok: true };
}

function setRoutePill(route) {
  if (!routePill) return;
  if (!route) { routePill.hidden = true; routePill.textContent = ''; return; }
  const r = String(route).toUpperCase();
  routePill.hidden = false;
  routePill.textContent = r === 'RAG' ? 'RAG · docs' : r;
  routePill.classList.toggle('rag', r === 'RAG');
}

function isIndexing() {
  if (store.state.uploadingFiles.size > 0) return true;
  return store.state.serverDocs.some(d => d.status === 'pending' || d.status === 'indexing');
}

function setComposerBlocked(blocked, reason='') {
  store.set({ composerBlocked: blocked });
  if (composer) {
    if (blocked) composer.setAttribute('data-disabled', '');
    else composer.removeAttribute('data-disabled');
  }
  if (input) input.disabled = blocked;
  if (sendBtn) sendBtn.disabled = blocked || store.state.isStreaming;
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
  return store.state.serverDocs.some(d => (d.status || 'indexed') === 'indexed');
}

function renderContextBar() {
  const indexed = store.state.serverDocs.filter(d => (d.status || 'indexed') === 'indexed');
  const indexing = store.state.uploadingFiles.size > 0 || store.state.serverDocs.some(d => d.status === 'pending' || d.status === 'indexing');
  if (contextFiles) {
    contextFiles.innerHTML = '';
    indexed.forEach(doc => {
      const pill = document.createElement('span');
      pill.className = 'context-pill';
      pill.title = (doc.summary || '').trim() || doc.filename;
      const name = document.createElement('span');
      name.className = 'doc-name';
      name.textContent = shortFileName(doc.filename);
      const ok = document.createElement('span');
      ok.className = 'toast-icon';
      ok.innerHTML = ICONS.check;
      ok.style.width = '14px'; ok.style.height = '14px'; ok.style.flexBasis = '14px';
      pill.append(name, ok);
      contextFiles.appendChild(pill);
    });
    if (indexing) {
      const pill = document.createElement('span');
      pill.className = 'context-pill indexing';
      const spin = document.createElement('span');
      spin.className = 'doc-spin';
      const label = document.createElement('span');
      label.className = 'doc-name';
      label.textContent = 'Indexing…';
      pill.append(spin, label);
      contextFiles.appendChild(pill);
    }
  }
  const show = indexed.length > 0 || indexing;
  if (contextBar) {
    contextBar.hidden = !show;
    contextBar.classList.toggle('collapsed', store.state.contextCollapsed && indexed.length > 0);
    contextBar.classList.toggle('is-overflowing', !contextBar.hidden && contextBar.scrollWidth > contextBar.clientWidth + 1);
  }
  if (contextToggle) contextToggle.textContent = (store.state.contextCollapsed && indexed.length > 0) ? '+' : '–';
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
  }
  thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
  updateHintVisibility();
  return { wrap, card, body, statusEl };
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
  const hasStaged = store.state.stagedFiles.size > 0;
  const hasUploading = store.state.uploadingFiles.size > 0;
  const hasServer = store.state.serverDocs.length > 0;
  if (!hasStaged && !hasUploading && !hasServer) {
    docTray.hidden = true;
    return;
  }
  docTray.hidden = false;

  store.state.uploadingFiles.forEach((info, name) => {
    const chip = document.createElement('span');
    const isUpdate = store.state.serverDocs.some(d => d.filename === name && d.status === 'indexed');
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

  store.state.stagedFiles.forEach((file, name) => {
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
    x.innerHTML = ICONS.close;
    x.addEventListener('click', () => { store.state.stagedFiles.delete(name); renderTray(); });
    chip.append(label, x);
    docTray.appendChild(chip);
  });

  store.state.serverDocs.forEach(doc => {
    if (store.state.uploadingFiles.has(doc.filename) && doc.status === 'indexed') return;
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
      retry.innerHTML = ICONS.retry;
      retry.addEventListener('click', () => retryFailedDoc(doc));
      chip.append(label, retry);
    } else {
      const label = document.createElement('span');
      label.className = 'doc-name';
      label.textContent = shortFileName(doc.filename);
      chip.append(label);
    }
    docTray.appendChild(chip);
  });
  const blocked = isIndexing();
  setComposerBlocked(blocked, blocked ? GATE_HINT : '');
  renderContextBar();
}

async function ensureConversationId() {
  if (store.state.currentConversationId) return store.state.currentConversationId;
  const id = await chatApi.ensureConversationId(store.state.currentConversationId);
  store.set({ currentConversationId: id, prevIndexedNames: new Set() });
  await loadConversations();
  renderHistory();
  return store.state.currentConversationId;
}

async function loadServerDocs(conversationId) {
  if (!conversationId) {
    store.set({ serverDocs: [], prevIndexedNames: new Set() });
    renderTray();
    renderContextBar();
    return;
  }
  try {
    const docs = await chatApi.loadServerDocs(conversationId);
    store.set({
      serverDocs: docs,
      prevIndexedNames: new Set(docs.filter(d => (d.status || 'indexed') === 'indexed').map(d => d.filename))
    });
  } catch {
    store.set({ serverDocs: [], prevIndexedNames: new Set() });
  }
  renderTray();
  renderContextBar();
}

function startDocPoll() {
  stopDocPoll();
  store.set({ docPollStartedAt: Date.now(), docPollTimer: setInterval(pollDocs, DOC_POLL_MS) });
}

function stopDocPoll() {
  if (store.state.docPollTimer) clearInterval(store.state.docPollTimer);
  store.set({ docPollTimer: null });
}

async function pollDocs() {
  if (!store.state.currentConversationId) { stopDocPoll(); return; }
  if (Date.now() - store.state.docPollStartedAt > DOC_POLL_MAX_MS) {
    store.state.uploadingFiles.clear();
    stopDocPoll();
    renderTray();
    renderContextBar();
    toast('Indexing timed out after 5 min', 'warning');
    return;
  }
  try {
    const docs = await chatApi.pollDocsOnce(store.state.currentConversationId);
    store.set({ serverDocs: docs });
    store.state.uploadingFiles.forEach((info, name) => {
      const match = docs.find(d => d.filename === name);
      if (!match) return;
      if (match.id !== info.oldId) store.state.uploadingFiles.delete(name);
    });
    const currIndexed = new Set(docs.filter(d => (d.status || 'indexed') === 'indexed').map(d => d.filename));
    docs.filter(d => (d.status || 'indexed') === 'indexed' && !store.state.prevIndexedNames.has(d.filename)).forEach(d => {
      toast(`${d.filename} indexed`, 'success');
    });
    docs.filter(d => d.status === 'failed' && !store.state.prevIndexedNames.has(d.filename)).forEach(d => {
      toast(`${d.filename} failed — retry`, 'error');
    });
    store.set({ prevIndexedNames: currIndexed });
    const stillIndexing = docs.some(d => d.status === 'pending' || d.status === 'indexing');
    const hasFailed = docs.some(d => d.status === 'failed');
    if (!store.state.uploadingFiles.size && !stillIndexing) stopDocPoll();
    renderTray();
    renderContextBar();
    if (hasFailed && !store.state.uploadingFiles.size && !stillIndexing) setComposerBlocked(false);
  } catch {}
}

async function uploadStaged(conversationId) {
  if (!store.state.stagedFiles.size) return;
  const entries = [...store.state.stagedFiles.entries()];
  store.state.stagedFiles.clear();
  const byName = new Map(store.state.serverDocs.map(d => [d.filename, d.id]));
  for (const [name, file] of entries) {
    const oldId = byName.get(name) || null;
    store.state.uploadingFiles.set(name, { file, queueDepth: 0, startTime: Date.now(), oldId });
  }
  renderTray();
  startDocPoll();
  for (const [name, file] of entries) {
    try {
      const data = await chatApi.uploadFiles(conversationId, file);
      const info = store.state.uploadingFiles.get(name);
      if (info && typeof data.queue_depth === 'number') {
        info.queueDepth = data.queue_depth;
        if (data.queue_depth > 0) toast(`+${data.queue_depth} ahead in queue`, 'info');
      }
      renderTray();
    } catch (e) {
      if (e && typeof e.status === 'number') {
        const info = store.state.uploadingFiles.get(name);
        if (info) {
          store.state.uploadingFiles.delete(name);
          store.set({ serverDocs: [{ id: `local-failed-${Date.now()}`, filename: name, status: 'failed', summary: `Upload failed: HTTP ${e.status}`, conversation_id: conversationId }, ...store.state.serverDocs] });
          toast(`Upload failed: ${name} (HTTP ${e.status})`, 'error');
        }
      } else {
        store.state.uploadingFiles.delete(name);
        toast(`Upload failed: ${name}`, 'error');
      }
      continue;
    }
  }
  renderTray();
  pollDocs();
}

async function uploadFilesNow(files) {
  await ensureConversationId();
  files.forEach(f => store.state.stagedFiles.set(f.name, f));
  renderTray();
  await uploadStaged(store.state.currentConversationId);
}

async function deleteServerDoc(documentId) {
  try {
    await chatApi.deleteServerDoc(documentId);
    const gone = store.state.serverDocs.find(d => String(d.id) === String(documentId));
    const next = store.state.serverDocs.filter(d => String(d.id) !== String(documentId));
    store.set({
      serverDocs: next,
      prevIndexedNames: new Set(next.filter(d => (d.status || 'indexed') === 'indexed').map(d => d.filename))
    });
    if (gone) toast(`Deleted ${gone.filename}`, 'success');
    renderTray();
  } catch (e) {
    console.error(e);
  }
}

async function retryFailedDoc(doc) {
  const cached = store.state.uploadingFiles.get(doc.filename);
  const file = cached ? cached.file : null;
  try {
    await chatApi.deleteServerDoc(doc.id);
  } catch {}
  store.set({ serverDocs: store.state.serverDocs.filter(d => String(d.id) !== String(doc.id)) });
  if (file && store.state.currentConversationId) {
    store.state.stagedFiles.set(file.name, file);
    renderTray();
    uploadStaged(store.state.currentConversationId);
  } else {
    renderTray();
    if (fileInput) fileInput.click();
  }
}

async function checkModelHealth() {
  return chatApi.checkModelHealth();
}

async function fetchModels(forcedHealthy = null) {
  let healthy = forcedHealthy;
  try {
    const { models, path } = await chatApi.fetchModels();
    if (healthy === null) healthy = await checkModelHealth();
    if (models.length) {
      setHeroStatus(healthy, healthy ? 'model loaded' : 'model offline');
      renderModelMenu(models);
      if (!store.state.selectedModel) store.set({ selectedModel: models[0] });
      const first = parseModel(store.state.selectedModel.name || store.state.selectedModel);
      modelNameEl.textContent = first.base;
      const pq = document.getElementById('modelQuant');
      if (pq) { pq.textContent = first.quant; pq.hidden = !first.quant; }
      hint.textContent = 'Enter to send \u2022 Shift+Enter for newline';
      updateHintVisibility();
      setModelPillDisabled(!healthy);
      if (!healthy) { const f2 = parseModel(store.state.selectedModel.name || store.state.selectedModel); modelNameEl.textContent = f2.base; }
      return healthy;
    }
    setHeroStatus(healthy, healthy ? 'no models — add GGUF to ' + (path || '~/.yourstrulyai/models') : 'model offline');
    hint.textContent = 'Enter to send \u2022 Shift+Enter for newline';
    updateHintVisibility();
    setModelPillDisabled(true);
    return healthy;
  } catch (e) {
    setHeroStatus(false, 'model offline');
    hint.textContent = '';
    hint.style.display = 'none';
    hint.setAttribute('hidden', '');
    setModelPillDisabled(true);
    modelNameEl.textContent = 'offline';
    modelMenu.innerHTML = `<div class="mono" style="padding:8px 10px; color:var(--muted-foreground)">Model offline — check ~/.yourstrulyai/models</div>`;
    return false;
  }
}

function renderModelMenu(models) {
  modelMenu.innerHTML = '';
  models.forEach(m => {
    const name = m.name || m;
    const path = m.path || m;
    const isSame = store.state.selectedModel && (store.state.selectedModel.path === path || store.state.selectedModel === m);
    const div = document.createElement('button');
    div.setAttribute('role', 'menuitem');
    div.className = 'model-option ghost' + (isSame ? ' active' : '');
    const p = parseModel(name);
    div.innerHTML = `<span class="model-name" title="${escapeHtml(name)}">${escapeHtml(p.base)}</span>${p.quant ? `<small class="model-quant">${escapeHtml(p.quant)}</small>` : ''}`;
    div.addEventListener('click', () => {
      store.set({ selectedModel: m });
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

function closeHistoryMenu() {
  store.set({ openHistoryMenuId: null });
  document.querySelectorAll('.history-item.menu-open').forEach(el => el.classList.remove('menu-open'));
  document.querySelectorAll('.history-menu').forEach(el => el.hidden = true);
}

function startHistoryRename(id) {
  if (store.state.isStreaming) return;
  store.set({ renamingId: id, openHistoryMenuId: null });
  renderHistory();
  const input = historyList.querySelector(`[data-rename-input="${id}"]`);
  if (input) { input.focus(); input.select(); }
}

function cancelHistoryRename() {
  store.set({ renamingId: null });
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
  const prev = store.state.conversations.find(x => x.id === id);
  const prevTitle = prev ? prev.title : '';
  if (raw === prevTitle) { cancelHistoryRename(); return; }
  input.disabled = true;
  try {
    const updated = await chatApi.saveRename(id, raw);
    const idx = store.state.conversations.findIndex(x => x.id === id);
    if (idx !== -1) store.state.conversations[idx].title = updated.title;
    if (store.state.conversations[idx]) store.state.conversations[idx].updated_at = updated.updated_at;
    store.set({ renamingId: null });
    await loadConversations();
  } catch (e) {
    input.disabled = false;
    input.style.borderColor = 'var(--danger)';
    input.focus();
    console.error(e);
  }
}

async function deleteHistoryConversation(id) {
  if (store.state.isStreaming) return;
  try {
    await chatApi.deleteConversation(id);
    store.set({ conversations: store.state.conversations.filter(x => x.id !== id) });
    if (store.state.currentConversationId === id) {
      store.set({
        currentConversationId: null,
        serverDocs: [],
        prevIndexedNames: new Set()
      });
      store.state.stagedFiles.clear();
      store.state.uploadingFiles.clear();

      stopDocPoll();
      renderTray();
      renderContextBar();
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
    store.set({ pendingHistoryDeleteId: null });
  }
}

function renderHistory() {
  if (!store.state.conversations.length) {
    historyList.innerHTML = `<div class="empty-hint">No conversations yet.</div>`;
    return;
  }
  historyList.innerHTML = '';
  store.state.conversations.forEach(c => {
    const isActive = c.id === store.state.currentConversationId;
    const isRenaming = store.state.renamingId === c.id;
    const isMenuOpen = store.state.openHistoryMenuId === c.id;
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
      titleEl.addEventListener('dblclick', e => { e.stopPropagation(); if (!store.state.isStreaming) startHistoryRename(c.id); });
      div.appendChild(titleEl);

      const menuBtn = document.createElement('button');
      menuBtn.type = 'button';
      menuBtn.className = 'history-menu-btn';
      menuBtn.setAttribute('aria-label', 'Conversation menu');
      menuBtn.setAttribute('aria-haspopup', 'true');
      menuBtn.setAttribute('aria-expanded', isMenuOpen ? 'true' : 'false');
      menuBtn.innerHTML = '';
      menuBtn.textContent = '⋮';
      menuBtn.disabled = store.state.isStreaming;
      menuBtn.addEventListener('click', e => {
        e.stopPropagation();
        if (store.state.isStreaming) return;
        if (store.state.openHistoryMenuId === c.id) closeHistoryMenu();
        else { closeHistoryMenu(); store.set({ openHistoryMenuId: c.id }); renderHistory(); }
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
        store.set({ pendingHistoryDeleteId: c.id });
        if (historyDeleteModal && typeof historyDeleteModal.showModal === 'function') {
          try { historyDeleteModal.showModal(); } catch {}
        } else if (historyDeleteModal) historyDeleteModal.setAttribute('open','');
      });
      menu.append(renameBtn, delBtn);
      div.appendChild(menu);

      div.addEventListener('click', () => {
        if (store.state.isStreaming) return;
        if (store.state.renamingId) return;
        if (store.state.openHistoryMenuId) { closeHistoryMenu(); }
        if (store.state.currentConversationId === c.id) return;
        store.set({ currentConversationId: c.id });
        store.state.stagedFiles.clear();
        store.state.uploadingFiles.clear();
        store.set({ prevIndexedNames: new Set() });

        stopDocPoll();
        setRoutePill(null);
        setComposerBlocked(false);
        ttftEl.textContent = '';
        renderContextBar();
        renderHistory();
        loadMessages(c.id);
      });
    }
    historyList.appendChild(div);
  });
  // dynamic placement: flip to open-up if not enough space below
  if (store.state.openHistoryMenuId) {
    const openItem = historyList.querySelector(`.history-item[data-id="${store.state.openHistoryMenuId}"]`);
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
    const data = await chatApi.loadConversations();
    store.set({ conversations: data });
    renderHistory();
  } catch {
    // keep empty hint, don't block
    if (!store.state.conversations.length) renderHistory();
  }
}

async function loadMessages(conversationId) {
  try {
    const msgs = await chatApi.loadMessages(conversationId);
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
  const hasStaged = store.state.stagedFiles.size > 0;
  if ((!query && !hasStaged) || store.state.isStreaming) return;
  if (store.state.composerBlocked || isIndexing()) {
    toast('Indexing docs… chat paused — ask after Indexed', 'warning');
    setComposerBlocked(true, GATE_HINT);
    return;
  }
  const isFirst = !store.state.currentConversationId && thread.querySelectorAll('.msg').length === 0;
  if (isFirst) {
    tracePath.classList.remove('animate');
    void tracePath.getBoundingClientRect();
    tracePath.classList.add('animate');
  }
  store.set({ isStreaming: true });
  closeHistoryMenu();
  sendBtn.disabled = true;
  ttftEl.textContent = '';
  setRoutePill(null);

  if (!query && hasStaged) {
    try {
      await ensureConversationId();
      const cid = store.state.currentConversationId;
      input.value = '';
      autoResize();
      await uploadStaged(cid);
      await loadConversations();
      renderHistory();
    } catch (e) {
      console.error(e);
    } finally {
      store.set({ isStreaming: false });
      sendBtn.disabled = false;
      input.focus();
    }
    return;
  }

  const attachedNames = [...store.state.stagedFiles.keys()];
  addMessage('user', query, { files: attachedNames });
  input.value = '';
  autoResize();

  const { card, statusEl } = addMessage('assistant', '', { streaming: true });
  setRagStatus(statusEl, 'deciding');
  updateAssistantCard(card, '', false);
  let acc = '';
  let start = performance.now();
  let ttftDone = false;

  try {
    await ensureConversationId();
    if (attachedNames.length) await uploadStaged(store.state.currentConversationId);

    const modelPath = store.state.selectedModel ? (store.state.selectedModel.path || store.state.selectedModel) : undefined;
    const { resp, conversationId: newId, route: routeHeader, ttft: ttftHeader } = await chatApi.chatStream({
      query,
      model: modelPath || undefined,
      conversationId: store.state.currentConversationId
    });
    if (newId) store.set({ currentConversationId: newId });
    if (routeHeader) setRoutePill(routeHeader);
    if (!resp.ok) {
      const t = await resp.text();
      if (newId) {
        await loadConversations();
        renderHistory();
      }
      throw new Error(`HTTP ${resp.status}: ${t}`);
    }
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
        let data = null;
        for (const ln of lines) {
          if (ln.startsWith('data:')) data = ln.slice(5).trim();
        }
        if (data === null || data === '') continue;
        if (data === '[DONE]') {
          stopTicker(statusEl);
          updateAssistantCard(card, acc, true);
          store.set({ isStreaming: false });
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
              stopTicker(statusEl);
              const ms = performance.now() - start;
              if (!ttftHeader) ttftEl.textContent = `TTFT ${ms.toFixed(0)} ms`;
            }
            acc += delta;
            updateAssistantCard(card, acc, false);
          }
        } catch {}
      }
    }
    stopTicker(statusEl);
    updateAssistantCard(card, acc || 'No response.', true);
    await loadConversations();
  } catch (e) {
    stopTicker(statusEl);
    const m = e && e.message || '';
    const isBusy = m.includes('429') || m.includes('System Busy');
    const msg = isBusy ? '**System Busy — model is generating.** Please wait and try again.' : '**Could not reach backend / model not loaded.** Check ~/.yourstrulyai/models and backend logs.';
    updateAssistantCard(card, msg, true);
  } finally {
    stopTicker(statusEl);
    store.set({ isStreaming: false });
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
    if (store.state.renamingId) {
      const input = historyList.querySelector(`[data-rename-input="${store.state.renamingId}"]`);
      const raw = input ? input.value.trim() : '';
      const orig = store.state.conversations.find(x => x.id === store.state.renamingId);
      const origTitle = orig ? orig.title : '';
      if (raw && raw !== origTitle && raw.length <= 250) saveHistoryRename(store.state.renamingId);
      else cancelHistoryRename();
    }
    return;
  }
  // click inside another item while renaming — treat as outside save
  if (store.state.renamingId && insideItem.dataset.id !== String(store.state.renamingId)) {
    const input = historyList.querySelector(`[data-rename-input="${store.state.renamingId}"]`);
    const raw = input ? input.value.trim() : '';
    const orig = store.state.conversations.find(x => x.id === store.state.renamingId);
    const origTitle = orig ? orig.title : '';
    if (raw && raw !== origTitle && raw.length <= 250) saveHistoryRename(store.state.renamingId);
    else cancelHistoryRename();
  }
});
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (store.state.renamingId) { e.preventDefault(); cancelHistoryRename(); }
  else if (store.state.openHistoryMenuId) { e.preventDefault(); closeHistoryMenu(); renderHistory(); }
});
if (historyDeleteCancel) historyDeleteCancel.addEventListener('click', () => {
  store.set({ pendingHistoryDeleteId: null });
  if (historyDeleteModal && typeof historyDeleteModal.close === 'function') try { historyDeleteModal.close(); } catch {}
  if (historyDeleteModal) historyDeleteModal.removeAttribute('open');
});
if (historyDeleteConfirm) historyDeleteConfirm.addEventListener('click', async (e) => {
  e.preventDefault();
  const id = store.state.pendingHistoryDeleteId;
  if (!id) return;
  if (historyDeleteModal && typeof historyDeleteModal.close === 'function') try { historyDeleteModal.close(); } catch {}
  if (historyDeleteModal) historyDeleteModal.removeAttribute('open');
  await deleteHistoryConversation(id);
});
if (historyDeleteModal) {
  historyDeleteModal.addEventListener('click', (e) => {
    if (e.target === historyDeleteModal) {
      store.set({ pendingHistoryDeleteId: null });
      try { historyDeleteModal.close(); } catch {}
      historyDeleteModal.removeAttribute('open');
    }
  });
  historyDeleteModal.addEventListener('cancel', (e) => {
    e.preventDefault();
    store.set({ pendingHistoryDeleteId: null });
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
if (contextToggle) {
  try { store.state.contextCollapsed = localStorage.getItem('yt_context_collapsed') === '1'; } catch {}
  contextToggle.addEventListener('click', (e) => {
    e.preventDefault();
    store.set({ contextCollapsed: !store.state.contextCollapsed });
    try { localStorage.setItem('yt_context_collapsed', store.state.contextCollapsed ? '1' : '0'); } catch {}
    renderContextBar();
  });
  window.addEventListener('resize', () => renderContextBar());
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
  store.set({ currentConversationId: null, serverDocs: [], prevIndexedNames: new Set() });
  store.state.stagedFiles.clear();
  store.state.uploadingFiles.clear();

  stopDocPoll();
  renderTray();
  renderContextBar();
  clearThread();
  showEmpty(true);
  renderStarters();
  ttftEl.textContent = '';
  setRoutePill(null);
  setComposerBlocked(false);
  closeHistoryMenu();
  cancelHistoryRename();
  renderHistory();
  input.focus();
});

const startersEl = $('#starters');
const STARTER_POOL = [
  { prompt: 'Help me plan my week day by day', title: 'Plan my week', sub: 'organise the days ahead' },
  { prompt: 'Draft a polite email to reschedule a meeting', title: 'Draft an email', sub: 'polite and to the point' },
  { prompt: 'Explain how interest rates work in plain words', title: 'Explain simply', sub: 'any topic, plain words' },
  { prompt: 'Brainstorm birthday gift ideas under $50', title: 'Brainstorm ideas', sub: 'names, gifts, trips' },
  { prompt: 'Summarise the following text in three bullet points: ', title: 'Summarise text', sub: 'paste it in after' },
  { prompt: 'Make a packing checklist for a weekend trip', title: 'Make a checklist', sub: 'packing, moving, tasks' },
  { prompt: 'Rewrite this to sound more confident: ', title: 'Write it better', sub: 'sharpen my draft' },
  { prompt: 'Give me pros and cons of buying vs renting a bike', title: 'Pros and cons', sub: 'decide with me' },
  { prompt: 'Teach me the basics of sourdough in five steps', title: 'Learn something', sub: 'a quick lesson' },
  { prompt: 'Suggest a simple 20-minute morning routine', title: 'Morning routine', sub: 'start the day right' },
  { prompt: 'Explain what the internet is to a five-year-old', title: 'Explain to a child', sub: 'really, really simple' },
  { prompt: 'Help me think through whether to take up running', title: 'Talk it through', sub: 'think out loud together' },
];

function renderStarters() {
  if (!startersEl) return;
  const pool = [...STARTER_POOL];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  startersEl.innerHTML = '';
  pool.slice(0, 4).forEach(s => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'starter';
    btn.append(document.createTextNode(s.title + ' '));
    const small = document.createElement('small');
    small.textContent = s.sub;
    btn.appendChild(small);
    btn.addEventListener('click', () => {
      input.value = s.prompt;
      autoResize();
      input.focus();
    });
    startersEl.appendChild(btn);
  });
}

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
  renderStarters();
  const h = await fetchModels();
  lastHealthy = h;
  autoResize();
  if (!store.state.currentConversationId) showEmpty(true);
  else loadServerDocs(store.state.currentConversationId);
  updateHintVisibility();
  setInterval(pollHealth, 10000);
})();
