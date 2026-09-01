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

let selectedModel = null;
let isStreaming = false;
let turns = []; // memory-only: {role, content}
let conversations = []; // local ledger for sidebar

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

function mdSimple(src) {
  // Store code fences
  const fences = [];
  let html = src.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
    const idx = fences.length;
    fences.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    return `\x00FENCE${idx}\x00`;
  });
  // inline code
  html = html.replace(/`([^`]+)`/g, (_, c) => `<code>${escapeHtml(c)}</code>`);
  html = escapeHtml(html);
  // restore fences (already escaped)
  html = html.replace(/\x00FENCE(\d+)\x00/g, (_, i) => fences[Number(i)]);
  // bold
  html = html.replace(/\*\*([^\n*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__([^\n_]+)__/g, '<strong>$1</strong>');
  // italic (after bold)
  html = html.replace(/\*([^\n*]+)\*/g, '<em>$1</em>');
  // paragraphs + lists
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
      // if line already contains block tag, keep as is
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
}

function autoResize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
}

function showEmpty(show) {
  empty.style.display = show ? 'block' : 'none';
  trace.style.display = show ? 'block' : 'none';
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
  // render — cursor only, no spinner
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
  return { wrap, card, avatar };
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

async function fetchModels() {
  try {
    const r = await fetch(`${API_BASE}/api/models`, { headers: { 'accept': 'application/json' } });
    if (!r.ok) throw new Error('no models');
    const data = await r.json();
    const models = data.models || [];
    if (models.length) {
      setStatus(true, 'local • ready');
      renderModelMenu(models);
      if (!selectedModel) selectedModel = models[0];
      modelNameEl.textContent = selectedModel;
      hint.textContent = `${models.length} model${models.length>1?'s':''} available`;
      return;
    }
    setStatus(true, 'no models — pull with ollama');
  } catch (e) {
    setStatus(false, 'offline — start ollama');
    hint.textContent = 'Start with: ollama serve & ollama pull qwen2.5:3b';
  }
}

function renderModelMenu(models) {
  modelMenu.innerHTML = '';
  models.forEach(m => {
    const div = document.createElement('button');
    div.setAttribute('role', 'menuitem');
    div.className = 'model-option ghost' + (m === selectedModel ? ' active' : '');
    div.innerHTML = `<span>${escapeHtml(m)}</span><small>${m.split(':').pop()}</small>`;
    div.addEventListener('click', () => {
      selectedModel = m;
      modelNameEl.textContent = m;
      [...modelMenu.children].forEach(c => c.classList.remove('active'));
      div.classList.add('active');
      modelMenu.hidePopover?.();
      // also hide via toggle
      try { modelMenu.hidePopover(); } catch {}
    });
    modelMenu.appendChild(div);
  });
}

function ensureConversation(title) {
  if (!conversations.length) {
    const id = Date.now().toString(36);
    conversations.unshift({ id, title: title.slice(0, 48), at: new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}), turns: 0 });
  }
  conversations[0].turns = turns.length;
  renderHistory();
}

function renderHistory() {
  if (!conversations.length) {
    historyList.innerHTML = `<div class="empty-hint">No conversations yet.</div>`;
    return;
  }
  historyList.innerHTML = '';
  conversations.forEach((c, idx) => {
    const div = document.createElement('div');
    div.className = 'history-item' + (idx===0 ? ' active' : '');
    const count = c.turns || turns.length;
    div.innerHTML = `<span>${escapeHtml(c.title)}</span><small>${c.at} • ${count} messages</small>`;
    historyList.appendChild(div);
  });
}

async function send() {
  const query = input.value.trim();
  if (!query || isStreaming) return;
  // trace animation on first send
  if (turns.length === 0) {
    tracePath.classList.remove('animate');
    void tracePath.getBoundingClientRect();
    tracePath.classList.add('animate');
  }
  isStreaming = true;
  sendBtn.disabled = true;
  ttftEl.textContent = '';
  turns.push({ role: 'user', content: query });
  addMessage('user', query);
  input.value = '';
  autoResize();
  ensureConversation(query);

  const { card } = addMessage('assistant', '', { streaming: true });
  let acc = '';
  let start = performance.now();
  let ttftDone = false;

  try {
    const resp = await fetch(`${API_BASE}/api/chat/stream`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ query, model: selectedModel || undefined })
    });
    if (!resp.ok) {
      const t = await resp.text();
      throw new Error(t || `HTTP ${resp.status}`);
    }
    // TTFT from header (backend calculates)
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
          turns.push({ role: 'assistant', content: acc });
          isStreaming = false;
          sendBtn.disabled = false;
          if (!ttftDone && !ttftHeader) {
            const ms = performance.now() - start;
            ttftEl.textContent = `TTFT ${ms.toFixed(0)} ms`;
          }
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
  } catch (e) {
    updateAssistantCard(card, `**Could not reach backend.** Is Ollama running?`, true);
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    input.focus();
  }
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
$('#newChat').addEventListener('click', () => {
  thread.querySelectorAll('.msg').forEach(el => el.remove());
  turns = [];
  conversations = [];
  renderHistory();
  showEmpty(true);
  ttftEl.textContent = '';
  input.focus();
});

// Starters
document.querySelectorAll('[data-starter]').forEach(btn => {
  btn.addEventListener('click', () => {
    input.value = btn.dataset.starter;
    autoResize();
    input.focus();
  });
});

// Popover fallback for browsers without popover API
if (!HTMLElement.prototype.hasOwnProperty('popover')) {
  modelPill.addEventListener('click', () => {
    modelMenu.style.display = modelMenu.style.display === 'block' ? 'none' : 'block';
    modelMenu.style.position = 'absolute';
  });
  document.addEventListener('click', (e) => {
    if (!modelPill.contains(e.target) && !modelMenu.contains(e.target)) modelMenu.style.display = 'none';
  });
}

// Init
renderHistory();
fetchModels();
autoResize();
showEmpty(true);
