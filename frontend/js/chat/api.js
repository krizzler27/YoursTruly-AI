import { API_BASE } from '../shared/config.js';
import { isChatModel } from '../shared/utils.js';

export async function ensureConversationId(currentConversationId) {
  if (currentConversationId) return currentConversationId;
  const r = await fetch(`${API_BASE}/api/conversations`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({})
  });
  if (!r.ok) throw new Error(`Could not create conversation: HTTP ${r.status}`);
  const conv = await r.json();
  return conv.id;
}

export async function loadServerDocs(conversationId) {
  if (!conversationId) return [];
  const r = await fetch(`${API_BASE}/api/documents?conversation_id=${encodeURIComponent(conversationId)}`);
  if (!r.ok) throw new Error('failed');
  return await r.json();
}

export async function pollDocsOnce(conversationId) {
  const r = await fetch(`${API_BASE}/api/documents?conversation_id=${encodeURIComponent(conversationId)}`);
  if (!r.ok) throw new Error('failed');
  return await r.json();
}

export async function uploadFiles(conversationId, file) {
  const form = new FormData();
  form.append('file', file, file.name);
  form.append('conversation_id', conversationId);
  const r = await fetch(`${API_BASE}/api/ingest`, { method: 'POST', body: form });
  if (!r.ok) {
    const err = new Error(`Upload failed: HTTP ${r.status}`);
    err.status = r.status;
    throw err;
  }
  const data = await r.json().catch(() => ({}));
  return data;
}

export async function deleteServerDoc(documentId) {
  const r = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return true;
}

export async function checkModelHealth() {
  try {
    const r = await fetch(`${API_BASE}/api/health`, { headers: { 'accept': 'application/json' } });
    if (!r.ok) return false;
    const data = await r.json();
    return (data.status === 'ready' || data.status === 'idle') && !!data.exists;
  } catch { return false; }
}

export async function fetchModels() {
  const r = await fetch(`${API_BASE}/api/models`, { headers: { 'accept': 'application/json' } });
  if (!r.ok) throw new Error('no models');
  const data = await r.json();
  const models = (data.models || [])
    .map(m => typeof m === 'string' ? { name: m, path: m } : m)
    .filter(m => isChatModel(m.name || m.path));
  return { models, path: data.path, raw: data };
}

export async function loadConversations() {
  const r = await fetch(`${API_BASE}/api/conversations`);
  if (!r.ok) throw new Error('failed');
  const data = await r.json();
  return Array.isArray(data) ? data : [];
}

export async function loadMessages(conversationId) {
  const r = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(conversationId)}/messages`);
  if (!r.ok) throw new Error('failed');
  return await r.json();
}

export async function saveRename(id, title) {
  const r = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ title })
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(d.detail || `HTTP ${r.status}`);
  }
  return await r.json();
}

export async function deleteConversation(id) {
  const r = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(d.detail || `HTTP ${r.status}`);
  }
  return true;
}

export async function chatStream({ query, model, conversationId }) {
  const payload = { query, model: model || undefined };
  if (conversationId) payload.conversation_id = conversationId;
  const resp = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const newId = resp.headers.get('X-Conversation-Id') || resp.headers.get('x-conversation-id');
  const routeHeader = resp.headers.get('X-Route') || resp.headers.get('x-route');
  const ttftHeader = resp.headers.get('X-TTFT') || resp.headers.get('x-ttft');
  return { resp, conversationId: newId, route: routeHeader, ttft: ttftHeader };
}
