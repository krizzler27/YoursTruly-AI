export const $ = (s, r = document) => r.querySelector(s);

export function escapeHtml(s) {
  return s.replace(/[&<>"']/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
}

export function mdSimple(src) {
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
  function closeList() { if (inList) { out += `</${listType}>`; inList = false; listType = null; } }
  for (let line of lines) {
    if (line.includes('<pre>')) { closeList(); out += line; continue; }
    if (/^\s*$/.test(line)) { closeList(); continue; }
    const ulMatch = line.match(/^\s*[-*]\s+(.*)$/);
    const olMatch = line.match(/^\s*\d+\.\s+(.*)$/);
    if (ulMatch) {
      if (!inList || listType !== 'ul') { closeList(); out += '<ul>'; inList = true; listType = 'ul'; }
      out += `<li>${ulMatch[1]}</li>`;
    } else if (olMatch) {
      if (!inList || listType !== 'ol') { closeList(); out += '<ol>'; inList = true; listType = 'ol'; }
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

export function shortFileName(name, maxLen = 22) {
  const t = (name || '').trim();
  return t.length > maxLen ? t.slice(0, maxLen) + '…' : t;
}

export function truncateTitle(title, maxLen = 32) {
  const t = title.trim();
  return t.length > maxLen ? t.slice(0, maxLen) + '…' : t;
}

export function parseModel(raw) {
  if (!raw) return { base: '', quant: '' };
  const baseRaw = raw.replace(/\.gguf$/i, '').trim();
  const m = baseRaw.match(/^(.*?)[-_](Q\d.*)$/i);
  if (m) return { base: m[1], quant: m[2] };
  return { base: baseRaw.replace(/_/g, ' '), quant: '' };
}

export function displayName(raw) {
  const p = parseModel(raw);
  return p.quant ? `${p.base} (${p.quant})` : p.base;
}

export function isChatModel(name) {
  return !String(name || '').toLowerCase().includes('embed');
}
