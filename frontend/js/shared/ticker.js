export const TICKER_HINTS = [
  "Ask with a filename for sharper answers",
  "Add a heading to narrow the search",
  "Cite answers as [filename:heading]",
  "Keep questions short for better hits",
  "Mention names exactly as written",
  "Follow up in same chat for context"
];

export const TICKER_JOKES = [
  "Thinking locally without any cloud",
  "Chewing tokens slowly like a llama",
  "Reading your files right at home",
  "Keeping your docs off the cloud",
  "Shuffling vectors into neat piles",
  "Warming up the tiny local model",
  "Sipping power to stay on 8 gigs",
  "Filing facts without leaving home"
];

export const TICKER_BACKEND = [
  "Deciding between direct and search",
  "Embedding query on nomic Q4",
  "Searching vectors and keywords together",
  "Fusing ranks with RRF k 60",
  "Reading top 5 chunks for context",
  "Building answer with llama.cpp"
];

export function createTicker(threadEl) {
  let timer = null;
  let idx = 0;
  const pool = [...TICKER_HINTS, ...TICKER_JOKES, ...TICKER_BACKEND];

  function stop(statusEl) {
    if (timer) clearInterval(timer);
    timer = null;
    if (statusEl) { statusEl.hidden = true; statusEl.textContent = ''; }
  }

  function set(statusEl) {
    if (!statusEl) return;
    if (timer) clearInterval(timer);
    statusEl.hidden = false;
    const render = () => {
      statusEl.innerHTML = '';
      const load = document.createElement('span');
      load.className = 'doc-spin';
      load.setAttribute('aria-hidden', 'true');
      const label = document.createElement('span');
      label.textContent = pool[idx % pool.length];
      idx += 1;
      statusEl.append(load, label);
      if (threadEl && threadEl.parentElement) {
        threadEl.parentElement.scrollTop = threadEl.parentElement.scrollHeight;
      }
    };
    render();
    timer = setInterval(render, 2500);
  }

  return { set, stop };
}
