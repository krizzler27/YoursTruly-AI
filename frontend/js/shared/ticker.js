export const TICKER_QUIPS = [
  "Thinking it over…",
  "Reading your files…",
  "Putting the pieces together…",
  "Writing it up…",
  "One moment, checking my notes…",
  "Loyal, local, and slightly overcaffeinated…",
  "No clouds were consulted…",
  "YoursTruly is on it…",
  "Pondering politely…",
  "Dotting the i's…",
  "Ask me anything - I live here…",
  "Slow and steady, like a good neighbour…",
  "Keeping it between us…",
  "Warming up the thinking chair…",
  "Almost there, promise…",
  "Crossing the t's…",
  "Just you, me, and this machine…",
  "Good things take a few tokens…",
  "Flipping through the pages…",
  "Nearly done, tying the bow…",
  "Hmm, let me think…",
  "Brewing a fresh answer…",
  "No peeking at the neighbours…",
  "Your secrets are safe here…",
  "Stretching the little grey cells…",
  "Mind the dust, tidying up…",
  "Taking the scenic route…",
  "Home-cooked answers only…",
  "Listening carefully…",
  "And… here it comes…"
];

export function createTicker(threadEl) {
  let timer = null;
  let idx = 0;
  const pool = [...TICKER_QUIPS];

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
