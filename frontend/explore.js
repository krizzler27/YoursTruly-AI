const API_BASE = 'http://127.0.0.1:8000';
const $ = (s, r=document) => r.querySelector(s);

const hwDot = $('#hwDot');
const hwStatus = $('#hwStatus');
const kpiRam = $('#kpiRam');
const kpiVram = $('#kpiVram');
const kpiCores = $('#kpiCores');
const kpiGpu = $('#kpiGpu');
const kpiMode = $('#kpiMode');
const ramFill = $('#ramFill');
const vramFill = $('#vramFill');
const kpiRamMeta = $('#kpiRamMeta');
const kpiVramMeta = $('#kpiVramMeta');
const cachePath = $('#cachePath');
const cacheCount = $('#cacheCount');
const localList = $('#localList');

const grid = $('#catalogGrid');
const countEl = $('#catalogCount');
const pathEl = $('#catalogPath');
const statusEl = $('#catalogStatus');
const emptyEl = $('#emptyCatalog');
const providerBtn = $('#providerBtn');
const providerMenu = $('#providerMenu');
const providersWrap = $('#providersWrap');
const searchEl = $('#search');
const limitEl = $('#limit');
const sortEl = $('#sort');
const perfectOnlyEl = $('#perfectOnly');
const includeCommunityEl = $('#includeCommunity');
const toastStack = $('#toastStack');
const downloadDock = $('#downloadDock');
const dockList = $('#dockList');
const dockFab = $('#dockFab');
const dockClose = $('#dockClose');

const PROVIDERS = ['meta','alibaba','google','mistral','microsoft','deepseek'];
let selectedProviders = new Set(PROVIDERS);
let lastData = null;

// --- Toast: info / success / warning / error ---
function toast(msg, type='info', ttl){
  if(!toastStack) return;
  const el = document.createElement('div');
  el.className = `toast-item ${type}`;
  const iconMap = {info:'i', success:'✓', warning:'!', error:'×'};
  const icon = document.createElement('span');
  icon.className = 'toast-icon';
  icon.textContent = iconMap[type] || 'i';
  const text = document.createElement('span');
  text.textContent = msg;
  text.style.flex='1';
  text.style.minWidth='0';
  const close = document.createElement('button');
  close.className='toast-close';
  close.type='button';
  close.textContent='×';
  close.addEventListener('click', ()=> dismiss());
  el.append(icon, text, close);
  toastStack.appendChild(el);
  const duration = ttl ?? (type==='error' ? 6000 : type==='warning' ? 5000 : 4000);
  let t = setTimeout(dismiss, duration);
  function dismiss(){
    clearTimeout(t);
    el.style.animation='toastOut 140ms ease forwards';
    setTimeout(()=> el.remove(), 150);
  }
  el.addEventListener('mouseenter', ()=> clearTimeout(t));
  el.addEventListener('mouseleave', ()=> t=setTimeout(dismiss, 1200));
}
function esc(s){ return s.replace(/[&<>"']/g, m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }

function formatProvidersLabel(set){
  if(!set || set.size===0 || set.size===PROVIDERS.length) return 'All';
  const csv = [...set].join(', ');
  // truncate to fit button ~14 chars; actual overflow also handled by CSS
  if(csv.length > 14) return csv.slice(0,10).trimEnd() + '…';
  return csv;
}
function renderProviderMenu(){
  providerMenu.innerHTML='';
  PROVIDERS.forEach(p=>{
    const label=document.createElement('label');
    label.className='provider-option mono';
    const cb=document.createElement('input');
    cb.type='checkbox';
    cb.value=p;
    cb.checked=selectedProviders.has(p);
    cb.addEventListener('change', ()=>{
      if(cb.checked) selectedProviders.add(p); else selectedProviders.delete(p);
      providerBtn.textContent=formatProvidersLabel(selectedProviders);
      providerBtn.title=[...selectedProviders].join(', ') || 'All';
    });
    const span=document.createElement('span');
    span.textContent=p;
    label.append(cb, span);
    providerMenu.appendChild(label);
  });
  providerBtn.textContent=formatProvidersLabel(selectedProviders);
  providerBtn.title=[...selectedProviders].join(', ') || 'All';
}
function getSelectedProviders(){
  return [...selectedProviders];
}
function resetProviders(){
  selectedProviders=new Set(PROVIDERS);
  renderProviderMenu();
}
function toggleProviderMenu(force){
  const willOpen = typeof force==='boolean' ? force : providerMenu.hidden;
  providerMenu.hidden=!willOpen;
  providerBtn.setAttribute('aria-expanded', String(willOpen));
  providersWrap.classList.toggle('open', willOpen);
}

// --- Download dock: single non-blocking + polling (3s active, 0 grace) ---
let dockJobs = new Map(); // job_id -> job
let dockPollTimer = null;
let dockGraceTimer = null;
let dockMinimized = false;

function stopDockPolling(){
  if(dockPollTimer){ clearInterval(dockPollTimer); dockPollTimer=null; }
}
function clearGrace(){
  if(dockGraceTimer){ clearTimeout(dockGraceTimer); dockGraceTimer=null; }
}

function ensureDockPolling(){
  clearGrace();
  if(dockPollTimer) return;
  dockPollTimer = setInterval(async ()=>{
    try{
      const r = await fetch(`${API_BASE}/api/downloads`);
      if(!r.ok) return;
      const data = await r.json();
      const jobs = data.jobs || [];
      jobs.forEach(j=>{
        const prev = dockJobs.get(j.id);
        dockJobs.set(j.id, j);
        if(prev && prev.status!==j.status){
          if(j.status==='completed') { toast(`Download complete: ${j.filename || j.repo_id}`, 'success'); fetchLocal(); }
          if(j.status==='failed') toast(`Download failed: ${j.repo_id} — ${j.error ? j.error.slice(0,120) : 'see logs'}`, 'error');
        }
      });
      renderDock();
      const stillActive = jobs.some(j=> j.status==='queued' || j.status==='downloading');
      if(!stillActive){
        // static 30s grace: stop polling, keep dock visible from cached dockJobs, then auto-clear
        stopDockPolling();
        clearGrace();
        dockGraceTimer = setTimeout(()=>{
          dockJobs.clear();
          renderDock();
          dockGraceTimer=null;
        }, 30000);
      }
    }catch{}
  }, 3000);
}

function renderDock(){
  const jobs = [...dockJobs.values()].sort((a,b)=> new Date(b.created_at)-new Date(a.created_at));
  const activeCount = jobs.filter(j=> j.status==='queued'||j.status==='downloading').length;
  const total = jobs.length;
  const hasActive = activeCount>0;
  downloadDock.classList.toggle('pulse', hasActive);
  dockFab.classList.toggle('pulse', hasActive);
  if(total===0){
    downloadDock.hidden = true;
    dockFab.hidden = true;
    dockList.innerHTML = '<div class="dock-empty mono">No downloads yet.</div>';
    return;
  }
  if(dockMinimized){
    downloadDock.hidden = true;
    dockFab.hidden = false;
    return;
  }
  downloadDock.hidden = false;
  dockFab.hidden = true;
  if(!jobs.length){
    dockList.innerHTML = '<div class="dock-empty mono">No downloads yet.</div>';
    return;
  }
  dockList.innerHTML = jobs.map(j=>{
    const pct = Math.max(0, Math.min(100, Number(j.progress)||0));
    const status = j.status || 'queued';
    const name = esc(j.repo_id || j.id);
    const quant = esc(j.quant || '');
    const barClass = status==='completed' ? 'completed' : status==='failed' ? 'failed' : 'downloading';
    const canCancel = status==='queued' || status==='downloading';
    const err = j.error ? ` title="${esc(j.error.slice(0,200))}"` : '';
    return `<div class="dock-item" data-id="${esc(j.id)}">
      <div class="dock-item-head">
        <span class="dock-item-name" title="${name}">${name}</span>
        <span class="dock-item-quant">${quant}</span>
        <span class="dock-item-status ${status}"${err}>${status}</span>
      </div>
      <div class="dock-bar"><i class="${barClass}" style="width:${status==='failed'?100:pct}%"></i></div>
      <div class="dock-item-foot">
        <span class="dock-meta mono">${j.filename ? esc(j.filename) : ''}</span>
        <span class="dock-actions">
          ${canCancel ? `<button type="button" data-cancel="${esc(j.id)}" class="danger">Cancel</button>` : ''}
          ${status==='completed' && j.path ? `<span class="mono" style="font-size:10px;color:var(--lab-moss)">✓ saved</span>` : ''}
        </span>
      </div>
    </div>`;
  }).join('');
  // bind cancel
  dockList.querySelectorAll('[data-cancel]').forEach(btn=>{
    btn.addEventListener('click', async ()=>{
      const id = btn.getAttribute('data-cancel');
      if(!id) return;
      btn.disabled=true; btn.textContent='…';
      try{
        const r = await fetch(`${API_BASE}/api/downloads/${id}`, {method:'DELETE'});
        const d = await r.json().catch(()=>({}));
        if(!r.ok) toast(d.detail||'Cancel failed','error');
        else toast('Cancelled download','warning');
        // refresh immediately
        const r2 = await fetch(`${API_BASE}/api/downloads`);
        if(r2.ok){ const dd=await r2.json(); dd.jobs.forEach(j=>dockJobs.set(j.id,j)); renderDock(); }
      }catch(e){ toast(e.message||'Cancel failed','error'); }
      finally{ btn.disabled=false; }
    });
  });
  // also update per-card buttons if catalog visible
  updateCardButtons(jobs);
}

function updateCardButtons(jobs){
  const byRepo = new Map();
  jobs.forEach(j=>{ if(j.status==='queued'||j.status==='downloading') byRepo.set(j.repo_id, j); });
  document.querySelectorAll('.download').forEach(btn=>{
    const repo = btn.dataset.repo;
    const job = repo ? byRepo.get(repo) : null;
    if(job){
      btn.disabled = true;
      btn.textContent = job.status==='queued' ? 'Queued…' : `Downloading ${job.progress||0}%`;
    } else {
      // only re-enable if not handled by current active job; leave completed as Download
      if(btn.dataset._busy) {
        btn.disabled=false;
        btn.textContent='Download';
        delete btn.dataset._busy;
      }
    }
  });
}

function showDock(){
  dockMinimized=false;
  renderDock();
}
function hideDockToFab(){
  dockMinimized=true;
  downloadDock.hidden=true;
  dockFab.hidden = dockJobs.size===0;
}

// dock controls — only X closes to fab
if(dockClose) dockClose.addEventListener('click', ()=>{
  clearGrace(); stopDockPolling();
  dockMinimized=true;
  downloadDock.hidden=true;
  dockFab.hidden = dockJobs.size===0;
});
if(dockFab) dockFab.addEventListener('click', showDock);

async function fetchSystem(){
  try{
    const r=await fetch(`${API_BASE}/api/system`);
    if(!r.ok) throw new Error();
    const j=await r.json();
    const sys=j.system || j.raw?.system || {};
    // llmfit keys: available_ram_gb (free now), total_ram_gb, gpu_vram_gb, backend/has_gpu
    const ramAvail = sys.available_ram_gb ?? sys.memory_available_gb ?? sys.ram_gb ?? 0;
    const ramTotal = sys.total_ram_gb ?? 0;
    const vram = sys.gpu_vram_gb ?? sys.vram_gb ?? sys.gpus?.[0]?.vram_gb ?? sys.gpu?.vram_gb ?? 0;
    const cores = sys.cpu_cores ?? sys.cores ?? sys.cpu?.cores ?? '—';
    const gpu = sys.gpu_name ?? sys.gpu?.name ?? sys.gpus?.[0]?.name ?? 'none';
    // dGPU = discrete (separate card, own VRAM/GDDR). iGPU = integrated inside CPU (shares system RAM, like 660M). Keep label VRAM — apt for both.
    const hasGpu = !!(sys.has_gpu ?? sys.gpu_available_gb ?? sys.gpus?.length);
    const backend = sys.backend ?? sys.gpus?.[0]?.backend ?? '';
    const mode = hasGpu && backend ? `${backend}` : hasGpu ? 'GPU' : 'CPU';
    hwDot.classList.remove('off');
    hwStatus.textContent = 'hardware ready';
    kpiRam.textContent = ramAvail ? `${Number(ramAvail).toFixed(1)} GB` : '—';
    kpiVram.textContent = vram ? `${Number(vram).toFixed(1)} GB` : '—';
    kpiCores.textContent = String(cores);
    kpiGpu.textContent = String(gpu).slice(0,22);
    kpiMode.textContent = String(mode);
    // bars: ram fill as available vs total (if total known)
    const ramBase = Number(ramTotal) || 16;
    const ramPct = Math.min(100, (Number(ramAvail)||8)/ramBase*100);
    ramFill.style.width = ramPct + '%';
    ramFill.className = ramPct > 85 ? 'over' : '';
    const vramPct = vram ? Math.min(100, Number(vram)/ramBase*100) : 35;
    vramFill.style.width = vramPct + '%';
    kpiRamMeta.textContent = ramTotal ? `${Number(ramAvail).toFixed(1)} free / ${Number(ramTotal).toFixed(1)} total` : `available for model`;
    kpiVramMeta.textContent = vram ? `VRAM for offload` : `CPU only`;
  }catch{
    hwDot.classList.add('off');
    hwStatus.textContent = 'offline — llmfit not found';
  }
}

let installedSet = new Set(); // lowercased gguf filenames for isInstalled check
function isRepoInstalled(repo, quant){
  if(!repo || !installedSet.size) return false;
  const rb = repo.split('/').pop().toLowerCase().replace(/-gguf$/,'').replace(/\.gguf$/,'');
  const core = rb.split('-').slice(0,2).join('-');
  const qn = (quant||'').toLowerCase().replace(/_/g,'-');
  for(const f of installedSet){
    if(core && !f.includes(core)) continue;
    if(qn){
      const fn = f.replace(/[-_]/g,'');
      const qf = qn.replace(/[-_]/g,'');
      if(!fn.includes(qf)) continue;
    }
    return true;
  }
  return false;
}
function refreshInstalledButtons(){
  document.querySelectorAll('.download').forEach(btn=>{
    if(btn.dataset.installed==='1') return; // already marked
    const repo=btn.dataset.repo, quant=btn.dataset.quant;
    if(isRepoInstalled(repo, quant)){
      btn.textContent='✓ Installed';
      btn.disabled=true;
      btn.dataset.installed='1';
      btn.title='Already installed';
      btn.style.background='var(--lab-moss)';
      btn.style.borderColor='var(--lab-moss)';
      btn.style.opacity='1';
    }
  });
}
async function fetchLocal(){
  try{
    const r=await fetch(`${API_BASE}/api/models`);
    const j=await r.json();
    const models=j.models||[];
    installedSet = new Set(models.map(m=>m.toLowerCase()));
    cachePath.textContent=j.path||'~/.yourstrulyai/models';
    cacheCount.textContent=`${models.length} installed`;
    if(!models.length){
      localList.innerHTML='<div class="empty mono">No models installed — pick a Perfect fit from the catalog below and Download.</div>';
    } else {
      localList.innerHTML='';
      models.forEach(name=>{
        const div=document.createElement('div');
        div.className='local-item';
        div.innerHTML=`<span>${esc(name)}</span><button type="button" data-file="${esc(name)}">Delete</button>`;
        div.querySelector('button').addEventListener('click', async ()=>{
          if(!confirm(`Delete ${name}?`)) return;
          const del=await fetch(`${API_BASE}/api/models/${encodeURIComponent(name)}`,{method:'DELETE'});
          const dj=await del.json().catch(()=>({}));
          if(!del.ok) toast(dj.detail||'Delete failed','error');
          else { toast('Deleted ' + name,'success'); }
          await fetchLocal();
          // re-enable catalog buttons after delete
          document.querySelectorAll('.download[data-installed]').forEach(b=>{ delete b.dataset.installed; b.disabled=false; b.textContent='Download'; b.title=''; b.style.background=''; b.style.borderColor=''; b.style.opacity=''; });
          refreshInstalledButtons();
        });
        localList.appendChild(div);
      });
    }
    refreshInstalledButtons();
  }catch{
    localList.innerHTML='<div class="empty mono">Could not load installed models.</div>';
  }
}

function fitClass(lvl){
  const s=String(lvl||'').toLowerCase();
  if(s==='perfect') return 'perfect';
  if(s==='good'||s==='marginal'||s==='runnable') return 'runnable';
  return 'other';
}

function cardTemplate(m){
  const lvl=fitClass(m.fit_level);
  const pct = lvl==='perfect' ? 92 : lvl==='runnable' ? 62 : 28;
  const sources = (m.gguf_sources||[]).map(g=>g.repo).join(', ') || '';
  const repo = m.hf_repo || (m.gguf_sources?.[0]?.repo) || m.name || '';
  const quant = m.quant || m.best_quant || '';
  const tps = m.tps ? `${m.tps} tok/s` : '—';
  const vram = m.vram_gb ? `${m.vram_gb} GB` : '—';
  const disk = m.disk_size_gb ? `${m.disk_size_gb} GB` : vram;
  const name = esc(m.name||'');
  const provider = esc(m.provider||'');
  const already = isRepoInstalled(repo, quant);
  const btnLabel = already ? '✓ Installed' : 'Download';
  const btnDisabled = already ? ' disabled' : '';
  const btnInstalled = already ? ' data-installed="1"' : '';
  const btnTitle = already ? ' title="Already installed"' : '';
  return `
  <article class="card" tabindex="0" aria-label="${name}">
    <div class="card-tape" aria-hidden="true"><div class="tape-fit ${lvl}">${lvl}</div></div>
    <div class="card-body">
      <div class="card-head">
        <div><div class="card-title">${name}</div><div class="card-provider">${provider} • ${esc(quant)} • ${esc(m.run_mode||'')}</div></div>
        <div class="card-badges"><span class="badge ${lvl}">${esc(m.fit_level||'')}</span><span class="badge gpu">${esc(m.run_mode||'GPU')}</span></div>
      </div>
      <div class="card-stats">
        <div class="stat"><span>Disk</span><strong>${esc(disk)}</strong></div>
        <div class="stat"><span>VRAM need</span><strong>${esc(vram)}</strong></div>
        <div class="stat"><span>Est. speed</span><strong>${esc(tps)}</strong></div>
      </div>
      <div class="fit-bar" aria-hidden="true"><i class="${lvl}" style="width:${pct}%"></i></div>
      <div class="card-foot">
        <small title="${esc(sources)}">${repo ? esc(repo) : 'repo via llmfit'}</small>
        <button class="download" type="button" data-repo="${esc(repo)}" data-quant="${esc(quant)}"${btnDisabled}${btnInstalled}${btnTitle}>${btnLabel}</button>
      </div>
      <span class="stub">tear to install →</span>
    </div>
  </article>`;
}

function applyClientFilters(models){
  const q=(searchEl.value||'').trim().toLowerCase();
  if(!q) return models;
  return models.filter(m=> (m.name+' '+m.provider+' '+m.quant).toLowerCase().includes(q));
}

async function fetchCatalog(){
  grid.innerHTML='<div class="skeleton mono">Probing llmfit catalog…</div>';
  emptyEl.hidden=true;
  statusEl.textContent='loading…';
  const providers = getSelectedProviders();
  const body = {
    limit: parseInt(limitEl.value,10) || 20,
    sort: sortEl.value,
    perfect_only: !!perfectOnlyEl.checked,
    include_community: !!includeCommunityEl.checked,
    providers: providers,
  };
  try{
    const r=await fetch(`${API_BASE}/api/catalog`,{
      method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify(body)
    });
    if(!r.ok) throw new Error(`HTTP ${r.status}`);
    const j=await r.json();
    lastData=j;
    const models=j.models||[];
    const filtered=applyClientFilters(models);
    countEl.textContent=`${filtered.length} of ${models.length} models • ${j.meta?.total ?? models.length} total`;
    pathEl.textContent=j.meta?.cmd ? '' : '';
    statusEl.textContent='';
    if(!filtered.length){
      grid.innerHTML='';
      emptyEl.hidden=false;
      return;
    }
    grid.innerHTML=filtered.map(cardTemplate).join('');
    // animate fit bars after paint
    requestAnimationFrame(()=> {
      grid.querySelectorAll('.fit-bar i').forEach(el=>{
        const w=el.style.width;
        el.style.width='0';
        requestAnimationFrame(()=> el.style.width=w);
      });
    });
    // bind downloads — non-blocking: POST 202 returns job_id, dock polls progress
    grid.querySelectorAll('.download').forEach(btn=>{
      btn.addEventListener('click', async ()=>{
        const repo=btn.dataset.repo;
        const quant=btn.dataset.quant;
        if(!repo){ toast('No repo for this model — try search','warning'); return; }
        if(btn.disabled) return;
        btn.disabled=true;
        btn.dataset._busy='1';
        const prev=btn.textContent;
        btn.textContent='Queued…';
        try{
          const resp=await fetch(`${API_BASE}/api/models/download`,{
            method:'POST',
            headers:{'content-type':'application/json'},
            body: JSON.stringify({repo_id: repo, quant: quant || undefined})
          });
          const data=await resp.json().catch(()=>({}));
          if(resp.status===202 && data.job_id){
            const job = data.job || {id:data.job_id, repo_id:repo, quant, status:'queued', progress:0, created_at: new Date().toISOString()};
            dockJobs.set(job.id, job);
            ensureDockPolling();
            renderDock();
            showDock();
            toast(`Queued download: ${repo} ${quant ? '('+quant+')' : ''}`, 'info');
          } else if(resp.status===409) {
            const isInstalled = (data.detail||'').includes('already installed');
            throw new Error(data.detail || (isInstalled ? 'Model already installed' : 'A download is already in progress — please wait'));
          } else if(!resp.ok) {
            throw new Error(data.detail || data['Exception occured'] || `HTTP ${resp.status}`);
          } else {
            toast(`Saved to ${data.path || repo}`, 'success');
            fetchLocal();
            btn.disabled=false;
            btn.textContent=prev;
            delete btn.dataset._busy;
          }
        }catch(e){
          const msg = e.message || 'Download failed';
          const type = msg.includes('already in progress') ? 'warning' : msg.includes('already installed') ? 'warning' : 'error';
          if(msg.includes('already installed')){
            btn.textContent='✓ Installed';
            btn.disabled=true;
            btn.dataset.installed='1';
            btn.style.background='var(--lab-moss)';
            btn.style.borderColor='var(--lab-moss)';
          } else {
            btn.disabled=false;
            btn.textContent=prev;
            delete btn.dataset._busy;
          }
          toast(msg, type);
        }
      });
    });
  }catch(e){
    grid.innerHTML=`<div class="empty mono">Could not load catalog. ${esc(e.message||'')}</div>`;
    statusEl.textContent='error';
  }
}

// events
$('#applyFilters').addEventListener('click', fetchCatalog);
$('#resetFilters').addEventListener('click', ()=>{
  searchEl.value='';
  perfectOnlyEl.checked=true;
  includeCommunityEl.checked=false;
  limitEl.value='20';
  sortEl.value='score';
  resetProviders();
  fetchCatalog();
});
searchEl.addEventListener('keydown', e=>{ if(e.key==='Enter') fetchCatalog(); });
$('#refreshLocal').addEventListener('click', fetchLocal);
providerBtn.addEventListener('click', ()=> toggleProviderMenu());
document.addEventListener('click', (e)=>{
  if(!providersWrap.contains(e.target)) toggleProviderMenu(false);
});
providerMenu.addEventListener('click', e=> e.stopPropagation());

renderProviderMenu();
fetchSystem();
fetchLocal();
fetchCatalog();

// restore existing downloads on load (if any) and start polling
(async ()=>{
  try{
    const r = await fetch(`${API_BASE}/api/downloads`);
    if(r.ok){
      const d = await r.json();
      (d.jobs||[]).forEach(j=> dockJobs.set(j.id, j));
      if(dockJobs.size>0){ ensureDockPolling(); renderDock(); }
    }
  }catch{}
})();
