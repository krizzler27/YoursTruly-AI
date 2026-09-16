import { API_BASE } from './js/shared/config.js';
import { $, escapeHtml as esc } from './js/shared/utils.js';

const kpiRam = $('#kpiRam');
const kpiVram = $('#kpiVram');
const kpiCores = $('#kpiCores');
const kpiGpu = $('#kpiGpu');
const kpiMode = $('#kpiMode');
const ramFill = $('#ramFill');
const vramFill = $('#vramFill');
const kpiRamMeta = $('#kpiRamMeta');
const kpiVramMeta = $('#kpiVramMeta');
const kpiCpuName = $('#kpiCpuName');
const kpiTotalRam = $('#kpiTotalRam');
const totalRamFill = $('#totalRamFill');
const kpiTotalRamMeta = $('#kpiTotalRamMeta');
const hwExtra = $('#hwExtra');
const hardwareLoading = $('#hardwareLoading');
const hwStack = $('#hwStack');
const cachePath = $('#cachePath');
const cacheCount = $('#cacheCount');
const cacheTotalSize = $('#cacheTotalSize');
const copyPathBtn = $('#copyPath');
const localList = $('#localList');
const installedLoading = $('#installedLoading');
const catalogLoading = $('#catalogLoading');
const deleteModal = $('#deleteModal');
const deleteModalName = $('#deleteModalName');
const deleteCancel = $('#deleteCancel');
const deleteConfirm = $('#deleteConfirm');

const grid = $('#catalogGrid');
const countEl = $('#catalogCount');
const pathEl = $('#catalogPath');
const statusEl = $('#catalogStatus');
const emptyEl = $('#emptyCatalog');
const providerBtn = $('#providerBtn');
const providerBtnLabel = $('#providerBtnLabel');
const providerMenu = $('#providerMenu');
const providersWrap = $('#providersWrap');
const searchEl = $('#search');
const limitEl = $('#limit');
const sortEl = $('#sort');
const loadMoreWrap = $('#loadMoreWrap');
const loadMoreBtn = $('#loadMore');
const loadMoreInfo = $('#loadMoreInfo');
const perfectOnlyEl = $('#perfectOnly');
const includeCommunityEl = $('#includeCommunity');
const toastStack = $('#toastStack');
const downloadDock = $('#downloadDock');
const dockList = $('#dockList');
const dockFab = $('#dockFab');
const dockClose = $('#dockClose');

const tabBtns = document.querySelectorAll('.tab-btn');
const panels = {
  hardware: document.getElementById('panel-hardware'),
  explore: document.getElementById('panel-explore'),
  installed: document.getElementById('panel-installed'),
};

const PROVIDERS = ['meta','alibaba','google','mistral','microsoft','deepseek'];
let selectedProviders = new Set(PROVIDERS);
let lastData = null;
let displayed = 20;
let currentFiltered = [];

// --- Tabs: Hardware | Explore (center) | Installed ---
function switchTab(name, pushHash=true){
  const n = panels[name] ? name : 'explore';
  tabBtns.forEach(btn=>{
    const isActive = btn.dataset.tab === n;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', String(isActive));
  });
  Object.entries(panels).forEach(([k, el])=>{
    if(!el) return;
    el.hidden = k !== n;
  });
  if(n === 'hardware') fetchSystem();
  if(pushHash) {
    const hash = n === 'explore' ? '#explore' : `#${n}`;
    if(location.hash !== hash) history.replaceState(null, '', hash);
  }
}
tabBtns.forEach(btn=> btn.addEventListener('click', ()=> switchTab(btn.dataset.tab)));
window.addEventListener('hashchange', ()=>{
  const h = location.hash.replace('#','');
  if(panels[h]) switchTab(h, false);
});
// initial: hash or explore (center default)
{
  const h = location.hash.replace('#','');
  switchTab(panels[h] ? h : 'explore', false);
}

// --- Toast ---
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
function debounce(fn, ms){ let t; return (...a)=>{ clearTimeout(t); t=setTimeout(()=>fn(...a), ms); }; }

function formatProvidersLabel(set){
  if(!set || set.size===0) return 'None';
  if(set.size===PROVIDERS.length) return 'All';
  const csv = [...set].join(', ');
  if(csv.length > 14) return csv.slice(0,10).trimEnd() + '…';
  return csv;
}
function renderProviderMenu(){
  providerMenu.innerHTML='';
  const actions = document.createElement('div');
  actions.className = 'provider-actions';
  const allBtn = document.createElement('button');
  allBtn.type='button'; allBtn.className='ghost small mono'; allBtn.textContent='All';
  allBtn.addEventListener('click', (e)=>{ e.stopPropagation(); selectedProviders=new Set(PROVIDERS); renderProviderMenu(); });
  const noneBtn = document.createElement('button');
  noneBtn.type='button'; noneBtn.className='ghost small mono'; noneBtn.textContent='None';
  noneBtn.addEventListener('click', (e)=>{ e.stopPropagation(); selectedProviders=new Set(); renderProviderMenu(); });
  actions.append(allBtn, noneBtn);
  providerMenu.appendChild(actions);
  const divider = document.createElement('div');
  divider.style.cssText='height:1px;background:var(--lab-border);margin:6px 0';
  providerMenu.appendChild(divider);
  PROVIDERS.forEach(p=>{
    const label=document.createElement('label');
    label.className='provider-option mono';
    const cb=document.createElement('input');
    cb.type='checkbox';
    cb.value=p;
    cb.checked=selectedProviders.has(p);
    cb.addEventListener('change', ()=>{
      if(cb.checked) selectedProviders.add(p); else selectedProviders.delete(p);
      const lbl = formatProvidersLabel(selectedProviders);
      if(providerBtnLabel) providerBtnLabel.textContent = lbl; else providerBtn.textContent = lbl;
      providerBtn.title=[...selectedProviders].join(', ') || 'None';
      if(providersWrap) providersWrap.title = providerBtn.title;
    });
    const span=document.createElement('span');
    span.textContent=p;
    label.append(cb, span);
    providerMenu.appendChild(label);
  });
  const lbl = formatProvidersLabel(selectedProviders);
  if(providerBtnLabel) providerBtnLabel.textContent = lbl; else providerBtn.textContent = lbl;
  providerBtn.title=[...selectedProviders].join(', ') || 'None';
  if(providersWrap) providersWrap.title = providerBtn.title;
}
function getSelectedProviders(){ return [...selectedProviders]; }
function resetProviders(){ selectedProviders=new Set(PROVIDERS); renderProviderMenu(); }
function toggleProviderMenu(force){
  const willOpen = typeof force==='boolean' ? force : providerMenu.hidden;
  providerMenu.hidden=!willOpen;
  providerBtn.setAttribute('aria-expanded', String(willOpen));
  providersWrap.classList.toggle('open', willOpen);
}

// --- Download dock ---
let dockJobs = new Map();
let dockPollTimer = null;
let dockGraceTimer = null;
let dockMinimized = false;
function stopDockPolling(){ if(dockPollTimer){ clearInterval(dockPollTimer); dockPollTimer=null; } }
function clearGrace(){ if(dockGraceTimer){ clearTimeout(dockGraceTimer); dockGraceTimer=null; } }
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
          if(j.status==='failed') toast(`Download failed: ${j.repo_id} - ${j.error ? j.error.slice(0,120) : 'see logs'}`, 'error');
        }
      });
      renderDock();
      const stillActive = jobs.some(j=> j.status==='queued' || j.status==='downloading');
      if(!stillActive){
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
        const r2 = await fetch(`${API_BASE}/api/downloads`);
        if(r2.ok){ const dd=await r2.json(); dd.jobs.forEach(j=>dockJobs.set(j.id,j)); renderDock(); }
      }catch(e){ toast(e.message||'Cancel failed','error'); }
      finally{ btn.disabled=false; }
    });
  });
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
      if(btn.dataset._busy) {
        btn.disabled=false;
        btn.textContent='Download';
        delete btn.dataset._busy;
      }
    }
  });
}
function showDock(){ dockMinimized=false; renderDock(); }
if(dockClose) dockClose.addEventListener('click', ()=>{
  clearGrace(); stopDockPolling();
  dockMinimized=true;
  downloadDock.hidden=true;
  dockFab.hidden = dockJobs.size===0;
});
if(dockFab) dockFab.addEventListener('click', showDock);

async function fetchSystem(){
  // cached 60s to avoid refetch on tab switch; hardware changes slowly
  const hwKey = 'yt_hw_cache';
  try{
    const c = JSON.parse(localStorage.getItem(hwKey)||'null');
    if(c && c.ts && Date.now() - c.ts < 60*1000 && c.data){
      const j=c.data;
      const sys=j.system || j.raw?.system || {};
      const ramAvail = sys.available_ram_gb ?? sys.memory_available_gb ?? 0;
      const ramTotal = sys.total_ram_gb ?? 0;
      const vram = sys.gpu_vram_gb ?? sys.vram_gb ?? sys.gpus?.[0]?.vram_gb ?? 0;
      const cores = sys.cpu_cores ?? '-';
      const cpuName = sys.cpu_name ?? '-';
      const gpu = sys.gpu_name ?? sys.gpus?.[0]?.name ?? '-';
      if(kpiRam) kpiRam.textContent = ramAvail ? `${Number(ramAvail).toFixed(1)} GB` : '-';
      if(kpiVram) kpiVram.textContent = vram ? `${Number(vram).toFixed(1)} GB` : '-';
      if(kpiCores) kpiCores.textContent = String(cores);
      if(kpiGpu) kpiGpu.textContent = String(gpu).slice(0,28);
      if(kpiCpuName) kpiCpuName.textContent = String(cpuName);
      if(kpiTotalRam) kpiTotalRam.textContent = ramTotal ? `${Number(ramTotal).toFixed(1)} GB` : '-';
      const ramBase = Number(ramTotal) || 16;
      const ramPct = Math.min(100, (Number(ramAvail)||8)/ramBase*100);
      if(ramFill){ ramFill.style.width = ramPct + '%'; ramFill.className = ramPct > 85 ? 'over' : ''; }
      if(totalRamFill){ totalRamFill.style.width = '100%'; }
      if(kpiTotalRamMeta) kpiTotalRamMeta.textContent = ramTotal ? `${Number(ramTotal).toFixed(1)} GB installed` : 'installed';
      const vramPct = vram ? Math.min(100, Number(vram)/ramBase*100) : 35;
      if(vramFill) vramFill.style.width = vramPct + '%';
      if(kpiRamMeta) kpiRamMeta.textContent = ramTotal ? `${Number(ramAvail).toFixed(1)} free / ${Number(ramTotal).toFixed(1)} total` : `available`;
      if(kpiVramMeta) kpiVramMeta.textContent = vram ? `${vram} GB VRAM` : `CPU only`;
      if(hwExtra){
        hwExtra.innerHTML = '';
        const chips = [];
        if(sys.gpu_available_gb != null) chips.push(`GPU free ${sys.gpu_available_gb} GB`);
        if(sys.memory_bandwidth_gbps || sys.gpus?.[0]?.memory_bandwidth_gbps) chips.push(`BW ${sys.memory_bandwidth_gbps ?? sys.gpus[0].memory_bandwidth_gbps} GB/s`);
        chips.forEach(c=>{ const s=document.createElement('span'); s.textContent=c; hwExtra.appendChild(s); });
        if(!chips.length) hwExtra.textContent = '';
      }
      if(hardwareLoading) hardwareLoading.hidden = true;
      if(hwStack) hwStack.hidden = false;
      return;
    }
  }catch{}
  if(hardwareLoading) hardwareLoading.hidden = false;
  if(hwStack) hwStack.hidden = true;
  try{
    const r=await fetch(`${API_BASE}/api/system`);
    if(!r.ok) throw new Error();
    const j=await r.json();
    try{ localStorage.setItem(hwKey, JSON.stringify({ts: Date.now(), data: j})); }catch{}
    const sys=j.system || j.raw?.system || {};
    const ramAvail = sys.available_ram_gb ?? sys.memory_available_gb ?? 0;
    const ramTotal = sys.total_ram_gb ?? 0;
    const vram = sys.gpu_vram_gb ?? sys.vram_gb ?? sys.gpus?.[0]?.vram_gb ?? 0;
    const cores = sys.cpu_cores ?? '-';
    const cpuName = sys.cpu_name ?? '-';
    const gpu = sys.gpu_name ?? sys.gpus?.[0]?.name ?? '-';
    if(kpiRam) kpiRam.textContent = ramAvail ? `${Number(ramAvail).toFixed(1)} GB` : '-';
    if(kpiVram) kpiVram.textContent = vram ? `${Number(vram).toFixed(1)} GB` : '-';
    if(kpiCores) kpiCores.textContent = String(cores);
    if(kpiGpu) kpiGpu.textContent = String(gpu).slice(0,28);
    if(kpiCpuName) kpiCpuName.textContent = String(cpuName);
    if(kpiTotalRam) kpiTotalRam.textContent = ramTotal ? `${Number(ramTotal).toFixed(1)} GB` : '-';
    const ramBase = Number(ramTotal) || 16;
    const ramPct = Math.min(100, (Number(ramAvail)||8)/ramBase*100);
    if(ramFill){ ramFill.style.width = ramPct + '%'; ramFill.className = ramPct > 85 ? 'over' : ''; }
    if(totalRamFill){ totalRamFill.style.width = '100%'; }
    if(kpiTotalRamMeta) kpiTotalRamMeta.textContent = ramTotal ? `${Number(ramTotal).toFixed(1)} GB installed` : 'installed';
    const vramPct = vram ? Math.min(100, Number(vram)/ramBase*100) : 35;
    if(vramFill) vramFill.style.width = vramPct + '%';
    if(kpiRamMeta) kpiRamMeta.textContent = ramTotal ? `${Number(ramAvail).toFixed(1)} free / ${Number(ramTotal).toFixed(1)} total` : `available`;
    if(kpiVramMeta) kpiVramMeta.textContent = vram ? `${vram} GB VRAM` : `CPU only`;
    // extra chips for remaining meaningful fields (gpu count now in detail grid)
    if(hwExtra){
      hwExtra.innerHTML = '';
      const chips = [];
      if(sys.gpu_available_gb != null) chips.push(`GPU free ${sys.gpu_available_gb} GB`);
      if(sys.memory_bandwidth_gbps || sys.gpus?.[0]?.memory_bandwidth_gbps) chips.push(`BW ${sys.memory_bandwidth_gbps ?? sys.gpus[0].memory_bandwidth_gbps} GB/s`);
      chips.forEach(c=>{
        const s=document.createElement('span');
        s.textContent=c;
        hwExtra.appendChild(s);
      });
      if(!chips.length) hwExtra.textContent = '';
    }
    if(hardwareLoading) hardwareLoading.hidden = true;
    if(hwStack) hwStack.hidden = false;
  }catch{
    if(kpiCpuName) kpiCpuName.textContent='-';
    if(hardwareLoading) hardwareLoading.hidden = true;
    if(hwStack) hwStack.hidden = false;
  }
}

let installedSet = new Set();
let installedDetails = [];
let pendingDeleteFile = null;
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
    if(btn.dataset.installed==='1') return;
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
  if(localList && !localList.querySelector('.loading-state')){
    localList.innerHTML = '<div class="loading-state mono"><div class="spinner small"></div><span>Checking installed models…</span></div>';
  }
  try{
    const r=await fetch(`${API_BASE}/api/models`);
    const j=await r.json();
    const rawModels=j.models||[];
    // handle kv {name,path} from new API or old string array
    const models=rawModels.map(m=> typeof m==='string' ? m : (m.name || m.path || String(m)));
    const details=j.details||[];
    installedSet = new Set(models.map(m=>String(m).toLowerCase()));
    installedDetails = details;
    if(cachePath) cachePath.textContent=j.path||'~/.yourstrulyai/models';
    const countText = `${models.length} model${models.length===1?'':'s'}`;
    if(cacheCount) cacheCount.textContent=countText;
    if(cacheTotalSize){
      const total = details.reduce((s,d)=> s + (Number(d.size_gb)||0), 0);
      cacheTotalSize.textContent = models.length ? `• ${total.toFixed(2)} GB total` : '';
    }
    if(!models.length){
      localList.innerHTML='<div class="empty mono">No models installed - go to Explore and Download a Perfect fit.</div>';
    } else {
      const sizeMap = new Map(details.map(d=>[d.name, d.size_gb]));
      const mtimeMap = new Map(details.map(d=>[d.name, d.modified]));
      localList.innerHTML='';
      models.forEach(name=>{
        const size = sizeMap.get(name);
        const div=document.createElement('div');
        div.className='local-item';
        const sizeLabel = size != null ? `${size} GB` : '';
        const dateLabel = mtimeMap.get(name) ? new Date(mtimeMap.get(name)*1000).toLocaleDateString() : '';
        const meta = [sizeLabel, dateLabel].filter(Boolean).join(' • ');
        div.innerHTML=`<div class="local-item-main"><span class="local-item-name">${esc(name)}</span><span class="local-item-meta">${esc(meta)}</span></div><button type="button" data-file="${esc(name)}">Delete</button>`;
        div.querySelector('button').addEventListener('click', ()=>{
          pendingDeleteFile = name;
          if(deleteModalName) deleteModalName.textContent = name;
          if(deleteModal && typeof deleteModal.showModal === 'function') deleteModal.showModal();
          else if(deleteModal) deleteModal.setAttribute('open','');
        });
        localList.appendChild(div);
      });
    }
    refreshInstalledButtons();
  }catch{
    localList.innerHTML='<div class="empty mono">Could not load installed models.</div>';
  }
}
async function handleDeleteConfirm(){
  const name = pendingDeleteFile;
  if(!name) return;
  try{
    const del=await fetch(`${API_BASE}/api/models/${encodeURIComponent(name)}`,{method:'DELETE'});
    const dj=await del.json().catch(()=>({}));
    if(!del.ok) toast(dj.detail||'Delete failed','error');
    else { toast('Deleted ' + name,'success'); }
    await fetchLocal();
    document.querySelectorAll('.download[data-installed]').forEach(b=>{ delete b.dataset.installed; b.disabled=false; b.textContent='Download'; b.title=''; b.style.background=''; b.style.borderColor=''; b.style.opacity=''; });
    refreshInstalledButtons();
  } finally {
    pendingDeleteFile = null;
    if(deleteModal && typeof deleteModal.close === 'function') try{ deleteModal.close(); }catch{}
    if(deleteModal) deleteModal.removeAttribute('open');
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
  const sources = (m.gguf_sources||[]).map(g=>g.repo).join(', ') || '';
  const repo = m.hf_repo || (m.gguf_sources?.[0]?.repo) || m.name || '';
  const quant = m.quant || m.best_quant || '';
  const tps = m.tps ? `${m.tps} tok/s` : '-';
  const vram = m.vram_gb ? `${m.vram_gb} GB` : '-';
  const disk = m.disk_size_gb ? `${m.disk_size_gb} GB` : '-';
  const name = esc(m.name||'');
  const provider = esc(m.provider||'');
  const already = isRepoInstalled(repo, quant);
  const btnLabel = already ? '✓ Installed' : 'Download';
  const btnDisabled = already ? ' disabled' : '';
  const btnInstalled = already ? ' data-installed="1"' : '';
  const btnTitle = already ? ' title="Already installed"' : '';
  return `
  <article class="card ${lvl}" tabindex="0" aria-label="${name}">
      <div class="card-head">
        <div><div class="card-title">${name}</div><div class="card-provider">${provider} • ${esc(quant)}</div></div>
        <span class="badge ${lvl}">${esc(m.fit_level||'')}</span>
      </div>
      <div class="card-stats">
        <span><strong>${esc(disk)}</strong> disk</span>
        <span><strong>${esc(vram)}</strong> VRAM</span>
        <span><strong>${esc(tps)}</strong></span>
      </div>
      <div class="card-foot">
        <small title="${esc(sources)}">${repo ? esc(repo) : 'repo via llmfit'}</small>
        <button class="download" type="button" data-repo="${esc(repo)}" data-quant="${esc(quant)}"${btnDisabled}${btnInstalled}${btnTitle}>${btnLabel}</button>
      </div>
  </article>`;
}

function applyClientFilters(models){
  const q=(searchEl.value||'').trim().toLowerCase();
  if(!q) return models;
  return models.filter(m=> (m.name+' '+m.provider+' '+m.quant).toLowerCase().includes(q));
}
function updateLoadMore(){
  if(!loadMoreWrap) return;
  if(!currentFiltered.length || displayed >= currentFiltered.length){
    loadMoreWrap.hidden = true;
    return;
  }
  loadMoreWrap.hidden = false;
  if(loadMoreInfo) loadMoreInfo.textContent = `Showing ${Math.min(displayed, currentFiltered.length)} of ${currentFiltered.length}`;
}
function renderCatalogFromLast(){
  if(!lastData) return;
  const models=lastData.models||[];
  const filtered=applyClientFilters(models);
  currentFiltered = filtered;
  if(countEl) countEl.textContent=`${filtered.length} models • ${lastData.meta?.total ?? models.length} total`;
  if(!filtered.length){
    grid.innerHTML='';
    emptyEl.hidden=false;
    if(loadMoreWrap) loadMoreWrap.hidden=true;
    return;
  }
  emptyEl.hidden=true;
  const slice = filtered.slice(0, displayed);
  grid.innerHTML=slice.map(cardTemplate).join('');
  bindDownloadButtons();
  refreshInstalledButtons();
  const jobs = [...dockJobs.values()].filter(j=> j.status==='queued'||j.status==='downloading');
  if(jobs.length) updateCardButtons(jobs);
  updateLoadMore();
}
function handleLoadMore(){
  displayed += 10;
  const slice = currentFiltered.slice(0, displayed);
  grid.innerHTML=slice.map(cardTemplate).join('');
  bindDownloadButtons();
  refreshInstalledButtons();
  const jobs = [...dockJobs.values()].filter(j=> j.status==='queued'||j.status==='downloading');
  if(jobs.length) updateCardButtons(jobs);
  updateLoadMore();
}
function bindDownloadButtons(){
  grid.querySelectorAll('.download').forEach(btn=>{
    btn.addEventListener('click', async ()=>{
      const repo=btn.dataset.repo;
      const quant=btn.dataset.quant;
      if(!repo){ toast('No repo for this model - try search','warning'); return; }
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
          throw new Error(data.detail || (isInstalled ? 'Model already installed' : 'A download is already in progress - please wait'));
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
}
function isHardRefresh(){
  try{
    const nav = performance.getEntriesByType('navigation')[0];
    if(nav) return nav.type === 'reload';
  }catch{}
  try{ return performance.navigation && performance.navigation.type === 1; }catch{ return false; }
}
function catalogCacheKey(body){
  return `yt_catalog_${body.sort}_${body.perfect_only}_${body.include_community}_${(body.providers||[]).join(',')}_${body.limit}`;
}
async function fetchCatalog(){
  grid.innerHTML='<div class="loading-state mono"><div class="spinner"></div><span>Loading models…</span></div>';
  emptyEl.hidden=true;
  if(statusEl) statusEl.textContent='';
  if(loadMoreWrap) loadMoreWrap.hidden=true;
  displayed = 20;
  let providers = getSelectedProviders();
  let includeCommunity = !!includeCommunityEl.checked;
  if(providers.length===0){
    providers = null;
    includeCommunity = true;
  }
  const body = {
    limit: 100,
    sort: sortEl.value,
    perfect_only: !!perfectOnlyEl.checked,
    include_community: includeCommunity,
    providers: providers,
  };
  const cacheKey = catalogCacheKey(body);
  try{
    const cached = JSON.parse(localStorage.getItem(cacheKey)||'null');
    if(cached && cached.ts && Date.now() - cached.ts < 30*60*1000 && cached.data){
      lastData=cached.data;
      const models=lastData.models||[];
      const filtered=applyClientFilters(models);
      currentFiltered = filtered;
      if(countEl) countEl.textContent=`${filtered.length} models • ${lastData.meta?.total ?? models.length} total (cached)`;
      if(pathEl) pathEl.textContent='';
      if(statusEl) statusEl.textContent='cached';
      if(!filtered.length){
        grid.innerHTML='';
        emptyEl.hidden=false;
        if(loadMoreWrap) loadMoreWrap.hidden=true;
        return;
      }
      emptyEl.hidden=true;
      const slice = filtered.slice(0, displayed);
      grid.innerHTML=slice.map(cardTemplate).join('');
      bindDownloadButtons();
      refreshInstalledButtons();
      const jobs = [...dockJobs.values()].filter(j=> j.status==='queued'||j.status==='downloading');
      if(jobs.length) updateCardButtons(jobs);
      updateLoadMore();
      return;
    }
  }catch{}
  try{
    const r=await fetch(`${API_BASE}/api/catalog`,{
      method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify(body)
    });
    if(!r.ok) throw new Error(`HTTP ${r.status}`);
    const j=await r.json();
    lastData=j;
    try{ localStorage.setItem(cacheKey, JSON.stringify({ts: Date.now(), data: j})); }catch{}
    const models=j.models||[];
    const filtered=applyClientFilters(models);
    currentFiltered = filtered;
    if(countEl) countEl.textContent=`${filtered.length} models • ${j.meta?.total ?? models.length} total`;
    if(pathEl) pathEl.textContent='';
    if(statusEl) statusEl.textContent='';
    if(!filtered.length){
      grid.innerHTML='';
      emptyEl.hidden=false;
      if(loadMoreWrap) loadMoreWrap.hidden=true;
      return;
    }
    emptyEl.hidden=true;
    const slice = filtered.slice(0, displayed);
    grid.innerHTML=slice.map(cardTemplate).join('');
    bindDownloadButtons();
    refreshInstalledButtons();
    const jobs = [...dockJobs.values()].filter(j=> j.status==='queued'||j.status==='downloading');
    if(jobs.length) updateCardButtons(jobs);
    updateLoadMore();
  }catch(e){
    grid.innerHTML=`<div class="empty mono">Could not load catalog. ${esc(e.message||'')}</div>`;
    if(statusEl) statusEl.textContent='error';
    if(loadMoreWrap) loadMoreWrap.hidden=true;
  }
}

// events
$('#applyFilters').addEventListener('click', fetchCatalog);
$('#resetFilters').addEventListener('click', ()=>{
  searchEl.value='';
  perfectOnlyEl.checked=true;
  includeCommunityEl.checked=false;
  if(limitEl) limitEl.value='100';
  displayed = 20;
  sortEl.value='score';
  resetProviders();
  fetchCatalog();
});
if(loadMoreBtn) loadMoreBtn.addEventListener('click', handleLoadMore);
const debouncedRender = debounce(()=> renderCatalogFromLast(), 250);
searchEl.addEventListener('input', debouncedRender);
searchEl.addEventListener('keydown', e=>{ if(e.key==='Enter') { e.preventDefault(); fetchCatalog(); }});
$('#refreshLocal').addEventListener('click', fetchLocal);
function handleProvidersToggle(e){
  e.preventDefault();
  e.stopPropagation();
  toggleProviderMenu();
}
providerBtn.addEventListener('click', handleProvidersToggle);
if(providersWrap){
  providersWrap.addEventListener('click', (e)=>{
    if(providerMenu.contains(e.target)) return;
    handleProvidersToggle(e);
  });
}
document.addEventListener('click', (e)=>{
  if(!providersWrap.contains(e.target)) toggleProviderMenu(false);
});
providerMenu.addEventListener('click', e=> e.stopPropagation());
if(copyPathBtn) copyPathBtn.addEventListener('click', async ()=>{
  const txt = cachePath ? cachePath.textContent : '';
  try{
    if(navigator.clipboard && txt) await navigator.clipboard.writeText(txt);
    else if(txt){ const ta=document.createElement('textarea'); ta.value=txt; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove(); }
    toast('Path copied','success',2000);
  }catch{ toast('Copy failed','error'); }
});
if(deleteCancel) deleteCancel.addEventListener('click', ()=>{
  pendingDeleteFile=null;
  if(deleteModal && typeof deleteModal.close==='function') try{ deleteModal.close(); }catch{}
  if(deleteModal) deleteModal.removeAttribute('open');
});
if(deleteConfirm) deleteConfirm.addEventListener('click', (e)=>{ e.preventDefault(); handleDeleteConfirm(); });
if(deleteModal) deleteModal.addEventListener('click', (e)=>{
  if(e.target===deleteModal){ pendingDeleteFile=null; try{ deleteModal.close(); }catch{} deleteModal.removeAttribute('open'); }
});
if(deleteModal) deleteModal.addEventListener('cancel', (e)=>{ e.preventDefault(); pendingDeleteFile=null; try{ deleteModal.close(); }catch{} });

renderProviderMenu();
fetchSystem();
fetchLocal();
fetchCatalog();

// restore existing downloads on load
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
