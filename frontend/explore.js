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
const chipsEl = $('#providerChips');
const searchEl = $('#search');
const limitEl = $('#limit');
const sortEl = $('#sort');
const perfectOnlyEl = $('#perfectOnly');
const includeCommunityEl = $('#includeCommunity');
const toastEl = $('#toast');

const PROVIDERS = ['meta','alibaba','google','mistral','microsoft','deepseek'];
let activeProviders = new Set();
let lastData = null;

function toast(msg){
  toastEl.textContent = msg;
  toastEl.hidden = false;
  clearTimeout(toastEl._t);
  toastEl._t = setTimeout(()=> toastEl.hidden = true, 2600);
}
function esc(s){ return s.replace(/[&<>"']/g, m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }

function renderProviderChips(){
  chipsEl.innerHTML='';
  PROVIDERS.forEach(p=>{
    const btn=document.createElement('button');
    btn.type='button';
    btn.className='chip' + (activeProviders.has(p) ? ' active' : '');
    btn.textContent=p;
    btn.setAttribute('aria-pressed', activeProviders.has(p) ? 'true':'false');
    btn.addEventListener('click', ()=>{
      if(activeProviders.has(p)) activeProviders.delete(p); else activeProviders.add(p);
      renderProviderChips();
    });
    chipsEl.appendChild(btn);
  });
}

async function fetchSystem(){
  try{
    const r=await fetch(`${API_BASE}/api/system`);
    if(!r.ok) throw new Error();
    const j=await r.json();
    const sys=j.system || j.raw?.system || {};
    // try common llmfit keys
    const ram = sys.memory_available_gb ?? sys.ram_gb ?? sys.total_ram_gb ?? j.raw?.memory_available_gb ?? 0;
    const vram = sys.vram_gb ?? sys.gpu_vram_gb ?? sys.gpu?.vram_gb ?? 0;
    const cores = sys.cpu_cores ?? sys.cores ?? sys.cpu?.cores ?? '—';
    const gpu = sys.gpu_name ?? sys.gpu?.name ?? sys.gpus?.[0]?.name ?? 'none';
    const mode = sys.run_mode ?? sys.gpu?.run_mode ?? 'CPU';
    hwDot.classList.remove('off');
    hwStatus.textContent = 'hardware ready';
    kpiRam.textContent = ram ? `${Number(ram).toFixed(1)} GB` : '—';
    kpiVram.textContent = vram ? `${Number(vram).toFixed(1)} GB` : 'UMA / CPU';
    kpiCores.textContent = String(cores);
    kpiGpu.textContent = String(gpu).slice(0,22);
    kpiMode.textContent = String(mode);
    // bars: ram fill as 8GB baseline vs available
    const ramPct = Math.min(100, (Number(ram)||8)/16*100);
    ramFill.style.width = ramPct + '%';
    ramFill.className = ramPct > 85 ? 'over' : '';
    const vramPct = vram ? Math.min(100, Number(vram)/16*100) : 35;
    vramFill.style.width = vramPct + '%';
    kpiRamMeta.textContent = `available for model`;
    kpiVramMeta.textContent = vram ? `VRAM for offload` : `shared memory`;
  }catch{
    hwDot.classList.add('off');
    hwStatus.textContent = 'offline — llmfit not found';
  }
}

async function fetchLocal(){
  try{
    const r=await fetch(`${API_BASE}/api/models`);
    const j=await r.json();
    const models=j.models||[];
    cachePath.textContent=j.path||'~/.yourstrulyai/models';
    cacheCount.textContent=`${models.length} GGUF`;
    if(!models.length){
      localList.innerHTML='<div class="empty mono">No GGUF yet — download a Perfect fit to start.</div>';
      return;
    }
    localList.innerHTML='';
    models.forEach(name=>{
      const div=document.createElement('div');
      div.className='local-item';
      div.innerHTML=`<span>${esc(name)}</span><button type="button" data-file="${esc(name)}">Delete</button>`;
      div.querySelector('button').addEventListener('click', async ()=>{
        if(!confirm(`Delete ${name}?`)) return;
        const del=await fetch(`${API_BASE}/api/models/${encodeURIComponent(name)}`,{method:'DELETE'});
        const dj=await del.json().catch(()=>({}));
        if(!del.ok) toast(dj.detail||'Delete failed');
        else toast('Deleted ' + name);
        fetchLocal();
      });
      localList.appendChild(div);
    });
  }catch{
    localList.innerHTML='<div class="empty mono">Could not load local cache.</div>';
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
  const repo = m.hf_repo || (m.gguf_sources?.[0]?.repo) || '';
  const quant = m.quant || m.best_quant || '';
  const tps = m.tps ? `${m.tps} tok/s` : '—';
  const vram = m.vram_gb ? `${m.vram_gb} GB` : '—';
  const disk = m.disk_size_gb ? `${m.disk_size_gb} GB` : vram;
  const name = esc(m.name||'');
  const provider = esc(m.provider||'');
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
        <button class="download" type="button" data-repo="${esc(repo)}" data-file="" data-quant="${esc(quant)}">Download</button>
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
  const params=new URLSearchParams();
  params.set('limit', limitEl.value);
  params.set('sort', sortEl.value);
  if(perfectOnlyEl.checked) params.set('perfect_only','true');
  if(includeCommunityEl.checked) params.set('include_community','true');
  if(activeProviders.size) params.set('providers', [...activeProviders].join(','));
  else params.set('providers','');
  // if no providers and not includeCommunity, backend will use trusted set; we send empty to force trusted
  try{
    const r=await fetch(`${API_BASE}/api/catalog?${params.toString()}`);
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
    // bind downloads
    grid.querySelectorAll('.download').forEach(btn=>{
      btn.addEventListener('click', async ()=>{
        const repo=btn.dataset.repo;
        const quant=btn.dataset.quant;
        if(!repo){ toast('No repo for this model — try search'); return; }
        btn.disabled=true;
        const prev=btn.textContent;
        btn.textContent='Downloading…';
        try{
          const resp=await fetch(`${API_BASE}/api/models/download`,{
            method:'POST',
            headers:{'content-type':'application/json'},
            body: JSON.stringify({repo_id: repo, quant: quant || undefined})
          });
          const data=await resp.json().catch(()=>({}));
          if(!resp.ok) throw new Error(data.detail || data['Exception occured'] || `HTTP ${resp.status}`);
          toast(`Saved to ${data.path || repo}`);
          fetchLocal();
        }catch(e){
          toast(e.message || 'Download failed');
        }finally{
          btn.disabled=false;
          btn.textContent=prev;
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
  perfectOnlyEl.checked=false;
  includeCommunityEl.checked=false;
  limitEl.value='20';
  sortEl.value='score';
  activeProviders.clear();
  renderProviderChips();
  fetchCatalog();
});
searchEl.addEventListener('keydown', e=>{ if(e.key==='Enter') fetchCatalog(); });
$('#refreshLocal').addEventListener('click', fetchLocal);

renderProviderChips();
fetchSystem();
fetchLocal();
fetchCatalog();
