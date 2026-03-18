'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let state = {};          // tuning params from server
let videos = [];         // video metadata list
let recommendations = {}; // video_id → recommendation obj
let classColors = {};    // class → hex color
let difficultyNames = {};

let currentVideoId = null;
let videoFramePredictions = []; // [{t, scores}]
let videoDuration = 0;
let tuneDebounceTimer = null;
let smoothWindowSec = 3.0;       // backend temporal smoothing window
let autoTunePollTimer = null;
let uiSettingsDebounceTimer = null;

// Pending parameter changes (accumulated before debounced send)
let pendingParams = {};

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  const res = await fetch('/tuning/api/init');
  const data = await res.json();

  videos = data.videos;
  state = data.state;
  classColors = data.class_colors;
  difficultyNames = data.difficulty_names;
  if (data.crop_rect) cropRect = data.crop_rect;
  cropOverrides = data.crop_overrides || {};

  // Store recommendations by video_id
  for (const r of data.recommendations) {
    recommendations[r.video_id] = r;
  }

  applyStateToSliders(state);
  renderVideoGrid();
  updateScoreDisplay(data.n_match, data.total, data.recommendations);
  updateWeightBar(state);

  // Load model selector
  await initModelSelector();

  // Initialise crop toggle from server state
  initCropToggle(data.analyzer?.use_crop ?? false);

  const persistedSmooth = Number(
    data.settings?.analyzer?.smooth_window_sec ?? data.settings?.ui?.smooth_window_sec
  );
  if (Number.isFinite(persistedSmooth) && persistedSmooth >= 0.5 && persistedSmooth <= 10) {
    smoothWindowSec = persistedSmooth;
  }
  const gEl = document.getElementById('global_smooth_window');
  const gLabel = document.getElementById('global_smooth_window_val');
  const svEl = document.getElementById('sv_smooth_window');
  const svLabel = document.getElementById('sv_smooth_window_val');
  if (gEl) gEl.value = smoothWindowSec;
  if (gLabel) gLabel.textContent = smoothWindowSec.toFixed(1) + 's';
  if (svEl) svEl.value = smoothWindowSec;
  if (svLabel) svLabel.textContent = smoothWindowSec.toFixed(1) + 's';

  // Poll analyzer status until all done
  pollAnalyzerStatus();
}

function scheduleSaveUiSettings() {
  clearTimeout(uiSettingsDebounceTimer);
  uiSettingsDebounceTimer = setTimeout(() => saveUiSettings(), 150);
}

async function saveUiSettings() {
  try {
    await fetch('/tuning/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        analyzer: {
          smooth_window_sec: smoothWindowSec,
        },
      }),
    });
  } catch (e) {
    console.error('Failed to persist UI settings:', e);
  }
}

// ── Model selector ────────────────────────────────────────────────────────────
let activeModel = '';
let useCrop = false;
let cropRect = { top: 0.60, bottom: 0.95, left: 0.35, right: 0.65 };
let cropOverrides = {};
let draftVideoCrop = null;

async function initModelSelector() {
  const res = await fetch('/tuning/api/models');
  const data = await res.json();
  activeModel = data.current;
  renderModelButtons(data.models, data.current);
}

function renderModelButtons(models, current) {
  const group = document.getElementById('model-btn-group');
  if (!group) return;
  group.innerHTML = '';
  for (const m of models) {
    const btn = document.createElement('button');
    btn.className = 'model-btn' + (m === current ? ' model-btn--active' : '');
    // Strip extension for display, e.g. "rscd_resnet18_v2"
    btn.textContent = m.replace(/\.onnx$/i, '');
    btn.dataset.model = m;
    btn.addEventListener('click', () => switchModel(m));
    group.appendChild(btn);
  }
  setModelBadge(current, false);
}

function setModelBadge(modelName, loading) {
  const badge = document.getElementById('model-status-badge');
  if (!badge) return;
  if (loading) {
    badge.className = 'model-status-badge model-status-badge--loading';
    badge.textContent = 'Re-analysing…';
  } else {
    badge.className = 'model-status-badge model-status-badge--active';
    badge.textContent = 'active';
  }
}

async function switchModel(modelName) {
  if (modelName === activeModel) return;

  // Disable all buttons while switching
  const btns = document.querySelectorAll('.model-btn');
  btns.forEach(b => { b.disabled = true; });
  setModelBadge(modelName, true);

  // Update active highlight immediately for responsiveness
  btns.forEach(b => b.classList.toggle('model-btn--active', b.dataset.model === modelName));

  const res = await fetch('/tuning/api/model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: modelName }),
  });
  const data = await res.json();

  if (!data.ok) {
    console.error('Model switch failed:', data.error);
    // Revert highlight
    btns.forEach(b => b.classList.toggle('model-btn--active', b.dataset.model === activeModel));
    setModelBadge(activeModel, false);
    btns.forEach(b => { b.disabled = false; });
    return;
  }

  activeModel = modelName;
  // Re-poll until new inference is done, then refresh everything
  const analyzerBadge = document.getElementById('analyzer-badge');
  if (analyzerBadge) {
    analyzerBadge.className = 'badge';
    analyzerBadge.innerHTML = '<span class="dot"></span> Processing videos…';
  }
  pollAnalyzerStatusThen(() => {
    btns.forEach(b => { b.disabled = false; });
    setModelBadge(modelName, false);
  });
}

// ── Road crop toggle ──────────────────────────────────────────────────────────
function initCropToggle(useC) {
  useCrop = useC;
  updateCropButtons();
}

function updateCropButtons() {
  const btnFull = document.getElementById('crop-btn-full');
  const btnRoad = document.getElementById('crop-btn-road');
  if (btnFull) btnFull.classList.toggle('model-btn--active', !useCrop);
  if (btnRoad) btnRoad.classList.toggle('model-btn--active', useCrop);

  const hint = document.getElementById('crop-region-hint');
  if (hint) {
    const y1 = Math.round(cropRect.top * 100);
    const y2 = Math.round(cropRect.bottom * 100);
    const x1 = Math.round(cropRect.left * 100);
    const x2 = Math.round(cropRect.right * 100);
    hint.textContent = useCrop
      ? `y ${y1}-${y2} % · x ${x1}-${x2} % of frame`
      : '';
  }

  applyCropOverlays();
  updateVideoCropEditorEnabled();
}

function getCropForVideo(videoId) {
  if (videoId && cropOverrides[videoId]) return cropOverrides[videoId];
  return cropRect;
}

function applyCropRectToOverlay(el, rect) {
  el.style.top = `${rect.top * 100}%`;
  el.style.bottom = `${(1 - rect.bottom) * 100}%`;
  el.style.left = `${rect.left * 100}%`;
  el.style.right = `${(1 - rect.right) * 100}%`;
}

function applyCropOverlays() {
  const overlays = document.querySelectorAll('.model-crop-overlay');
  overlays.forEach((el) => {
    const videoId = el.dataset.videoId || currentVideoId;
    const rect = getCropForVideo(videoId);
    applyCropRectToOverlay(el, rect);
    el.style.display = useCrop ? 'block' : 'none';
  });
}

async function setCrop(useC) {
  if (useC === useCrop) return;

  const btnFull = document.getElementById('crop-btn-full');
  const btnRoad = document.getElementById('crop-btn-road');
  const badge   = document.getElementById('crop-status-badge');

  if (btnFull) btnFull.disabled = true;
  if (btnRoad) btnRoad.disabled = true;
  if (badge) {
    badge.className = 'model-status-badge model-status-badge--loading';
    badge.textContent = 'Re-analysing…';
  }

  const res = await fetch('/tuning/api/crop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ use_crop: useC }),
  });
  const data = await res.json();

  if (!data.ok) {
    if (btnFull) btnFull.disabled = false;
    if (btnRoad) btnRoad.disabled = false;
    if (badge) { badge.className = ''; badge.textContent = ''; }
    return;
  }

  useCrop = useC;
  updateCropButtons();

  const analyzerBadge = document.getElementById('analyzer-badge');
  if (analyzerBadge) {
    analyzerBadge.className = 'badge';
    analyzerBadge.innerHTML = '<span class="dot"></span> Processing videos…';
  }

  pollAnalyzerStatusThen(() => {
    if (btnFull) btnFull.disabled = false;
    if (btnRoad) btnRoad.disabled = false;
    if (badge) {
      badge.className = 'model-status-badge model-status-badge--active';
      badge.textContent = 'active';
    }
  });
}

// ── Analyzer status polling ───────────────────────────────────────────────────
function pollAnalyzerStatus() { pollAnalyzerStatusThen(null); }

async function pollAnalyzerStatusThen(onDone) {
  const res = await fetch('/tuning/api/analyzer-status');
  const data = await res.json();
  cropOverrides = data.crop_overrides || cropOverrides;

  const statuses = Object.values(data.videos).map(v => v.status);
  const allDone = statuses.every(s => s === 'done' || s.startsWith('error') || s === 'file_not_found' || s === 'no_opencv');
  const anyError = statuses.some(s => s.startsWith('error') || s === 'file_not_found');

  const badge = document.getElementById('analyzer-badge');
  if (allDone) {
    if (anyError) {
      badge.className = 'badge badge-error';
      badge.innerHTML = '<span class="dot"></span> Some videos failed';
    } else {
      badge.className = 'badge badge-done';
      badge.innerHTML = '<span class="dot"></span> All videos processed';
    }
    // Refresh recommendations now that inference is done
    await refreshRecommendations();
    applyCropOverlays();
    if (onDone) onDone();
  } else {
    const done = statuses.filter(s => s === 'done').length;
    badge.innerHTML = `<span class="dot"></span> Processing ${done}/${statuses.length} videos…`;
    setTimeout(() => pollAnalyzerStatusThen(onDone), 2000);
  }
}

// ── Slider sync ───────────────────────────────────────────────────────────────
function applyStateToSliders(s) {
  setValue('w_camera', s.w_camera, true);
  setValue('w_weather', s.w_weather, true);
  setValue('w_confidence', s.w_confidence, true);
  setValue('neutral_cam', s.neutral_cam, false, true);
  for (let lvl = 1; lvl <= 5; lvl++) {
    const val = s.difficulty_factors[String(lvl)];
    if (val !== undefined) setValue(`df_${lvl}`, val, false, true);
  }
  // Weather factor sliders
  if (s.wf_light_precip  !== undefined) setValue('wf_light_precip',  s.wf_light_precip,  false, true);
  if (s.wf_mod_precip    !== undefined) setValue('wf_mod_precip',    s.wf_mod_precip,    false, true);
  if (s.wf_heavy_precip  !== undefined) setValue('wf_heavy_precip',  s.wf_heavy_precip,  false, true);
  if (s.wf_near_freeze   !== undefined) setValue('wf_near_freeze',   s.wf_near_freeze,   false, true);
  if (s.wf_freeze        !== undefined) setValue('wf_freeze',        s.wf_freeze,        false, true);

  if (s.smooth_window_sec !== undefined) {
    smoothWindowSec = Number(s.smooth_window_sec);
    const gEl = document.getElementById('global_smooth_window');
    const gLabel = document.getElementById('global_smooth_window_val');
    const svEl = document.getElementById('sv_smooth_window');
    const svLabel = document.getElementById('sv_smooth_window_val');
    if (gEl) gEl.value = smoothWindowSec;
    if (gLabel) gLabel.textContent = smoothWindowSec.toFixed(1) + 's';
    if (svEl) svEl.value = smoothWindowSec;
    if (svLabel) svLabel.textContent = smoothWindowSec.toFixed(1) + 's';
  }
  // Also sync single-video sliders
  syncSvSliders(s);
}

function syncSvSliders(s) {
  setSvValue('sv_w_camera', s.w_camera, true);
  setSvValue('sv_w_weather', s.w_weather, true);
  setSvValue('sv_w_confidence', s.w_confidence, true);
  for (let lvl = 1; lvl <= 5; lvl++) {
    const val = s.difficulty_factors[String(lvl)];
    if (val !== undefined) setSvValue(`sv_df_${lvl}`, val, false, true);
  }
}

function setValue(id, val, isInt, isFloat) {
  const el = document.getElementById(id);
  const label = document.getElementById(id + '_val');
  if (!el) return;
  el.value = val;
  if (label) label.textContent = isInt ? Math.round(val) : parseFloat(val).toFixed(2);
}

function setSvValue(id, val, isInt, isFloat) {
  setValue(id, val, isInt, isFloat);
}

// ── Slider event wiring ───────────────────────────────────────────────────────
function buildParams() {
  const df = {};
  for (let i = 1; i <= 5; i++) {
    df[String(i)] = parseFloat(document.getElementById(`df_${i}`).value);
  }
  return {
    w_camera: parseFloat(document.getElementById('w_camera').value),
    w_weather: parseFloat(document.getElementById('w_weather').value),
    w_confidence: parseFloat(document.getElementById('w_confidence').value),
    neutral_cam: parseFloat(document.getElementById('neutral_cam').value),
    difficulty_factors: df,
    wf_light_precip:  parseFloat(document.getElementById('wf_light_precip').value),
    wf_mod_precip:    parseFloat(document.getElementById('wf_mod_precip').value),
    wf_heavy_precip:  parseFloat(document.getElementById('wf_heavy_precip').value),
    wf_near_freeze:   parseFloat(document.getElementById('wf_near_freeze').value),
    wf_freeze:        parseFloat(document.getElementById('wf_freeze').value),
  };
}

function buildSvParams() {
  const df = {};
  for (let i = 1; i <= 5; i++) {
    df[String(i)] = parseFloat(document.getElementById(`sv_df_${i}`).value);
  }
  return {
    w_camera: parseFloat(document.getElementById('sv_w_camera').value),
    w_weather: parseFloat(document.getElementById('sv_w_weather').value),
    w_confidence: parseFloat(document.getElementById('sv_w_confidence').value),
    difficulty_factors: df,
  };
}

function onMasterSliderChange(id, isInt) {
  const el = document.getElementById(id);
  const label = document.getElementById(id + '_val');
  const val = parseFloat(el.value);
  if (label) label.textContent = isInt ? Math.round(val) : val.toFixed(2);

  // Mirror to SV slider if open (only if the sv slider exists)
  const svId = 'sv_' + id;
  const svEl = document.getElementById(svId);
  const svLabel = document.getElementById(svId + '_val');
  if (svEl) {
    svEl.value = val;
    if (svLabel) svLabel.textContent = label ? label.textContent : (isInt ? Math.round(val) : val.toFixed(2));
  }

  const p = buildParams();
  scheduleTune(p);
  updateWeightBar(p);
}

function onSvSliderChange(id, isInt) {
  const el = document.getElementById(id);
  const label = document.getElementById(id + '_val');
  const val = parseFloat(el.value);
  if (label) label.textContent = isInt ? Math.round(val) : val.toFixed(2);

  // Mirror to master slider
  const masterId = id.replace('sv_', '');
  const masterEl = document.getElementById(masterId);
  const masterLabel = document.getElementById(masterId + '_val');
  if (masterEl) {
    masterEl.value = val;
    if (masterLabel) masterLabel.textContent = label.textContent;
  }

  scheduleTune(buildSvParams());
  updateWeightBar(buildSvParams());
  // Re-render single video prediction using current frame
  recomputeCurrentFramePrediction();
}

function scheduleTune(params) {
  clearTimeout(tuneDebounceTimer);
  tuneDebounceTimer = setTimeout(() => sendTune(params), 80);
}

async function sendTune(params) {
  try {
    const res = await fetch('/tuning/api/tune', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params),
    });
    const data = await res.json();
    state = data.state;
    for (const r of data.recommendations) {
      recommendations[r.video_id] = r;
    }
    updateAllVideoCards(data.recommendations);
    updateScoreDisplay(data.n_match, data.total, data.recommendations);
    // Also update single-video if open
    if (currentVideoId) {
      const rec = recommendations[currentVideoId];
      if (rec) updateSvPrediction(rec);
    }
  } catch (e) {
    console.error('Tune failed:', e);
  }
}

async function refreshRecommendations() {
  try {
    const res = await fetch('/tuning/api/recommendations');
    const data = await res.json();
    state = data.state;
    for (const r of data.recommendations) {
      recommendations[r.video_id] = r;
    }
    updateAllVideoCards(data.recommendations);
    updateScoreDisplay(data.n_match, data.total, data.recommendations);
  } catch (e) {}
}

// ── Weight bar ────────────────────────────────────────────────────────────────
function updateWeightBar(params) {
  const total = (params.w_camera || 0) + (params.w_weather || 0) + (params.w_confidence || 0);
  if (total <= 0) return;
  const camPct = (params.w_camera / total * 100).toFixed(1);
  const weaPct = (params.w_weather / total * 100).toFixed(1);
  const confPct = (params.w_confidence / total * 100).toFixed(1);
  const segCam = document.getElementById('seg-camera');
  const segWea = document.getElementById('seg-weather');
  const segConf = document.getElementById('seg-confidence');
  if (segCam) segCam.style.width = camPct + '%';
  if (segWea) segWea.style.width = weaPct + '%';
  if (segConf) segConf.style.width = confPct + '%';
}

// ── Video grid ────────────────────────────────────────────────────────────────
function renderVideoGrid() {
  const grid = document.getElementById('video-grid');
  grid.innerHTML = '';
  for (const v of videos) {
    const rec = recommendations[v.video_id] || {};
    grid.appendChild(buildVideoCard(v, rec));
  }
  applyCropOverlays();
}

function buildVideoCard(v, rec) {
  const card = document.createElement('div');
  card.className = 'video-card';
  card.id = `card-${v.video_id}`;
  card.onclick = () => openVideo(v.video_id);

  const recommended = rec.recommended_speed ?? '—';
  const target = v.target_speed ?? '—';
  const delta = rec.delta ?? null;
  const topLabel = rec.details?.top_label ?? '';
  const matchClass = getMatchClass(delta);
  const deltaText = getDeltaText(delta);
  const diffClass = `diff-${v.difficulty}`;
  const weather = rec.weather || {};
  const color = classColors[topLabel] || '#8b949e';

  const wfactor = weather.weather_factor ?? 1.0;
  const wfactorColor = wfactor >= 0.85 ? 'var(--green)' : wfactor >= 0.5 ? 'var(--yellow)' : 'var(--red)';
  const tempStr = weather.temp_c !== undefined ? `${weather.temp_c}°C` : '';
  const precipStr = weather.precipitation_mm_h !== undefined && weather.precipitation_mm_h > 0
    ? ` · ${weather.precipitation_mm_h}mm/h` : '';

  const processingDot = `
    <div class="vc-processing" id="proc-${v.video_id}">
      <div class="vc-spinner"></div>
      <span>Analysing…</span>
    </div>`;

  card.innerHTML = `
    <div class="vc-thumb">
      <video muted preload="none" src="/tuning/videos/${v.filename}" id="thumb-${v.video_id}"></video>
      <div class="model-crop-overlay" data-video-id="${v.video_id}"></div>
      ${processingDot}
      <div class="vc-overlay">
        <div class="vc-play-icon">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="white">
            <path d="M3 2l7 4-7 4V2z"/>
          </svg>
        </div>
      </div>
    </div>
    <div class="vc-body">
      <div class="vc-id-row">
        <span class="vc-id">${v.video_id.toUpperCase()}</span>
        <span class="difficulty-badge ${diffClass}">${v.difficulty}</span>
      </div>
      <div class="vc-posted">Posted: ${v.posted_speed} km/h · ${v.n_responses} responses</div>
      <div class="vc-weather-row">
        <span class="vc-weather-temp">${tempStr}${precipStr}</span>
        <span class="vc-weather-factor" style="color:${wfactorColor}" title="Weather factor">
          ⚡${wfactor.toFixed(2)}
        </span>
      </div>
      <div class="vc-speeds">
        <div class="vc-speed-col vc-target-col">
          <span class="vc-speed-label">Target</span>
          <span class="vc-speed-val" id="card-target-${v.video_id}">${target}</span>
          <span class="vc-speed-unit">km/h</span>
        </div>
        <div class="vc-speed-col vc-model-col">
          <span class="vc-speed-label">Model</span>
          <span class="vc-speed-val" id="card-model-${v.video_id}">${recommended}</span>
          <span class="vc-speed-unit">km/h</span>
        </div>
      </div>
      <div class="vc-delta ${matchClass === 'match-yes' ? 'delta-zero' : matchClass === 'match-near' ? 'delta-near' : 'delta-far'}"
           id="card-delta-${v.video_id}">
        ${deltaText}
      </div>
      ${topLabel ? `
      <div class="vc-top-class">
        <span class="vc-class-dot" style="background:${color}"></span>
        <span id="card-class-${v.video_id}">${formatLabel(topLabel)}</span>
      </div>` : ''}
    </div>
    <div class="vc-match-badge ${matchClass}" id="card-badge-${v.video_id}">
      ${matchClass === 'match-yes' ? '✓' : matchClass === 'match-near' ? '~' : '✗'}
    </div>`;

  return card;
}

function updateAllVideoCards(recList) {
  for (const rec of recList) {
    updateVideoCard(rec);
    // Hide processing overlay once we have data
    const proc = document.getElementById(`proc-${rec.video_id}`);
    if (proc && rec.details?.top_label && rec.details.top_label !== 'unknown') {
      proc.style.display = 'none';
    }
  }
}

function updateVideoCard(rec) {
  const modelEl = document.getElementById(`card-model-${rec.video_id}`);
  const deltaEl = document.getElementById(`card-delta-${rec.video_id}`);
  const badgeEl = document.getElementById(`card-badge-${rec.video_id}`);
  const classEl = document.getElementById(`card-class-${rec.video_id}`);

  if (!modelEl) return;

  const delta = rec.delta;
  const matchClass = getMatchClass(delta);
  const deltaText = getDeltaText(delta);

  modelEl.textContent = rec.recommended_speed;

  if (deltaEl) {
    deltaEl.textContent = deltaText;
    deltaEl.className = `vc-delta ${matchClass === 'match-yes' ? 'delta-zero' : matchClass === 'match-near' ? 'delta-near' : 'delta-far'}`;
  }
  if (badgeEl) {
    badgeEl.className = `vc-match-badge ${matchClass}`;
    badgeEl.textContent = matchClass === 'match-yes' ? '✓' : matchClass === 'match-near' ? '~' : '✗';
  }
  if (classEl && rec.details?.top_label) {
    classEl.textContent = formatLabel(rec.details.top_label);
  }
}

// ── Score display ─────────────────────────────────────────────────────────────
function updateScoreDisplay(nMatch, total, recList) {
  document.getElementById('score-label').textContent = `${nMatch} / ${total}`;
  document.getElementById('score-num').textContent = nMatch;

  // Ring
  const circumference = 213.63;
  const pct = total > 0 ? nMatch / total : 0;
  const offset = circumference * (1 - pct);
  const ring = document.getElementById('score-ring-fill');
  if (ring) {
    ring.style.strokeDashoffset = offset;
    ring.style.stroke = pct >= 0.8 ? '#3fb950' : pct >= 0.5 ? '#d29922' : '#f85149';
  }

  // Breakdown
  let exact = 0, near = 0, off = 0;
  for (const r of recList) {
    const d = Math.abs(r.delta || 0);
    if (d === 0) exact++;
    else if (d <= 10) near++;
    else off++;
  }
  const accExact = document.getElementById('acc-exact');
  const accNear = document.getElementById('acc-near');
  const accOff = document.getElementById('acc-off');
  if (accExact) accExact.textContent = `${exact} exact`;
  if (accNear) accNear.textContent = `${near} within ±10`;
  if (accOff) accOff.textContent = `${off} off`;
}

// ── Single video view ─────────────────────────────────────────────────────────
async function openVideo(videoId) {
  currentVideoId = videoId;
  const v = videos.find(x => x.video_id === videoId);
  if (!v) return;

  // Set metadata in header
  document.getElementById('sv-title').textContent = videoId.toUpperCase();
  document.getElementById('sv-posted').textContent = `Posted: ${v.posted_speed} km/h`;
  document.getElementById('sv-responses').textContent = `${v.n_responses} responses`;
  document.getElementById('sv-target').textContent = v.target_speed;
  document.getElementById('spcb-target').textContent = v.target_speed;
  document.getElementById('sv-badge').textContent = v.difficulty;
  document.getElementById('sv-badge').className = `difficulty-badge diff-${v.difficulty}`;

  // Set video source
  const videoEl = document.getElementById('main-video');
  videoEl.src = `/tuning/videos/${v.filename}`;
  videoEl.load();
  const mainOverlay = document.getElementById('main-crop-overlay');
  if (mainOverlay) mainOverlay.dataset.videoId = videoId;
  applyCropOverlays();
  await loadVideoCropEditor(videoId);

  // Switch views
  document.getElementById('master-view').classList.remove('active');
  document.getElementById('video-view').classList.add('active');

  // Sync sliders
  syncSvSliders(state);

  // Fetch pre-computed frame predictions
  videoFramePredictions = [];
  videoDuration = 0;
  await loadVideoPredictions(videoId, v);

  // Wire video timeupdate
  videoEl.ontimeupdate = () => onVideoTimeUpdate(videoEl.currentTime);
}

async function loadVideoPredictions(videoId, videoMeta) {
  try {
    const res = await fetch(`/tuning/api/video/${videoId}/predictions`);
    const data = await res.json();
    videoFramePredictions = data.frames || [];
    videoDuration = data.duration || 0;
    renderFrameTimeline();

    if (data.weather) {
      renderWeatherInfo(data.weather);
    }

    if (data.avg_recommendation) {
      updateSvPrediction({
        recommended_speed: data.avg_recommendation.recommended,
        target_speed: videoMeta.target_speed,
        delta: data.avg_recommendation.recommended - videoMeta.target_speed,
        details: data.avg_recommendation,
      });
    }
  } catch (e) {
    console.error('Failed to load predictions:', e);
  }
}

function showMaster() {
  const videoEl = document.getElementById('main-video');
  if (videoEl) { videoEl.pause(); videoEl.src = ''; }
  currentVideoId = null;
  document.getElementById('video-view').classList.remove('active');
  document.getElementById('master-view').classList.add('active');
}

function updateVideoCropEditorEnabled() {
  const editor = document.getElementById('sv-crop-editor');
  if (!editor) return;
  const disabled = !useCrop;
  editor.classList.toggle('sv-crop-editor--disabled', disabled);
  editor.querySelectorAll('input,button').forEach((el) => {
    el.disabled = disabled;
  });
  const status = document.getElementById('sv-crop-status');
  if (status && disabled) {
    status.textContent = 'Enable "Road crop" in dashboard to use per-video crop';
  }
}

function readDraftVideoCrop() {
  const left = parseFloat(document.getElementById('sv_crop_left').value);
  const right = parseFloat(document.getElementById('sv_crop_right').value);
  const top = parseFloat(document.getElementById('sv_crop_top').value);
  const bottom = parseFloat(document.getElementById('sv_crop_bottom').value);

  const rect = { left, right, top, bottom };
  if (rect.right - rect.left < 0.05) rect.right = Math.min(1.0, rect.left + 0.05);
  if (rect.bottom - rect.top < 0.05) rect.bottom = Math.min(1.0, rect.top + 0.05);
  return rect;
}

function setCropEditorRect(rect) {
  const ids = ['left', 'right', 'top', 'bottom'];
  for (const k of ids) {
    const input = document.getElementById(`sv_crop_${k}`);
    const val = document.getElementById(`sv_crop_${k}_val`);
    if (input) input.value = Number(rect[k]).toFixed(2);
    if (val) val.textContent = Number(rect[k]).toFixed(2);
  }
}

function updateDraftCropPreview() {
  if (!currentVideoId || !draftVideoCrop) return;
  const overlay = document.getElementById('main-crop-overlay');
  if (overlay) applyCropRectToOverlay(overlay, draftVideoCrop);
}

async function loadVideoCropEditor(videoId) {
  try {
    const res = await fetch(`/tuning/api/video/${videoId}/crop`);
    const data = await res.json();
    if (!data.ok) return;
    const rect = data.crop || cropRect;
    draftVideoCrop = { ...rect };
    setCropEditorRect(rect);
    const status = document.getElementById('sv-crop-status');
    if (status) status.textContent = data.has_override ? 'Custom override active' : 'Using global crop';
    updateVideoCropEditorEnabled();
    updateDraftCropPreview();
  } catch (e) {
    console.error('Failed to load video crop:', e);
  }
}

async function saveVideoCrop() {
  if (!currentVideoId) return;
  const status = document.getElementById('sv-crop-status');
  const rect = readDraftVideoCrop();
  draftVideoCrop = { ...rect };
  if (status) status.textContent = 'Saving and re-analysing…';

  try {
    const res = await fetch(`/tuning/api/video/${currentVideoId}/crop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ crop: rect }),
    });
    const data = await res.json();
    if (!data.ok) {
      if (status) status.textContent = `Save failed: ${data.error || 'unknown error'}`;
      return;
    }

    cropOverrides[currentVideoId] = data.crop;
    applyCropOverlays();
    const analyzerBadge = document.getElementById('analyzer-badge');
    if (analyzerBadge) {
      analyzerBadge.className = 'badge';
      analyzerBadge.innerHTML = '<span class="dot"></span> Processing videos…';
    }
    pollAnalyzerStatusThen(async () => {
      const v = videos.find(x => x.video_id === currentVideoId);
      if (v) await loadVideoPredictions(currentVideoId, v);
      if (status) status.textContent = 'Custom override active';
      applyCropOverlays();
    });
  } catch (e) {
    if (status) status.textContent = 'Save failed: network error';
    console.error('Failed to save video crop:', e);
  }
}

async function resetVideoCrop() {
  if (!currentVideoId) return;
  const status = document.getElementById('sv-crop-status');
  if (status) status.textContent = 'Resetting and re-analysing…';
  try {
    const res = await fetch(`/tuning/api/video/${currentVideoId}/crop`, { method: 'DELETE' });
    const data = await res.json();
    if (!data.ok) {
      if (status) status.textContent = `Reset failed: ${data.error || 'unknown error'}`;
      return;
    }
    delete cropOverrides[currentVideoId];
    draftVideoCrop = { ...(data.crop || cropRect) };
    setCropEditorRect(draftVideoCrop);
    applyCropOverlays();
    const analyzerBadge = document.getElementById('analyzer-badge');
    if (analyzerBadge) {
      analyzerBadge.className = 'badge';
      analyzerBadge.innerHTML = '<span class="dot"></span> Processing videos…';
    }
    pollAnalyzerStatusThen(async () => {
      const v = videos.find(x => x.video_id === currentVideoId);
      if (v) await loadVideoPredictions(currentVideoId, v);
      if (status) status.textContent = 'Using global crop';
      applyCropOverlays();
    });
  } catch (e) {
    if (status) status.textContent = 'Reset failed: network error';
    console.error('Failed to reset video crop:', e);
  }
}

/**
 * Average class probabilities over all pre-computed frames within
 * [currentTime - smoothWindowSec, currentTime].
 * Falls back to the single nearest frame when no frames are in the window.
 * Returns { scores, frameCount, topLabel }.
 */
function getSmoothedScores(currentTime) {
  if (videoFramePredictions.length === 0) return null;

  const cutoff = currentTime - smoothWindowSec;
  const window = videoFramePredictions.filter(f => f.t >= cutoff && f.t <= currentTime + 0.05);

  // Fallback: single nearest frame
  if (window.length === 0) {
    let nearest = videoFramePredictions[0];
    let minDist = Math.abs(currentTime - nearest.t);
    for (const frame of videoFramePredictions) {
      const d = Math.abs(currentTime - frame.t);
      if (d < minDist) { minDist = d; nearest = frame; }
    }
    return { scores: nearest.scores, frameCount: 1 };
  }

  // Average probabilities across the window
  const avg = {};
  const classes = Object.keys(window[0].scores);
  for (const cls of classes) avg[cls] = 0;
  for (const frame of window) {
    for (const cls of classes) avg[cls] += frame.scores[cls];
  }
  for (const cls of classes) avg[cls] /= window.length;

  return { scores: avg, frameCount: window.length };
}

function onVideoTimeUpdate(currentTime) {
  if (videoFramePredictions.length === 0) return;

  updateFrameTimeline(currentTime);

  const smoothed = getSmoothedScores(currentTime);
  if (!smoothed) return;

  // Update frame count indicator
  updateSmoothingIndicator(smoothed.frameCount);

  // Update class bars with smoothed scores
  renderClassBars(smoothed.scores);

  // Compute recommendation from smoothed scores
  recomputeFramePrediction(smoothed.scores);
}

function recomputeCurrentFramePrediction() {
  const videoEl = document.getElementById('main-video');
  if (!videoEl || videoFramePredictions.length === 0) return;
  const smoothed = getSmoothedScores(videoEl.currentTime);
  if (!smoothed) return;
  updateSmoothingIndicator(smoothed.frameCount);
  recomputeFramePrediction(smoothed.scores);
}

function updateSmoothingIndicator(frameCount) {
  const el = document.getElementById('smooth-frame-count');
  if (el) el.textContent = `${frameCount} frame${frameCount !== 1 ? 's' : ''} window (backend)`;
}

async function recomputeFramePrediction(scores) {
  if (!currentVideoId) return;
  try {
    const res = await fetch(`/tuning/api/video/${currentVideoId}/frame-recommendation`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ scores }),
    });
    const rec = await res.json();
    const v = videos.find(x => x.video_id === currentVideoId);
    if (!v) return;
    updateSvPrediction({
      recommended_speed: rec.recommended,
      target_speed: v.target_speed,
      delta: rec.recommended - v.target_speed,
      details: rec,
    });
  } catch (e) {}
}

function updateSvPrediction(rec) {
  const det = rec.details || {};
  const topLabel = det.top_label || '—';
  const conf = det.cam_confidence ?? 0;
  const rawCam = det.raw_cam_factor ?? det.effective_cam_factor ?? 0;
  const combined = det.combined_factor ?? 0;
  const weather = det.weather_factor ?? 1;

  document.getElementById('spc-top-label').textContent = formatLabel(topLabel);
  document.getElementById('spc-conf-val').textContent = (conf * 100).toFixed(0) + '%';
  document.getElementById('spc-cam-factor').textContent = rawCam.toFixed(3);
  document.getElementById('spc-wea-factor').textContent = weather.toFixed(3);
  document.getElementById('spc-combined').textContent = combined.toFixed(3);
  document.getElementById('spcb-model').textContent = rec.recommended_speed ?? '—';

  const delta = rec.delta ?? 0;
  const matchCls = getMatchClass(delta);
  const modelBox = document.getElementById('spc-model-box');
  if (modelBox) modelBox.className = `spc-speed-box spc-model-box ${matchCls}`;

  const matchIcon = document.getElementById('spc-match-icon');
  if (matchIcon) {
    matchIcon.textContent = matchCls === 'match-yes' ? '✓' : matchCls === 'match-near' ? '≈' : '✗';
    matchIcon.style.color = matchCls === 'match-yes' ? 'var(--green)' : matchCls === 'match-near' ? 'var(--yellow)' : 'var(--red)';
  }

  const deltaLabel = document.getElementById('spc-delta-label');
  if (deltaLabel) {
    deltaLabel.textContent = delta === 0 ? 'On target' : `${delta > 0 ? '+' : ''}${delta} km/h`;
    deltaLabel.style.color = matchCls === 'match-yes' ? 'var(--green)' : matchCls === 'match-near' ? 'var(--yellow)' : 'var(--red)';
  }
}

// ── Weather info (single video view) ─────────────────────────────────────────
function renderWeatherInfo(weather) {
  const tempEl = document.getElementById('svwi-temp');
  const precipEl = document.getElementById('svwi-precip');
  const humEl = document.getElementById('svwi-humidity');
  const factorEl = document.getElementById('svwi-factor');
  const reasonsEl = document.getElementById('svwi-reasons');

  if (tempEl) tempEl.textContent = `${weather.temp_c} °C`;
  if (precipEl) precipEl.textContent = `${weather.precipitation_mm_h} mm/h`;
  if (humEl) humEl.textContent = `${weather.humidity} %`;
  if (factorEl) {
    factorEl.textContent = weather.weather_factor.toFixed(2);
    factorEl.style.color = weather.weather_factor >= 0.85 ? 'var(--green)'
      : weather.weather_factor >= 0.5 ? 'var(--yellow)' : 'var(--red)';
  }
  if (reasonsEl) {
    reasonsEl.innerHTML = (weather.reasons || [])
      .map(r => `<span class="svwi-reason-tag">${r}</span>`)
      .join('');
  }
}

// ── Frame timeline ────────────────────────────────────────────────────────────
function renderFrameTimeline() {
  const tl = document.getElementById('frame-timeline');
  tl.innerHTML = '';
  if (!videoFramePredictions.length) return;

  for (const frame of videoFramePredictions) {
    const sorted = Object.entries(frame.scores).sort((a, b) => b[1] - a[1]);
    const topCls = sorted[0][0];
    const color = classColors[topCls] || '#8b949e';
    const tick = document.createElement('div');
    tick.className = 'ft-tick';
    tick.style.background = color;
    tick.title = `${frame.t.toFixed(1)}s: ${formatLabel(topCls)} (${(sorted[0][1] * 100).toFixed(0)}%)`;
    tick.dataset.t = frame.t;
    tick.onclick = (e) => {
      e.stopPropagation();
      const videoEl = document.getElementById('main-video');
      if (videoEl) videoEl.currentTime = parseFloat(tick.dataset.t);
    };
    tl.appendChild(tick);
  }
}

function updateFrameTimeline(currentTime) {
  const tl = document.getElementById('frame-timeline');
  const ticks = tl.querySelectorAll('.ft-tick');
  ticks.forEach(tick => {
    const t = parseFloat(tick.dataset.t);
    tick.classList.toggle('active', Math.abs(t - currentTime) < (videoDuration / videoFramePredictions.length / 2 + 0.1));
  });
}

// ── Class bars ────────────────────────────────────────────────────────────────
function renderClassBars(scores) {
  const list = document.getElementById('scb-list');
  if (!list) return;
  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  list.innerHTML = '';
  for (const [cls, prob] of sorted) {
    const pct = (prob * 100).toFixed(1);
    const color = classColors[cls] || '#8b949e';
    const item = document.createElement('div');
    item.className = 'scb-item';
    item.innerHTML = `
      <span class="scb-name">
        <span class="scb-dot" style="background:${color}"></span>
        ${formatLabel(cls)}
      </span>
      <div class="scb-bar-bg">
        <div class="scb-bar-fill" style="width:${Math.min(pct, 100)}%;background:${color}"></div>
      </div>
      <span class="scb-pct">${pct}%</span>`;
    list.appendChild(item);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function getMatchClass(delta) {
  if (delta === null || delta === undefined) return 'match-no';
  const absDelta = Math.abs(delta);
  if (absDelta === 0) return 'match-yes';
  if (absDelta <= 10) return 'match-near';
  return 'match-no';
}

function getDeltaText(delta) {
  if (delta === null || delta === undefined) return '—';
  if (delta === 0) return '✓ On target';
  return `${delta > 0 ? '+' : ''}${delta} km/h`;
}

function formatLabel(label) {
  return (label || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ── Wire up master sliders ────────────────────────────────────────────────────
// ── Auto-tune ─────────────────────────────────────────────────────────────────
function applyAutoTuneResult(data) {
  const resultPanel = document.getElementById('auto-tune-result');
  if (resultPanel) resultPanel.style.display = 'block';

  applyStateToSliders(data.state);
  updateWeightBar(data.state);

  updateAllVideoCards(data.recommendations);
  updateScoreDisplay(data.n_match_after, data.total, data.recommendations);
  for (const r of data.recommendations) {
    recommendations[r.video_id] = r;
  }

  const itersEl = document.getElementById('atr-iters');
  const convEl = document.getElementById('atr-converged');

  const exactBefore = document.getElementById('atr-exact-before');
  const exactAfter = document.getElementById('atr-exact-after');
  const nearBefore = document.getElementById('atr-near-before');
  const nearAfter = document.getElementById('atr-near-after');
  if (exactBefore) exactBefore.textContent = `${data.n_match_before}/10`;
  if (exactAfter) exactAfter.textContent = `${data.n_match_after}/10`;
  if (nearBefore) nearBefore.textContent = `${data.n_near_before}/10`;
  if (nearAfter) nearAfter.textContent = `${data.n_near_after}/10`;
  if (itersEl) itersEl.textContent = data.n_iterations;
  if (convEl) {
    convEl.textContent = data.converged ? '✓ Converged' : '⚠ Stopped early';
    convEl.style.color = data.converged ? 'var(--green)' : 'var(--yellow)';
  }

  const listEl = document.getElementById('atr-video-list');
  if (listEl && data.per_video) {
    listEl.innerHTML = data.per_video.map(v => {
      const matchCls = v.match ? 'atr-match-yes' : Math.abs(v.delta) <= 10 ? 'atr-match-near' : 'atr-match-no';
      const deltaStr = v.delta === 0 ? '✓' : `${v.delta > 0 ? '+' : ''}${v.delta}`;
      const diffLabel = v.difficulty.charAt(0).toUpperCase();
      return `<div class="atr-row ${matchCls}">
        <span class="atr-vid">${v.video_id.toUpperCase()}</span>
        <span class="atr-diff diff-${v.difficulty}">${diffLabel}</span>
        <span class="atr-target">${v.target} km/h</span>
        <span class="atr-rec">${v.recommended} km/h</span>
        <span class="atr-delta">${deltaStr}</span>
      </div>`;
    }).join('');
  }
}

function setAutoTuneUiRunning(running) {
  const btn = document.getElementById('auto-tune-btn');
  const stopBtn = document.getElementById('auto-tune-stop');
  const prog = document.getElementById('auto-tune-progress');
  if (!btn || !stopBtn || !prog) return;
  btn.disabled = running;
  stopBtn.style.display = running ? 'inline-block' : 'none';
  prog.style.display = running ? 'block' : 'none';
  if (!running) {
    btn.innerHTML = `<span class="atb-icon">⚡</span><span class="atb-text">Auto-tune</span><span class="atb-sub">Run again</span>`;
  }
}

function updateAutoTuneProgressUi(status) {
  const fill = document.getElementById('auto-tune-progress-fill');
  const label = document.getElementById('auto-tune-progress-label');
  const meta = document.getElementById('auto-tune-progress-meta');
  if (!fill || !label || !meta) return;

  const pct = Math.max(0, Math.min(1, Number(status.progress || 0)));
  fill.style.width = `${(pct * 100).toFixed(1)}%`;
  label.textContent = status.message || 'Optimising...';
  meta.textContent = `${status.iteration || 0} / ${status.maxiter || 0} · exact ${status.best_exact || 0}/10`;
}

async function pollAutoTuneStatus() {
  const res = await fetch('/tuning/api/auto-tune/status');
  const status = await res.json();
  if (!status.ok) return;

  updateAutoTuneProgressUi(status);

  if (status.status === 'done' && status.result) {
    if (autoTunePollTimer) clearInterval(autoTunePollTimer);
    autoTunePollTimer = null;
    applyAutoTuneResult(status.result);
    setAutoTuneUiRunning(false);
    return;
  }

  if (status.status === 'error') {
    if (autoTunePollTimer) clearInterval(autoTunePollTimer);
    autoTunePollTimer = null;
    setAutoTuneUiRunning(false);
    const btn = document.getElementById('auto-tune-btn');
    if (btn) {
      btn.innerHTML = `<span class="atb-icon">⚡</span><span class="atb-text">Auto-tune</span><span class="atb-sub">Error: ${status.message || 'failed'}</span>`;
    }
  }
}

async function runAutoTune() {
  const btn = document.getElementById('auto-tune-btn');
  setAutoTuneUiRunning(true);
  if (btn) {
    btn.innerHTML = `<span class="atb-icon at-spin">⚙</span><span class="atb-text">Optimising…</span><span class="atb-sub">Can take several minutes</span>`;
  }

  try {
    const res = await fetch('/tuning/api/auto-tune/start', { method: 'POST' });
    const data = await res.json();
    if (!data.ok) {
      setAutoTuneUiRunning(false);
      if (btn) {
        btn.innerHTML = `<span class="atb-icon">⚡</span><span class="atb-text">Auto-tune</span><span class="atb-sub">Error: ${data.error || 'failed'}</span>`;
      }
      return;
    }

    if (autoTunePollTimer) clearInterval(autoTunePollTimer);
    await pollAutoTuneStatus();
    autoTunePollTimer = setInterval(() => {
      pollAutoTuneStatus().catch((e) => console.error('Auto-tune status poll failed:', e));
    }, 1000);
  } catch (e) {
    setAutoTuneUiRunning(false);
    if (btn) {
      btn.innerHTML = `<span class="atb-icon">⚡</span><span class="atb-text">Auto-tune</span><span class="atb-sub">Network error</span>`;
    }
    console.error('Auto-tune failed:', e);
  }
}

async function stopAutoTune() {
  try {
    await fetch('/tuning/api/auto-tune/stop', { method: 'POST' });
  } catch (e) {
    console.error('Failed to stop auto-tune:', e);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  // Master sliders
  ['w_camera', 'w_weather', 'w_confidence'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', () => onMasterSliderChange(id, true));
  });
  ['neutral_cam'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', () => onMasterSliderChange(id, false));
  });
  for (let i = 1; i <= 5; i++) {
    const id = `df_${i}`;
    document.getElementById(id)?.addEventListener('input', () => onMasterSliderChange(id, false));
  }

  // Single-video sliders
  ['sv_w_camera', 'sv_w_weather', 'sv_w_confidence'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', () => onSvSliderChange(id, true));
  });
  for (let i = 1; i <= 5; i++) {
    const id = `sv_df_${i}`;
    document.getElementById(id)?.addEventListener('input', () => onSvSliderChange(id, false));
  }

  // Weather factor sliders
  ['wf_light_precip', 'wf_mod_precip', 'wf_heavy_precip', 'wf_near_freeze', 'wf_freeze'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', () => onMasterSliderChange(id, false));
  });

  // Global smoothing window slider (master dashboard)
  const globalSmoothEl = document.getElementById('global_smooth_window');
  if (globalSmoothEl) {
    globalSmoothEl.addEventListener('input', () => {
      smoothWindowSec = parseFloat(globalSmoothEl.value);
      const label = document.getElementById('global_smooth_window_val');
      if (label) label.textContent = smoothWindowSec.toFixed(1) + 's';
      // Keep single-video slider in sync
      const svEl = document.getElementById('sv_smooth_window');
      if (svEl) { svEl.value = smoothWindowSec; }
      const svLabel = document.getElementById('sv_smooth_window_val');
      if (svLabel) svLabel.textContent = smoothWindowSec.toFixed(1) + 's';
      scheduleSaveUiSettings();
      recomputeCurrentFramePrediction();
    });
  }

  // Single-video smoothing slider — also updates global slider
  const smoothEl = document.getElementById('sv_smooth_window');
  if (smoothEl) {
    smoothEl.addEventListener('input', () => {
      smoothWindowSec = parseFloat(smoothEl.value);
      const label = document.getElementById('sv_smooth_window_val');
      if (label) label.textContent = smoothWindowSec.toFixed(1) + 's';
      // Keep global slider in sync
      const gEl = document.getElementById('global_smooth_window');
      if (gEl) { gEl.value = smoothWindowSec; }
      const gLabel = document.getElementById('global_smooth_window_val');
      if (gLabel) gLabel.textContent = smoothWindowSec.toFixed(1) + 's';
      scheduleSaveUiSettings();
      recomputeCurrentFramePrediction();
    });
  }

  ['left', 'right', 'top', 'bottom'].forEach((k) => {
    const el = document.getElementById(`sv_crop_${k}`);
    if (!el) return;
    el.addEventListener('input', () => {
      draftVideoCrop = readDraftVideoCrop();
      setCropEditorRect(draftVideoCrop);
      updateDraftCropPreview();
    });
  });

  init();
});
