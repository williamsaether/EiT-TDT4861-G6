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
let smoothWindowSec = 3.0;       // rolling average window for live playback

// Pending parameter changes (accumulated before debounced send)
let pendingParams = {};

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  const res = await fetch('/api/init');
  const data = await res.json();

  videos = data.videos;
  state = data.state;
  classColors = data.class_colors;
  difficultyNames = data.difficulty_names;

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

  // Poll analyzer status until all done
  pollAnalyzerStatus();
}

// ── Model selector ────────────────────────────────────────────────────────────
let activeModel = '';
let useCrop = false;

async function initModelSelector() {
  const res = await fetch('/api/models');
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

  const res = await fetch('/api/model', {
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
    hint.textContent = useCrop
      ? 'y 35–85 % · x 10–90 % of frame'
      : '';
  }
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

  const res = await fetch('/api/crop', {
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
  const res = await fetch('/api/analyzer-status');
  const data = await res.json();

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
    const res = await fetch('/api/tune', {
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
    const res = await fetch('/api/recommendations');
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
      <video muted preload="none" src="/videos/${v.filename}" id="thumb-${v.video_id}"></video>
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
  videoEl.src = `/videos/${v.filename}`;
  videoEl.load();

  // Switch views
  document.getElementById('master-view').classList.remove('active');
  document.getElementById('video-view').classList.add('active');

  // Sync sliders
  syncSvSliders(state);

  // Fetch pre-computed frame predictions
  videoFramePredictions = [];
  videoDuration = 0;
  try {
    const res = await fetch(`/api/video/${videoId}/predictions`);
    const data = await res.json();
    videoFramePredictions = data.frames || [];
    videoDuration = data.duration || 0;
    renderFrameTimeline();

    // Show weather info from metadata
    if (data.weather) {
      renderWeatherInfo(data.weather);
    }

    // Show avg recommendation
    if (data.avg_recommendation) {
      updateSvPrediction({
        recommended_speed: data.avg_recommendation.recommended,
        target_speed: v.target_speed,
        delta: data.avg_recommendation.recommended - v.target_speed,
        details: data.avg_recommendation,
      });
    }
  } catch (e) {
    console.error('Failed to load predictions:', e);
  }

  // Wire video timeupdate
  videoEl.ontimeupdate = () => onVideoTimeUpdate(videoEl.currentTime);
}

function showMaster() {
  const videoEl = document.getElementById('main-video');
  if (videoEl) { videoEl.pause(); videoEl.src = ''; }
  currentVideoId = null;
  document.getElementById('video-view').classList.remove('active');
  document.getElementById('master-view').classList.add('active');
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
  if (el) el.textContent = `${frameCount} frame${frameCount !== 1 ? 's' : ''} averaged`;
}

async function recomputeFramePrediction(scores) {
  if (!currentVideoId) return;
  try {
    const res = await fetch(`/api/video/${currentVideoId}/frame-recommendation`, {
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
async function runAutoTune() {
  const btn = document.getElementById('auto-tune-btn');
  const resultPanel = document.getElementById('auto-tune-result');

  // Show loading state
  btn.disabled = true;
  btn.innerHTML = `<span class="atb-icon at-spin">⚙</span><span class="atb-text">Optimising…</span><span class="atb-sub">This takes ~5 seconds</span>`;

  try {
    const res = await fetch('/api/auto-tune', { method: 'POST' });
    const data = await res.json();

    if (data.error) {
      btn.disabled = false;
      btn.innerHTML = `<span class="atb-icon">⚡</span><span class="atb-text">Auto-tune</span><span class="atb-sub">Error: ${data.error}</span>`;
      return;
    }

    // Apply new state to all sliders
    applyStateToSliders(data.state);
    updateWeightBar(data.state);

    // Update all video cards
    updateAllVideoCards(data.recommendations);
    updateScoreDisplay(data.n_match_after, data.total, data.recommendations);
    for (const r of data.recommendations) {
      recommendations[r.video_id] = r;
    }

    // Show result panel
    resultPanel.style.display = 'block';
    const beforeEl = document.getElementById('atr-before');
    const afterEl  = document.getElementById('atr-after');
    const itersEl  = document.getElementById('atr-iters');
    const convEl   = document.getElementById('atr-converged');

    const exactBefore = document.getElementById('atr-exact-before');
    const exactAfter  = document.getElementById('atr-exact-after');
    const nearBefore  = document.getElementById('atr-near-before');
    const nearAfter   = document.getElementById('atr-near-after');
    if (exactBefore) exactBefore.textContent = `${data.n_match_before}/10`;
    if (exactAfter)  exactAfter.textContent  = `${data.n_match_after}/10`;
    if (nearBefore)  nearBefore.textContent  = `${data.n_near_before}/10`;
    if (nearAfter)   nearAfter.textContent   = `${data.n_near_after}/10`;
    if (itersEl)  itersEl.textContent  = data.n_iterations;
    if (convEl) {
      convEl.textContent = data.converged ? '✓ Converged' : '⚠ Not converged';
      convEl.style.color = data.converged ? 'var(--green)' : 'var(--yellow)';
    }

    // Per-video breakdown
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

    // Reset button
    btn.disabled = false;
    btn.innerHTML = `<span class="atb-icon">⚡</span><span class="atb-text">Auto-tune</span><span class="atb-sub">Run again</span>`;

  } catch (e) {
    btn.disabled = false;
    btn.innerHTML = `<span class="atb-icon">⚡</span><span class="atb-text">Auto-tune</span><span class="atb-sub">Network error</span>`;
    console.error('Auto-tune failed:', e);
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
      recomputeCurrentFramePrediction();
    });
  }

  init();
});
