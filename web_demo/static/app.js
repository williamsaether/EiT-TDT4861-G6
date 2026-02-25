// ── Element refs ─────────────────────────────────────────────────────────────
const uploadBtn          = document.getElementById('uploadBtn');
const videoInput         = document.getElementById('videoInput');
const uploadStatusEl     = document.getElementById('uploadStatus');
const videoPreview       = document.getElementById('videoPreview');

const latInput           = document.getElementById('latInput');
const lonInput           = document.getElementById('lonInput');
const altInput           = document.getElementById('altInput');
const maxSpeedInput      = document.getElementById('maxSpeedInput');

const weightWeather      = document.getElementById('weightWeather');
const weightCamera       = document.getElementById('weightCamera');
const weightConfidence   = document.getElementById('weightConfidence');
const weightWeatherValue = document.getElementById('weightWeatherValue');
const weightCameraValue  = document.getElementById('weightCameraValue');
const weightConfidenceValue = document.getElementById('weightConfidenceValue');
const weightsStatusEl    = document.getElementById('weightsStatus');
const weightsPreviewEl   = document.getElementById('weightsPreview');

const startBtn           = document.getElementById('startBtn');
const stopBtn            = document.getElementById('stopBtn');
const resetHistogramBtn  = document.getElementById('resetHistogramBtn');

const runStateEl             = document.getElementById('runState');
const modelStateEl           = document.getElementById('modelState');
const legalSpeedEl           = document.getElementById('legalSpeed');
const userMaxSpeedEl         = document.getElementById('userMaxSpeed');
const recommendedSpeedEl     = document.getElementById('recommendedSpeed');
const recommendationReasonEl = document.getElementById('recommendationReason');
const weatherOnlySpeedEl     = document.getElementById('weatherOnlySpeed');
const cameraOnlySpeedEl      = document.getElementById('cameraOnlySpeed');
const weatherFactorEl        = document.getElementById('weatherFactor');
const cameraFactorEl         = document.getElementById('cameraFactor');
const combinedFactorEl       = document.getElementById('combinedFactor');
const combinedLabelEl        = document.getElementById('combinedLabel');
const factorChartLegendEl    = document.getElementById('factorChartLegend');
const cameraGroupsEl         = document.getElementById('cameraGroups');
const sourceInfoEl           = document.getElementById('sourceInfo');
const cameraWindowSlider     = document.getElementById('cameraWindowSlider');
const cameraWindowValue      = document.getElementById('cameraWindowValue');
const windowSecondsLabel     = document.getElementById('windowSecondsLabel');
const windowFrameCountEl     = document.getElementById('windowFrameCount');

let hasInitializedWeights = false;
let hasInitializedWindow  = false;
let isWeightEditing = false;
let isMaxSpeedEditing = false;
let isWindowEditing = false;
let weightsDebounceTimer = null;
let maxSpeedDebounceTimer = null;
let windowDebounceTimer = null;

// ── Helpers ───────────────────────────────────────────────────────────────────
function debounce(fn, delay, timerRefSetter) {
  return (...args) => {
    const current = timerRefSetter();
    if (current) clearTimeout(current);
    const t = setTimeout(() => fn(...args), delay);
    timerRefSetter(t);
  };
}

function pct(v) { return `${(Number(v) * 100).toFixed(1)}%`; }
function fmt(v, digits = 2) { return Number(v).toFixed(digits); }

function renderList(target, items) {
  target.innerHTML = '';
  if (!items || items.length === 0) {
    const li = document.createElement('li');
    li.textContent = 'No data';
    target.appendChild(li);
    return;
  }
  items.forEach(item => {
    const li = document.createElement('li');
    if (Array.isArray(item)) {
      const [label, value] = item;
      const numeric = Number(value);
      if (Number.isFinite(numeric)) {
        li.textContent = numeric >= 0 && numeric <= 1
          ? `${label}: ${(numeric * 100).toFixed(1)}%`
          : `${label}: ${numeric}`;
      } else {
        li.textContent = `${label}: ${value}`;
      }
    } else {
      li.textContent = JSON.stringify(item);
    }
    target.appendChild(li);
  });
}

// ── Distribution charts (accumulated probability bars) ────────────────────────
function createBarChart(canvasId, color) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{ label: 'Accumulated probability', data: [], backgroundColor: color, borderRadius: 6 }],
    },
    options: {
      responsive: true,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          ticks: { callback: v => `${Math.round(v * 100)}%`, color: '#475569' },
          grid: { color: '#e7edf5' },
        },
        x: { ticks: { color: '#475569' }, grid: { display: false } },
      },
    },
  });
}

const charts = {
  friction: createBarChart('frictionChart', '#d56f3e'),
  surface:  createBarChart('surfaceChart',  '#3b82f6'),
  winter:   createBarChart('winterChart',   '#6366f1'),
};

function updateDistributionCharts(cameraDist) {
  ['friction', 'surface', 'winter'].forEach(cat => {
    const chart = charts[cat];
    const entries = (cameraDist && cameraDist[cat]) || [];
    chart.data.labels = entries.map(([label]) => label);
    chart.data.datasets[0].data = entries.map(([, p]) => Number(p));
    chart.update();
  });
}

// ── Factor donut chart ────────────────────────────────────────────────────────
const factorDonutCtx = document.getElementById('factorDonut').getContext('2d');
const factorDonut = new Chart(factorDonutCtx, {
  type: 'doughnut',
  data: {
    labels: ['Weather reduction', 'Camera reduction', 'Unreduced speed'],
    datasets: [{
      data: [0, 0, 100],
      backgroundColor: ['#e8703a', '#3b82f6', '#d1fae5'],
      borderColor: ['#c75b27', '#1d4ed8', '#6ee7b7'],
      borderWidth: 1.5,
      hoverOffset: 6,
    }],
  },
  options: {
    responsive: true,
    animation: { duration: 300 },
    cutout: '62%',
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => ` ${ctx.label}: ${fmt(ctx.parsed, 1)} km/h`,
        },
      },
    },
  },
});

function updateFactorDonut(weatherKmh, cameraKmh, effectiveMax) {
  const unreduced = Math.max(0, effectiveMax - weatherKmh - cameraKmh);
  factorDonut.data.datasets[0].data = [
    Math.max(0, weatherKmh),
    Math.max(0, cameraKmh),
    unreduced,
  ];
  factorDonut.update();
  factorChartLegendEl.innerHTML =
    `<span class="legend-dot weather-dot"></span>Weather: <b>${fmt(weatherKmh,1)} km/h</b> &nbsp;` +
    `<span class="legend-dot camera-dot"></span>Camera: <b>${fmt(cameraKmh,1)} km/h</b> &nbsp;` +
    `<span class="legend-dot ok-dot"></span>Unreduced: <b>${fmt(unreduced,1)} km/h</b>`;
}

// ── Weight sliders ─────────────────────────────────────────────────────────────
function currentRawWeights() {
  return {
    weather:    Number(weightWeather.value    || 0),
    camera:     Number(weightCamera.value     || 0),
    confidence: Number(weightConfidence.value || 0),
  };
}

function refreshSliderValueLabels() {
  weightWeatherValue.textContent    = weightWeather.value;
  weightCameraValue.textContent     = weightCamera.value;
  weightConfidenceValue.textContent = weightConfidence.value;
}

function updateWeightsPreview() {
  refreshSliderValueLabels();
  const raw = currentRawWeights();
  const blendTotal = raw.weather + raw.camera;
  if (blendTotal <= 0) {
    weightsPreviewEl.textContent = 'Set weather or camera above 0.';
    return;
  }
  const ww = raw.weather / blendTotal;
  const wc = raw.camera  / blendTotal;
  const cs = (raw.weather + raw.camera + raw.confidence) > 0
    ? raw.confidence / (raw.weather + raw.camera + raw.confidence)
    : 0;
  weightsPreviewEl.textContent =
    `Blend → weather: ${(ww*100).toFixed(0)}%  camera: ${(wc*100).toFixed(0)}%  |  conf. sensitivity: ${(cs*100).toFixed(0)}%`;
}

async function applyWeightsLive() {
  const payload = currentRawWeights();
  const res = await fetch('/api/weights', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    weightsStatusEl.textContent = data.error || 'Could not apply weights.';
    return;
  }
  if (data.weight_inputs) {
    weightWeather.value    = Number(data.weight_inputs.weather).toFixed(0);
    weightCamera.value     = Number(data.weight_inputs.camera).toFixed(0);
    weightConfidence.value = Number(data.weight_inputs.confidence).toFixed(0);
    refreshSliderValueLabels();
  }
  weightsStatusEl.textContent =
    `Applied — weather: ${fmt(data.weights.weather)}, camera: ${fmt(data.weights.camera)}, conf.sens: ${fmt(data.weights.confidence)}`;
  updateWeightsPreview();
}

async function applyMaxSpeedLive() {
  const raw = maxSpeedInput.value.trim();
  const payload = { max_speed_limit: raw === '' ? null : Number(raw) };
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    recommendationReasonEl.textContent = data.error || 'Could not apply custom max speed.';
    return;
  }
  const maxVal = data?.config?.user_max_speed_limit;
  userMaxSpeedEl.textContent = maxVal ? `${maxVal} km/h` : '-';
}

const debouncedWeightsApply = debounce(applyWeightsLive, 180, v => {
  if (v !== undefined) weightsDebounceTimer = v;
  return weightsDebounceTimer;
});
const debouncedMaxSpeedApply = debounce(applyMaxSpeedLive, 250, v => {
  if (v !== undefined) maxSpeedDebounceTimer = v;
  return maxSpeedDebounceTimer;
});

[weightWeather, weightCamera, weightConfidence].forEach(el => {
  el.addEventListener('pointerdown', () => { isWeightEditing = true; });
  el.addEventListener('pointerup',   () => { isWeightEditing = false; });
  el.addEventListener('input', () => {
    updateWeightsPreview();
    debouncedWeightsApply();
  });
  el.addEventListener('change', () => {
    isWeightEditing = false;
    applyWeightsLive();
  });
});

maxSpeedInput.addEventListener('focus', () => { isMaxSpeedEditing = true; });
maxSpeedInput.addEventListener('blur',  () => { isMaxSpeedEditing = false; applyMaxSpeedLive(); });
maxSpeedInput.addEventListener('input', () => { debouncedMaxSpeedApply(); });

// ── Camera window slider ───────────────────────────────────────────────────────
function refreshWindowLabel() {
  const v = Number(cameraWindowSlider.value).toFixed(1);
  cameraWindowValue.textContent  = v;
  windowSecondsLabel.textContent = v;
}

async function applyCameraWindow() {
  const seconds = Number(cameraWindowSlider.value);
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ camera_window_seconds: seconds }),
  });
  if (!res.ok) {
    const d = await res.json();
    windowFrameCountEl.textContent = d.error || 'Could not apply window.';
  }
}

const debouncedWindowApply = debounce(applyCameraWindow, 200, v => {
  if (v !== undefined) windowDebounceTimer = v;
  return windowDebounceTimer;
});

cameraWindowSlider.addEventListener('pointerdown', () => { isWindowEditing = true; });
cameraWindowSlider.addEventListener('pointerup',   () => { isWindowEditing = false; });
cameraWindowSlider.addEventListener('input', () => {
  refreshWindowLabel();
  debouncedWindowApply();
});
cameraWindowSlider.addEventListener('change', () => {
  isWindowEditing = false;
  applyCameraWindow();
});

// ── Upload & pipeline controls ─────────────────────────────────────────────────
async function uploadVideo() {
  if (!videoInput.files || videoInput.files.length === 0) {
    uploadStatusEl.textContent = 'Select a video file first.';
    return;
  }
  const form = new FormData();
  form.append('video', videoInput.files[0]);
  uploadStatusEl.textContent = 'Uploading...';
  const res  = await fetch('/api/upload-video', { method: 'POST', body: form });
  const data = await res.json();
  if (!res.ok) { uploadStatusEl.textContent = data.error || 'Upload failed.'; return; }
  uploadStatusEl.textContent = `Uploaded: ${data.filename}`;
  videoPreview.src = data.video_url;
}

async function startPipeline() {
  const rawMax = maxSpeedInput.value.trim();
  const payload = {
    lat:             Number(latInput.value),
    lon:             Number(lonInput.value),
    altitude:        Number(altInput.value),
    max_speed_limit: rawMax === '' ? null : Number(rawMax),
  };
  const res  = await fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) { recommendationReasonEl.textContent = data.error || 'Failed to start pipeline.'; return; }
  runStateEl.textContent = data.status;
}

async function stopPipeline() { await fetch('/api/stop', { method: 'POST' }); }
async function resetHistogram() { await fetch('/api/reset-histogram', { method: 'POST' }); }

// ── Main render ────────────────────────────────────────────────────────────────
function renderState(data) {
  runStateEl.textContent   = data.running ? 'running' : 'stopped';
  modelStateEl.textContent = data.model_status || '-';

  if (data.video && data.video.url && !videoPreview.src) {
    videoPreview.src = data.video.url;
  }

  const latest         = data.latest_packet || {};
  const recommendation = latest.recommendation || {};
  const features       = latest.features || {};

  // Speed summary
  legalSpeedEl.textContent       = recommendation.legal_speed_limit_kmh
    ? `${recommendation.legal_speed_limit_kmh} km/h` : '-';
  userMaxSpeedEl.textContent     = recommendation.user_max_speed_limit_kmh
    ? `${recommendation.user_max_speed_limit_kmh} km/h` : '-';
  recommendedSpeedEl.textContent = recommendation.recommended_speed_limit_kmh != null
    ? `${recommendation.recommended_speed_limit_kmh} km/h` : '-';
  recommendationReasonEl.textContent = recommendation.reason || 'Waiting for data...';

  // Per-factor breakdown
  const effMax = recommendation.effective_max_speed_limit_kmh || 80;
  const wOnly  = recommendation.weather_only_kmh;
  const cOnly  = recommendation.camera_only_kmh;
  const wF     = features.weather_factor;
  const cF     = features.effective_camera_factor;
  const combo  = features.combined_factor;

  weatherOnlySpeedEl.textContent = wOnly != null ? `${wOnly} km/h` : '-';
  cameraOnlySpeedEl.textContent  = cOnly != null ? `${cOnly} km/h` : '-';
  weatherFactorEl.textContent    = wF  != null ? `factor: ${fmt(wF)}` : 'factor: -';
  cameraFactorEl.textContent     = cF  != null ? `factor: ${fmt(cF)}` : 'factor: -';
  combinedFactorEl.textContent   = combo != null ? fmt(combo) : '-';
  combinedLabelEl.textContent    = 'combined factor';

  // Factor donut chart
  const contrib = recommendation.factor_contributions || {};
  updateFactorDonut(
    contrib.weather_kmh ?? 0,
    contrib.camera_kmh  ?? 0,
    effMax,
  );

  // Camera window slider sync
  if (!hasInitializedWindow || !isWindowEditing) {
    if (data.camera_window_seconds != null) {
      cameraWindowSlider.value = data.camera_window_seconds;
      refreshWindowLabel();
      hasInitializedWindow = true;
    }
  }

  // Windowed camera info (frame count)
  const windowed = data.windowed_camera;
  if (windowed) {
    windowFrameCountEl.textContent =
      `${windowed.frame_count} frame${windowed.frame_count !== 1 ? 's' : ''} in window — ` +
      `top: ${windowed.top_label.replace(/_/g, ' ')} (${(windowed.confidence * 100).toFixed(1)}%)`;
  } else {
    windowFrameCountEl.textContent = 'No frames in window yet.';
  }

  // Camera top predictions — show windowed averaged result
  if (windowed && windowed.top_3 && windowed.top_3.length > 0) {
    const items = windowed.top_3.map(([label, prob]) => [label.replace(/_/g, ' '), prob]);
    renderList(cameraGroupsEl, items);
  } else {
    renderList(cameraGroupsEl, []);
  }

  // Weather + speed source info
  const sourceItems = [];
  if (data.latest_weather) {
    sourceItems.push(['weather_status',   data.latest_weather.status]);
    sourceItems.push(['temperature_c',    data.latest_weather.temperature_c]);
    sourceItems.push(['precipitation',    data.latest_weather.precipitation_mm_h + ' mm/h']);
    sourceItems.push(['wind_kmh',         data.latest_weather.wind_speed_kmh]);
    sourceItems.push(['visibility_m',     data.latest_weather.visibility_m]);
  }
  if (data.latest_speed) {
    sourceItems.push(['speed_status', data.latest_speed.status]);
    sourceItems.push(['road',         data.latest_speed.road_reference || '-']);
  }
  renderList(sourceInfoEl, sourceItems);

  // GPS inputs (only update when not being edited)
  if (data.location) {
    latInput.value = Number(data.location.lat).toFixed(4);
    lonInput.value = Number(data.location.lon).toFixed(4);
    altInput.value = Number(data.location.altitude).toFixed(1);
  }

  if (!isMaxSpeedEditing) {
    maxSpeedInput.value = (data.user_max_speed_limit != null && data.user_max_speed_limit !== undefined)
      ? String(data.user_max_speed_limit) : '';
  }

  if (data.weight_inputs && (!hasInitializedWeights || !isWeightEditing)) {
    weightWeather.value    = Number(data.weight_inputs.weather).toFixed(0);
    weightCamera.value     = Number(data.weight_inputs.camera).toFixed(0);
    weightConfidence.value = Number(data.weight_inputs.confidence).toFixed(0);
    hasInitializedWeights  = true;
    updateWeightsPreview();
  }

  updateDistributionCharts(data.camera_distribution || {});
}

// ── Event bindings ─────────────────────────────────────────────────────────────
uploadBtn.addEventListener('click', uploadVideo);
startBtn.addEventListener('click', startPipeline);
stopBtn.addEventListener('click', stopPipeline);
resetHistogramBtn.addEventListener('click', resetHistogram);

// ── Poll loop ──────────────────────────────────────────────────────────────────
async function refreshStatus() {
  const res = await fetch('/api/status');
  if (!res.ok) return;
  renderState(await res.json());
}

updateWeightsPreview();
refreshStatus();
setInterval(refreshStatus, 1000);
