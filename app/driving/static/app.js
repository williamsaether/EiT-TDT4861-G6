'use strict';

const cameraEl = document.getElementById('camera');
const canvasEl = document.getElementById('work-canvas');

const speedLimitSignEl = document.getElementById('speed-limit-sign');
const weatherEl = document.getElementById('weather');
const recommendedEl = document.getElementById('recommended');
const classificationEl = document.getElementById('classification');
const cropToggleEl = document.getElementById('crop-toggle');
const cropOverlayEl = document.getElementById('model-crop-overlay');

const ctx = canvasEl.getContext('2d', { willReadFrequently: true });
const MODEL_CROP = { top: 0.60, bottom: 0.95, left: 0.35, right: 0.65 };

const state = {
  speedLimit: 60,
  weather: { temp_c: 1, precip_mm_h: 0, humidity: 80, summary: 'clear' },
  location: null,
  useCrop: true,
  classifier: null,
  smoothWindowSec: 3.0,
  scoreHistory: [],
};

const SUPPORTED_SPEED_SIGNS = new Set([30, 40, 50, 60, 70, 80, 90, 100, 110]);

function speedSignPath(speedLimit) {
  const speed = SUPPORTED_SPEED_SIGNS.has(speedLimit) ? speedLimit : 60;
  return `/driving/static/speed-limit-signs/362_${speed}.webp`;
}

function renderCropMode() {
  if (cropToggleEl) {
    cropToggleEl.textContent = `Crop: ${state.useCrop ? 'on' : 'off'}`;
  }
  if (cropOverlayEl) {
    cropOverlayEl.style.display = state.useCrop ? 'block' : 'none';
  }
}

function normalizeScores(scores) {
  if (!scores || typeof scores !== 'object') return {};
  let total = 0;
  const out = {};
  for (const [k, v] of Object.entries(scores)) {
    const n = Number(v);
    if (!Number.isFinite(n) || n <= 0) continue;
    out[k] = n;
    total += n;
  }
  if (total <= 0) return {};
  for (const k of Object.keys(out)) {
    out[k] /= total;
  }
  return out;
}

function applyTemporalSmoothing(scores) {
  const normalized = normalizeScores(scores);
  if (!Object.keys(normalized).length) {
    return normalized;
  }

  const now = Date.now();
  state.scoreHistory.push({ ts: now, scores: normalized });

  const windowMs = Math.max(100, Math.round(state.smoothWindowSec * 1000));
  const minTs = now - windowMs;
  state.scoreHistory = state.scoreHistory.filter((x) => x.ts >= minTs);

  const aggregate = {};
  let n = 0;
  for (const item of state.scoreHistory) {
    n += 1;
    for (const [label, value] of Object.entries(item.scores)) {
      aggregate[label] = (aggregate[label] || 0) + value;
    }
  }
  if (n <= 0) return normalized;

  for (const label of Object.keys(aggregate)) {
    aggregate[label] /= n;
  }
  return normalizeScores(aggregate);
}

function pickTopClass(scores) {
  let bestLabel = 'unknown';
  let bestScore = 0;
  for (const [label, score] of Object.entries(scores || {})) {
    if (score > bestScore) {
      bestScore = score;
      bestLabel = label;
    }
  }
  return { label: bestLabel, confidence: bestScore };
}

async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    cameraEl.srcObject = stream;
    await cameraEl.play();
  } catch (error) {
    console.error(`Camera error: ${error.message}`);
    throw error;
  }
}

async function setupClassifier() {
  const res = await fetch('/driving/api/models');
  const data = await res.json();
  if (!data.ok || !Array.isArray(data.models) || data.models.length === 0) {
    throw new Error('No ONNX models available');
  }

  const modelName = data.current || data.models[0];
  state.classifier = new window.DrivingOnnxClassifier({
    modelUrl: `/driving/models/${encodeURIComponent(modelName)}`,
    classNames: data.class_names || [],
    excludedClasses: data.excluded_classes || [],
    imageSize: data.image_size || 224,
  });
  const smoothSec = Number(data.smooth_window_sec);
  state.smoothWindowSec = Number.isFinite(smoothSec) && smoothSec > 0 ? smoothSec : 3.0;
  state.scoreHistory = [];
  await state.classifier.init();
}

function getLocation() {
  return new Promise((resolve) => {
    if (!navigator.geolocation) {
      resolve(null);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
      () => resolve(null),
      { enableHighAccuracy: true, timeout: 2500, maximumAge: 30000 }
    );
  });
}

async function fetchContext() {
  const location = await getLocation();
  const payload = location || {};

  const res = await fetch('/driving/api/context', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();

  if (!data.ok) {
    throw new Error('Failed to load context');
  }

  state.location = data.location;
  state.speedLimit = data.speed_limit;
  state.weather = data.weather;

  if (speedLimitSignEl) {
    speedLimitSignEl.src = speedSignPath(state.speedLimit);
    speedLimitSignEl.alt = `Speed limit ${state.speedLimit} km/h`;
  }
  weatherEl.textContent = `${data.weather.summary}, ${data.weather.temp_c.toFixed(1)} C`;
}

async function classifyFrameLocally() {
  if (!state.classifier || !state.classifier.ready) {
    return null;
  }

  const vw = cameraEl.videoWidth || 0;
  const vh = cameraEl.videoHeight || 0;
  if (state.useCrop && vw > 0 && vh > 0) {
    const sx = Math.round(vw * MODEL_CROP.left);
    const sy = Math.round(vh * MODEL_CROP.top);
    const sw = Math.round(vw * (MODEL_CROP.right - MODEL_CROP.left));
    const sh = Math.round(vh * (MODEL_CROP.bottom - MODEL_CROP.top));
    ctx.drawImage(cameraEl, sx, sy, sw, sh, 0, 0, canvasEl.width, canvasEl.height);
  } else {
    ctx.drawImage(cameraEl, 0, 0, canvasEl.width, canvasEl.height);
  }

  const imageData = ctx.getImageData(0, 0, canvasEl.width, canvasEl.height);
  const raw = await state.classifier.classify(imageData);
  const smoothedScores = applyTemporalSmoothing(raw.scores || {});
  const top = pickTopClass(smoothedScores);
  return {
    label: top.label,
    confidence: Number(top.confidence.toFixed(4)),
    scores: smoothedScores,
  };
}

async function updateRecommendation() {
  if (cameraEl.readyState < 2) {
    return;
  }

  const classification = await classifyFrameLocally();
  if (!classification) {
    classificationEl.textContent = 'loading model...';
    return;
  }

  classificationEl.textContent = `${classification.label} (${classification.confidence.toFixed(2)})`;

  const res = await fetch('/driving/api/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      speed_limit: state.speedLimit,
      classification,
      weather: state.weather,
    }),
  });

  const data = await res.json();
  if (!data.ok) {
    return;
  }

  recommendedEl.textContent = String(data.recommended_speed);
}

async function start() {
  renderCropMode();
  cropToggleEl?.addEventListener('click', () => {
    state.useCrop = !state.useCrop;
    state.scoreHistory = [];
    renderCropMode();
  });

  await setupCamera();
  await setupClassifier();
  await fetchContext();

  await updateRecommendation();
  setInterval(updateRecommendation, 1200);
  setInterval(fetchContext, 45000);
}

start().catch((error) => {
  console.error(`Startup error: ${error.message}`);
});
