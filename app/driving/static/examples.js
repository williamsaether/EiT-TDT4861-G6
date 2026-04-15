'use strict';

const videoEl = document.getElementById('example-video');
const canvasEl = document.getElementById('work-canvas');
const selectEl = document.getElementById('example-select');

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
  currentExample: null,
  examples: [],
  useCrop: true,
  classifier: null,
};

const SUPPORTED_SPEED_SIGNS = new Set([30, 40, 50, 60, 70, 80, 90, 100, 110]);

function speedSignPath(speedLimit) {
  const speed = SUPPORTED_SPEED_SIGNS.has(speedLimit) ? speedLimit : 60;
  return `/driving/static/speed-limit-signs/362_${speed}.webp`;
}

function weatherFromMetadata(example) {
  return {
    temp_c: Number(example.temp ?? 1.0),
    humidity: Number(example.humidity ?? 80.0),
    precip_mm_h: Number(example.precipitation ?? 0.0),
    summary: 'metadata',
    source: 'metadata',
  };
}

function renderCropMode() {
  if (cropToggleEl) {
    cropToggleEl.textContent = `Crop: ${state.useCrop ? 'on' : 'off'}`;
  }
  if (cropOverlayEl) {
    cropOverlayEl.style.display = state.useCrop ? 'block' : 'none';
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
  await state.classifier.init();
}

async function classifyFrameLocally() {
  if (!state.classifier || !state.classifier.ready) {
    return null;
  }

  const vw = videoEl.videoWidth || 0;
  const vh = videoEl.videoHeight || 0;
  if (state.useCrop && vw > 0 && vh > 0) {
    const sx = Math.round(vw * MODEL_CROP.left);
    const sy = Math.round(vh * MODEL_CROP.top);
    const sw = Math.round(vw * (MODEL_CROP.right - MODEL_CROP.left));
    const sh = Math.round(vh * (MODEL_CROP.bottom - MODEL_CROP.top));
    ctx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, canvasEl.width, canvasEl.height);
  } else {
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
  }

  const imageData = ctx.getImageData(0, 0, canvasEl.width, canvasEl.height);
  return state.classifier.classify(imageData);
}

async function fetchContextForExample(example) {
  const res = await fetch('/driving/api/context', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lat: example.lat, lon: example.lon }),
  });
  const data = await res.json();
  if (!data.ok) {
    throw new Error('Failed to load context');
  }

  state.speedLimit = data.speed_limit;
  state.weather = weatherFromMetadata(example);

  if (speedLimitSignEl) {
    speedLimitSignEl.src = speedSignPath(state.speedLimit);
    speedLimitSignEl.alt = `Speed limit ${state.speedLimit} km/h`;
  }
  weatherEl.textContent = `Temp: ${state.weather.temp_c.toFixed(1)} C | Humidity: ${state.weather.humidity.toFixed(0)}% | Precip: ${state.weather.precip_mm_h.toFixed(1)} mm/h`;
}

async function updateRecommendation() {
  if (videoEl.readyState < 2 || videoEl.paused || videoEl.ended) {
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

function renderExampleOptions() {
  selectEl.innerHTML = '';
  for (const ex of state.examples) {
    const opt = document.createElement('option');
    opt.value = ex.video;
    opt.textContent = `${ex.video} (${ex.difficulty})`;
    selectEl.appendChild(opt);
  }
}

async function setExample(videoName) {
  const example = state.examples.find((e) => e.video === videoName);
  if (!example) return;

  state.currentExample = example;
  videoEl.src = `/driving/videos/${example.video}`;
  await fetchContextForExample(example);
  recommendedEl.textContent = '--';
  classificationEl.textContent = '--';
}

async function loadExamples() {
  const res = await fetch('/driving/api/examples');
  const data = await res.json();
  if (!data.ok) {
    throw new Error('Failed to load examples');
  }

  state.examples = data.examples;
  if (state.examples.length === 0) {
    throw new Error('No non-training videos available in metadata.json');
  }

  renderExampleOptions();
  await setExample(state.examples[0].video);
}

async function start() {
  renderCropMode();
  cropToggleEl?.addEventListener('click', () => {
    state.useCrop = !state.useCrop;
    renderCropMode();
  });

  await setupClassifier();
  await loadExamples();

  selectEl.addEventListener('change', async (e) => {
    await setExample(e.target.value);
  });

  videoEl.addEventListener('play', () => {
    updateRecommendation();
  });

  setInterval(updateRecommendation, 1200);
}

start().catch((error) => {
  console.error(error.message);
});
