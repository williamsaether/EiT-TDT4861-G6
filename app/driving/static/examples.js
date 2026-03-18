'use strict';

const videoEl = document.getElementById('example-video');
const canvasEl = document.getElementById('work-canvas');
const selectEl = document.getElementById('example-select');

const speedLimitEl = document.getElementById('speed-limit');
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
};

function renderCropMode() {
  if (cropToggleEl) {
    cropToggleEl.textContent = `Crop: ${state.useCrop ? 'on' : 'off'}`;
  }
  if (cropOverlayEl) {
    cropOverlayEl.style.display = state.useCrop ? 'block' : 'none';
  }
}

function classifyFrameLocally() {
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
  const frame = ctx.getImageData(0, 0, canvasEl.width, canvasEl.height).data;

  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let n = 0;

  for (let i = 0; i < frame.length; i += 4 * 8) {
    sumR += frame[i];
    sumG += frame[i + 1];
    sumB += frame[i + 2];
    n += 1;
  }

  const avgR = sumR / n;
  const avgG = sumG / n;
  const avgB = sumB / n;
  const brightness = (avgR + avgG + avgB) / 3;

  let label = 'wet_asphalt';
  let confidence = 0.6;

  if (brightness > 165 && avgB > avgR) {
    label = 'fresh_snow';
    confidence = 0.78;
  } else if (brightness > 145 && avgB - avgR > 14) {
    label = 'ice';
    confidence = 0.72;
  } else if (brightness < 95) {
    label = 'wet_asphalt';
    confidence = 0.69;
  } else if (brightness >= 95 && brightness < 145) {
    label = 'dry_asphalt';
    confidence = 0.64;
  }

  return { label, confidence };
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
  state.weather = data.weather;

  speedLimitEl.textContent = String(data.speed_limit);
  weatherEl.textContent = `${data.weather.summary}, ${data.weather.temp_c.toFixed(1)} C`;
}

async function updateRecommendation() {
  if (videoEl.readyState < 2 || videoEl.paused || videoEl.ended) {
    return;
  }

  const classification = classifyFrameLocally();
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
