'use strict';

const cameraEl = document.getElementById('camera');
const canvasEl = document.getElementById('work-canvas');

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
  location: null,
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

  speedLimitEl.textContent = String(state.speedLimit);
  weatherEl.textContent = `${data.weather.summary}, ${data.weather.temp_c.toFixed(1)} C`;
}

function classifyFrameLocally() {
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

  return {
    label,
    confidence,
  };
}

async function updateRecommendation() {
  if (cameraEl.readyState < 2) {
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

async function start() {
  renderCropMode();
  cropToggleEl?.addEventListener('click', () => {
    state.useCrop = !state.useCrop;
    renderCropMode();
  });

  await setupCamera();
  await fetchContext();

  await updateRecommendation();
  setInterval(updateRecommendation, 1200);
  setInterval(fetchContext, 45000);
}

start().catch((error) => {
  console.error(`Startup error: ${error.message}`);
});
