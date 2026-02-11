const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const intervalInput = document.getElementById('interval');
const cameraSelect = document.getElementById('cameraSelect');
const fileInput = document.getElementById('fileInput');
const sourceRadios = document.querySelectorAll('input[name="source"]');
const cameraControls = document.getElementById('cameraControls');
const fileControls = document.getElementById('fileControls');

const frictionEl = document.getElementById('friction');
const surfaceEl = document.getElementById('surface');
const unevenEl = document.getElementById('uneven');
const winterEl = document.getElementById('winter');
const rawEl = document.getElementById('raw');

let stream = null;
let timerId = null;
let cameraReady = false;
let currentSource = 'camera'; // 'camera' or 'file'

function renderList(el, items) {
  el.innerHTML = '';
  items.forEach(([label, score]) => {
    const li = document.createElement('li');
    li.textContent = `${label}: ${(score * 100).toFixed(1)}%`;
    el.appendChild(li);
  });
}

async function captureAndSend() {
  // Only process if video is playing and has valid data
  if (video.paused || video.ended || video.readyState < 2) return;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
  const form = new FormData();
  form.append('image', blob, 'frame.jpg');

  try {
    const res = await fetch('/predict', { method: 'POST', body: form });
    if (!res.ok) return;
    const data = await res.json();

    renderList(frictionEl, data.friction || []);
    renderList(surfaceEl, data.surface || []);
    renderList(unevenEl, data.uneven || []);
    renderList(winterEl, data.winter || []);
    renderList(rawEl, data.raw_top || []);
  } catch (err) {
    console.error("Prediction error:", err);
  }
}

function setSelectOptions(select, options) {
  const current = select.value;
  select.innerHTML = '';
  options.forEach(opt => {
    const option = document.createElement('option');
    option.value = opt.value;
    option.textContent = opt.label;
    select.appendChild(option);
  });
  if (current && options.some(opt => opt.value === current)) {
    select.value = current;
  }
}

async function refreshCameraList() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter(device => device.kind === 'videoinput');
  const options = cams.map((cam, idx) => ({
    value: cam.deviceId,
    label: cam.label || `Camera ${idx + 1}`,
  }));
  if (options.length === 0) {
    setSelectOptions(cameraSelect, [{ value: '', label: 'No camera found' }]);
    cameraSelect.disabled = true;
    return;
  }
  cameraSelect.disabled = false;
  setSelectOptions(cameraSelect, options);
  if (!cameraSelect.value && options[0]) {
    cameraSelect.value = options[0].value;
  }
}

async function ensureCameraReady() {
  if (cameraReady) return;
  try {
    const temp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    temp.getTracks().forEach(t => t.stop());
    cameraReady = true;
    await refreshCameraList();
  } catch (e) {
    console.warn("Camera permission denied or unavailable", e);
  }
}

async function getCameraStream() {
  const deviceId = cameraSelect.value;
  const constraints = deviceId
    ? { video: { deviceId: { exact: deviceId } }, audio: false }
    : { video: true, audio: false };
  return navigator.mediaDevices.getUserMedia(constraints);
}

function stopProcessing() {
  if (timerId) {
    clearInterval(timerId);
    timerId = null;
  }
}

function startProcessing() {
  stopProcessing();
  const interval = Math.max(100, parseInt(intervalInput.value || '500', 10));
  timerId = setInterval(captureAndSend, interval);
}

async function startCamera() {
  stopProcessing();
  // Cleanup file src
  if (currentSource === 'file') {
    video.pause();
    video.removeAttribute('src');
    video.load();
  }

  await ensureCameraReady();
  stream = await getCameraStream();
  video.srcObject = stream;
  try {
    await video.play();
  } catch (e) { console.error("Play error", e); }

  startBtn.disabled = true;
  stopBtn.disabled = false;

  startProcessing();
}

function stopCamera() {
  stopProcessing();
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

// Event Listeners
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

cameraSelect.addEventListener('change', async () => {
  if (currentSource === 'camera' && stream) {
    stopCamera();
    await startCamera();
  }
});

fileInput.addEventListener('change', (e) => {
  if (currentSource !== 'file') return;

  const file = e.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = url;
  video.play();
  startProcessing();
});

sourceRadios.forEach(radio => {
  radio.addEventListener('change', (e) => {
    currentSource = e.target.value;
    stopProcessing();

    // Reset video state
    video.pause();
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
      video.srcObject = null;
    }
    video.removeAttribute('src');
    video.load();

    if (currentSource === 'camera') {
      cameraControls.style.display = 'flex';
      fileControls.style.display = 'none';
      startBtn.disabled = false;
      stopBtn.disabled = true;
    } else {
      cameraControls.style.display = 'none';
      fileControls.style.display = 'flex';
      // If file already selected, play it
      if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        video.src = URL.createObjectURL(file);
        video.play();
        startProcessing();
      }
    }
  });
});


if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener('devicechange', refreshCameraList);
}

// Initial setup
(async () => {
  // Try to init camera list on load if possible without triggering permission prompt immediately if not granted
  // But usually we need to ask permission first.
  // We'll leave it to "Start Camera" to ask permission
})();
