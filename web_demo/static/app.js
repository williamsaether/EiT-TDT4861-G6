const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const intervalInput = document.getElementById('interval');
const cameraSelect = document.getElementById('cameraSelect');

const frictionEl = document.getElementById('friction');
const surfaceEl = document.getElementById('surface');
const unevenEl = document.getElementById('uneven');
const winterEl = document.getElementById('winter');
const rawEl = document.getElementById('raw');

let stream = null;
let timerId = null;
let cameraReady = false;

function renderList(el, items) {
  el.innerHTML = '';
  items.forEach(([label, score]) => {
    const li = document.createElement('li');
    li.textContent = `${label}: ${(score * 100).toFixed(1)}%`;
    el.appendChild(li);
  });
}

async function captureAndSend() {
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
  const form = new FormData();
  form.append('image', blob, 'frame.jpg');

  const res = await fetch('/predict', { method: 'POST', body: form });
  if (!res.ok) return;
  const data = await res.json();

  renderList(frictionEl, data.friction || []);
  renderList(surfaceEl, data.surface || []);
  renderList(unevenEl, data.uneven || []);
  renderList(winterEl, data.winter || []);
  renderList(rawEl, data.raw_top || []);
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
  const temp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  temp.getTracks().forEach(t => t.stop());
  cameraReady = true;
  await refreshCameraList();
}

async function getCameraStream() {
  const deviceId = cameraSelect.value;
  const constraints = deviceId
    ? { video: { deviceId: { exact: deviceId } }, audio: false }
    : { video: true, audio: false };
  return navigator.mediaDevices.getUserMedia(constraints);
}

async function startCamera() {
  await ensureCameraReady();
  stream = await getCameraStream();
  video.srcObject = stream;
  startBtn.disabled = true;
  stopBtn.disabled = false;

  const interval = Math.max(100, parseInt(intervalInput.value || '500', 10));
  timerId = setInterval(captureAndSend, interval);
}

function stopCamera() {
  if (timerId) {
    clearInterval(timerId);
    timerId = null;
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
cameraSelect.addEventListener('change', async () => {
  if (!stream) return;
  stopCamera();
  await startCamera();
});

if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener('devicechange', refreshCameraList);
}
