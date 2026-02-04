const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const intervalInput = document.getElementById('interval');

const frictionEl = document.getElementById('friction');
const surfaceEl = document.getElementById('surface');
const unevenEl = document.getElementById('uneven');
const winterEl = document.getElementById('winter');
const rawEl = document.getElementById('raw');

let stream = null;
let timerId = null;

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

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
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
