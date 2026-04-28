/**
 * PIXEL TRUTH — homepage.js
 * Upload handling, scan animation, result rendering with image preview
 */

(function() {
  const fileInput  = document.getElementById('fileInput');
  const uploadZone = document.getElementById('upload-zone');
  const resultPanel= document.getElementById('result-panel');
  const resultBody = document.getElementById('result-body');
  const quickGuide = document.getElementById('quick-guide');
  const scanLog    = document.getElementById('scan-log');
  const ssTotal    = document.getElementById('ss-total');
  const ssReal     = document.getElementById('ss-real');
  const ssFake     = document.getElementById('ss-fake');
  const sessionUser= document.getElementById('session-user');

  let stats = { total: 0, real: 0, fake: 0 };

  // ── Session username ──────────────────────────────────────────
  const username = getAuthUsername();
  if (sessionUser && username) sessionUser.textContent = username.toUpperCase();

  // ── Upload event handlers ─────────────────────────────────────
  fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) analyzeImage(file);
  });

  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) analyzeImage(file);
  });

  // ── SCAN LINES ────────────────────────────────────────────────
  const SCAN_LINES = [
    'Buffering image data...',
    'Extracting DCT frequency map...',
    'Running FREQ SENTINEL...',
    'Running PATCH ORACLE...',
    'Running TEXTURE JUDGE...',
    'Aggregating ensemble votes...',
    'Calibrating confidence scores...',
  ];

  function analyzeImage(file) {
    if (quickGuide) quickGuide.style.display = 'none';
    resultPanel.style.display = 'block';
    resultBody.innerHTML = '';

    // ── Image preview ────────────────────────────────────────────
    const previewWrap = document.createElement('div');
    previewWrap.className = 'image-preview-wrap';

    const objectURL = URL.createObjectURL(file);
    const img = document.createElement('img');
    img.src = objectURL;
    img.className = 'preview-img';
    img.alt = file.name;

    const imgLabel = document.createElement('div');
    imgLabel.className = 'preview-label';
    imgLabel.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB`;

    previewWrap.appendChild(img);
    previewWrap.appendChild(imgLabel);
    resultBody.appendChild(previewWrap);

    // ── Scan log lines ───────────────────────────────────────────
    const logWrap = document.createElement('div');
    logWrap.className = 'scan-log-lines';
    resultBody.appendChild(logWrap);

    let i = 0;
    const interval = setInterval(async () => {
      if (i < SCAN_LINES.length) {
        const line = document.createElement('div');
        line.textContent = SCAN_LINES[i];
        logWrap.appendChild(line);
        i++;
      } else {
        clearInterval(interval);
        await callAPI(file, objectURL);
      }
    }, 300);
  }

  async function callAPI(file, objectURL) {
    const formData = new FormData();
    formData.append('image', file);

    const headers = {};
    const token = getAuthToken();
    if (token) headers['Authorization'] = 'Bearer ' + token;

    try {
      const res  = await fetch('/api/detect', { method: 'POST', body: formData, headers });
      const data = await res.json();

      if (res.ok && data.success) {
        renderVerdict(data, file, objectURL);
      } else {
        renderError(data.error || 'Detection failed', file);
      }
    } catch(err) {
      renderError('Cannot reach backend — ensure Flask server is running on port 5000', file);
    }
  }

  function renderVerdict(data, file, objectURL) {
    const isReal   = (data.verdict || '').toUpperCase() === 'REAL';
    const conf     = data.confidence || 'N/A';
    const method   = (data.method || 'unknown').toUpperCase();
    const realProb = data.real_probability != null ? (data.real_probability * 100).toFixed(1) : '—';
    const aiProb   = data.ai_probability   != null ? (data.ai_probability   * 100).toFixed(1) : '—';

    const cls     = isReal ? 'real-verdict' : 'fake-verdict';
    const label   = isReal ? 'AUTHENTIC'    : 'SYNTHETIC';
    const icon    = isReal ? '✦'            : '◈';
    const textCls = isReal ? 'real-text'    : 'fake-text';

    const spatial   = data.models?.spatial;
    const frequency = data.models?.frequency;

    const modelsHtml = (spatial || frequency) ? `
      <div class="verdict-models">
        ${spatial   ? `<div class="vm-item">SPATIAL <strong>${(spatial.real_probability * 100).toFixed(1)}% real</strong></div>` : ''}
        ${frequency ? `<div class="vm-item">FREQUENCY <strong>${(frequency.real_probability * 100).toFixed(1)}% real</strong></div>` : ''}
      </div>` : '';

    // ── Replace scan log lines with verdict ──────────────────────
    const logWrap = resultBody.querySelector('.scan-log-lines');
    if (logWrap) logWrap.remove();

    const el = document.createElement('div');
    el.className = `verdict-display ${cls}`;
    el.innerHTML = `
      <div class="verdict-top ${textCls}">${icon} VERDICT: ${label}</div>
      <div class="verdict-meta">${method} · ${conf} confidence</div>
      <div class="verdict-probs">
        <div class="vp-item">
          <div class="vp-label">REAL PROBABILITY</div>
          <div class="vp-num green">${realProb}%</div>
        </div>
        <div class="vp-item">
          <div class="vp-label">AI PROBABILITY</div>
          <div class="vp-num red">${aiProb}%</div>
        </div>
        <div class="vp-item">
          <div class="vp-label">CONFIDENCE</div>
          <div class="vp-num blue">${conf}</div>
        </div>
      </div>
      ${modelsHtml}
      <div class="verdict-bar-wrap">
        <div class="verdict-bar">
          <div class="verdict-bar-fill ${isReal ? 'bar-real' : 'bar-fake'}"
               style="width:${isReal ? realProb : aiProb}%"></div>
        </div>
        <div class="verdict-bar-labels">
          <span class="green">REAL ${realProb}%</span>
          <span class="red">AI ${aiProb}%</span>
        </div>
      </div>`;

    resultBody.appendChild(el);
    addLogEntry(file.name, isReal);
    updateStats(isReal);

    // Release object URL after image is displayed
    if (objectURL) {
      const previewImg = resultBody.querySelector('.preview-img');
      if (!previewImg || previewImg.complete) URL.revokeObjectURL(objectURL);
      else previewImg.onload = () => URL.revokeObjectURL(objectURL);
    }
  }

  function renderError(message, file) {
    const logWrap = resultBody.querySelector('.scan-log-lines');
    if (logWrap) logWrap.remove();

    const el = document.createElement('div');
    el.className = 'error-display';
    el.innerHTML = `⚠ ${message}<br><span style="opacity:0.6;font-size:0.58rem">${file.name}</span>`;
    resultBody.appendChild(el);
  }

  function addLogEntry(filename, isReal) {
    const empty = scanLog.querySelector('.log-empty');
    if (empty) empty.remove();

    const entry = document.createElement('div');
    entry.className = `log-entry ${isReal ? 'real' : 'fake'}`;
    entry.innerHTML = `
      <div class="log-filename">${filename}</div>
      <div class="log-verdict ${isReal ? 'real' : 'fake'}">${isReal ? '✦ REAL' : '◈ FAKE'}</div>`;

    scanLog.insertBefore(entry, scanLog.firstChild);

    const entries = scanLog.querySelectorAll('.log-entry');
    if (entries.length > 20) entries[entries.length - 1].remove();
  }

  function updateStats(isReal) {
    stats.total++;
    if (isReal) stats.real++; else stats.fake++;
    if (ssTotal) animateCount(ssTotal, stats.total, 600);
    if (ssReal)  animateCount(ssReal,  stats.real,  600);
    if (ssFake)  animateCount(ssFake,  stats.fake,  600);
  }

  window.closeResult = function() {
    resultPanel.style.display = 'none';
    if (quickGuide) quickGuide.style.display = '';
    fileInput.value = '';
  };
})();