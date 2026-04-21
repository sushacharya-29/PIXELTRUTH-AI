/**
 * PIXEL TRUTH — admin.js
 * Admin access gate, live monitoring, stats polling
 */

(function() {
  const gate  = document.getElementById('access-gate');
  const panel = document.getElementById('admin-panel');

  // ── Access gate check ─────────────────────────────────────────
  function checkAdminAccess() {
    const role  = getAuthRole();
    const token = getAuthToken();

    if (token && role && role.toLowerCase() === 'admin') {
      gate.style.display  = 'none';
      panel.style.display = 'block';
      initPanel();
    } else {
      gate.style.display  = 'flex';
      panel.style.display = 'none';
    }
  }
  checkAdminAccess();

  // ── Panel init ────────────────────────────────────────────────
  let startTime   = Date.now();
  let pollInterval= null;
  let activityLog = [];
  let rowCounter  = 0;

  function initPanel() {
    const adminUname = document.getElementById('admin-username');
    if (adminUname) adminUname.textContent = (getAuthUsername() || 'ADMIN').toUpperCase();

    startUptimeClock();
    fetchStats();
    pollInterval = setInterval(fetchStats, 10000); // Poll every 10s
  }

  // ── Uptime clock ──────────────────────────────────────────────
  function startUptimeClock() {
    const el = document.getElementById('uptime-counter');
    if (!el) return;
    setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      const h = String(Math.floor(elapsed / 3600)).padStart(2,'0');
      const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2,'0');
      const s = String(elapsed % 60).padStart(2,'0');
      el.textContent = `${h}:${m}:${s}`;
    }, 1000);
  }

  // ── Fetch stats from API ──────────────────────────────────────
  window.fetchStats = async function() {
    const lastRefreshEl = document.getElementById('last-refresh');
    if (lastRefreshEl) lastRefreshEl.textContent = formatTime(new Date());

    const token = getAuthToken();
    const headers = token ? { 'Authorization': 'Bearer ' + token } : {};

    try {
      const res  = await fetch('/api/admin/stats', { headers });
      const data = await res.json();

      if (res.ok && data.success) {
        updateKPIs(data);
        updateDonut(data);
        updateModelBars(data);
        if (data.recent_scans) updateTable(data.recent_scans);
        if (data.latest_scan)  addActivityItem(data.latest_scan);
      } else {
        // Use demo data if backend unavailable
        useDemoData();
      }
    } catch(err) {
      useDemoData();
    }
  };

  // ── Demo / fallback data ──────────────────────────────────────
  function useDemoData() {
    const demo = {
      total_scans:      Math.floor(Math.random() * 20) + 140,
      correct_predictions: 0,
      real_count:       Math.floor(Math.random() * 10) + 82,
      fake_count:       Math.floor(Math.random() * 5)  + 52,
      model_accuracy: {
        freq_sentinel: 88.4 + (Math.random() * 2),
        patch_oracle:  91.2 + (Math.random() * 2),
        texture_judge: 93.8 + (Math.random() * 2),
        ensemble:      94.7 + (Math.random() * 0.5),
      },
      recent_scans: generateDemoScans(),
    };
    demo.correct_predictions = Math.floor(demo.total_scans * 0.947);
    updateKPIs(demo);
    updateDonut(demo);
    updateModelBars(demo);
    updateTable(demo.recent_scans);
  }

  function generateDemoScans() {
    const files = ['portrait_01.jpg','landscape_02.png','face_gen_03.jpg','photo_04.webp','synthetic_05.jpg','real_img_06.png'];
    return files.map((f, i) => ({
      id: i + 1,
      filename: f,
      verdict: i % 3 === 2 ? 'fake' : 'real',
      real_probability: i % 3 === 2 ? 0.12 : 0.91,
      ai_probability:   i % 3 === 2 ? 0.88 : 0.09,
      confidence: i % 3 === 2 ? 'HIGH' : 'HIGH',
      method: 'ENSEMBLE',
      timestamp: new Date(Date.now() - i * 120000).toISOString(),
    }));
  }

  // ── Update KPI cards ──────────────────────────────────────────
  function updateKPIs(data) {
    const total   = data.total_scans || 0;
    const correct = data.correct_predictions || 0;
    const real    = data.real_count  || 0;
    const fake    = data.fake_count  || 0;
    const accuracy= total > 0 ? ((correct / total) * 100).toFixed(1) : '0.0';

    setKPI('kpi-total',   total,   'bar-total',    100, null);
    setKPI('kpi-correct', correct, 'bar-accuracy', total > 0 ? (correct/total)*100 : 0, 'green');
    setKPI('kpi-real',    real,    'bar-real',      total > 0 ? (real/total)*100 : 0, 'green');
    setKPI('kpi-fake',    fake,    'bar-fake',      total > 0 ? (fake/total)*100 : 0, 'red');

    const fracEl = document.getElementById('kpi-frac');
    if (fracEl) fracEl.textContent = `/ ${total}`;

    const subAcc = document.getElementById('sub-accuracy');
    if (subAcc) subAcc.textContent = `${accuracy}% accuracy rate`;

    const subReal = document.getElementById('sub-real');
    if (subReal && total) subReal.textContent = `${((real/total)*100).toFixed(1)}% of total scans`;

    const subFake = document.getElementById('sub-fake');
    if (subFake && total) subFake.textContent = `${((fake/total)*100).toFixed(1)}% of total scans`;
  }

  function setKPI(numId, value, barId, pct, barClass) {
    const numEl = document.getElementById(numId);
    if (numEl) {
      const countEl = numEl.querySelector('.count-up') || numEl;
      animateCount(countEl, value);
    }
    const barEl = document.getElementById(barId);
    if (barEl) {
      if (barClass) { barEl.className = `kpi-bar ${barClass}`; }
      barEl.style.width = Math.min(pct, 100) + '%';
    }
  }

  // ── Donut chart ───────────────────────────────────────────────
  const CIRCUMFERENCE = 2 * Math.PI * 70; // r=70

  function updateDonut(data) {
    const total = (data.real_count || 0) + (data.fake_count || 0);
    const real  = data.real_count  || 0;
    const fake  = data.fake_count  || 0;

    const realArc = total > 0 ? (real / total) * CIRCUMFERENCE : 0;
    const fakeArc = total > 0 ? (fake / total) * CIRCUMFERENCE : 0;

    const realEl = document.getElementById('donut-real');
    const fakeEl = document.getElementById('donut-fake');
    const numEl  = document.getElementById('donut-center-num');

    if (realEl) realEl.style.strokeDasharray = `${realArc.toFixed(1)} ${CIRCUMFERENCE}`;
    if (fakeEl) {
      fakeEl.style.strokeDasharray  = `${fakeArc.toFixed(1)} ${CIRCUMFERENCE}`;
      fakeEl.style.strokeDashoffset = (110 - realArc).toString();
    }
    if (numEl) animateCount(numEl, total);
  }

  // ── Model performance bars ────────────────────────────────────
  function updateModelBars(data) {
    const acc = data.model_accuracy || {};
    setModelBar('mp-freq',    'mpb-freq',    acc.freq_sentinel);
    setModelBar('mp-patch',   'mpb-patch',   acc.patch_oracle);
    setModelBar('mp-texture', 'mpb-texture', acc.texture_judge);
    setModelBar('mp-ensemble','mpb-ensemble',acc.ensemble);
  }

  function setModelBar(valId, barId, value) {
    const v = value != null ? parseFloat(value).toFixed(1) : null;
    const valEl = document.getElementById(valId);
    const barEl = document.getElementById(barId);
    if (valEl) valEl.textContent = v ? `${v}%` : '—';
    if (barEl) barEl.style.width = v ? `${Math.min(parseFloat(v), 100)}%` : '0%';
  }

  // ── Activity feed ─────────────────────────────────────────────
  function addActivityItem(scan) {
    const feed = document.getElementById('activity-feed');
    if (!feed) return;

    const empty = feed.querySelector('.af-empty');
    if (empty) empty.remove();

    const isReal = (scan.verdict || '').toLowerCase().includes('real');
    const name   = scan.filename || 'unknown';
    const time   = formatTime(new Date());

    const item = document.createElement('div');
    item.className = 'af-item';
    item.innerHTML = `
      <span class="af-dot ${isReal ? 'real' : 'fake'}"></span>
      <span class="af-name">${name}</span>
      <span class="af-verdict ${isReal ? 'real' : 'fake'}">${isReal ? 'REAL' : 'FAKE'}</span>
      <span class="af-time">${time}</span>`;

    feed.insertBefore(item, feed.firstChild);
    if (feed.children.length > 12) feed.lastChild.remove();

    activityLog.unshift({ name, isReal, time });
    if (activityLog.length > 50) activityLog.pop();
  }

  // ── Scan table ────────────────────────────────────────────────
  function updateTable(scans) {
    const tbody     = document.getElementById('scan-table-body');
    const countEl   = document.getElementById('table-count');
    if (!tbody || !scans) return;

    if (countEl) countEl.textContent = `${scans.length} records`;
    tbody.innerHTML = '';

    if (!scans.length) {
      tbody.innerHTML = '<tr class="empty-row"><td colspan="8">No scan records available</td></tr>';
      return;
    }

    scans.forEach((s, i) => {
      const isReal  = (s.verdict || '').toLowerCase().includes('real');
      const realP   = s.real_probability != null ? (s.real_probability * 100).toFixed(1) + '%' : '—';
      const aiP     = s.ai_probability   != null ? (s.ai_probability   * 100).toFixed(1) + '%' : '—';
      const time    = s.timestamp ? new Date(s.timestamp).toLocaleTimeString() : '—';
      const method  = (s.method || 'unknown').toUpperCase();
      const conf    = (s.confidence || '—').toString().toUpperCase();

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="color:var(--muted)">${i + 1}</td>
        <td title="${s.filename || ''}">${truncate(s.filename || 'unknown', 20)}</td>
        <td><span class="td-verdict ${isReal ? 'real' : 'fake'}">${isReal ? '✦ REAL' : '◈ FAKE'}</span></td>
        <td class="td-green">${realP}</td>
        <td class="td-red">${aiP}</td>
        <td class="td-blue">${conf}</td>
        <td style="color:var(--muted)">${method}</td>
        <td style="color:var(--muted)">${time}</td>`;
      tbody.appendChild(tr);
    });
  }

  function truncate(str, maxLen) {
    return str.length > maxLen ? str.slice(0, maxLen) + '…' : str;
  }

  // Run demo on load if no backend
  setTimeout(() => {
    const token = getAuthToken();
    if (document.getElementById('admin-panel') && document.getElementById('admin-panel').style.display !== 'none') {
      // panel is visible, stats already fetched
    }
  }, 500);

})();