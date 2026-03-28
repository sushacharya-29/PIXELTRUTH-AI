/**
 * PIXEL TRUTH — Auth Pages Shared JS
 * Handles: cursor trail, form validation, API calls, password strength
 */

// ─── CURSOR TRAIL ───────────────────────────────────────────────
const trail = document.getElementById('cursor-trail');
if (trail) {
  document.addEventListener('mousemove', e => {
    trail.style.left = e.clientX + 'px';
    trail.style.top  = e.clientY + 'px';
  });
}

// ─── ALERT HELPERS ──────────────────────────────────────────────
function showAlert(id, message, type = 'error') {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = `alert ${type} show`;
  el.textContent = message;
}

function hideAlert(id) {
  const el = document.getElementById(id);
  if (el) el.className = 'alert';
}

// ─── PASSWORD TOGGLE ────────────────────────────────────────────
document.querySelectorAll('.toggle-pw').forEach(btn => {
  btn.addEventListener('click', () => {
    const targetId = btn.dataset.target;
    const input = document.getElementById(targetId);
    if (!input) return;
    const isText = input.type === 'text';
    input.type = isText ? 'password' : 'text';
    btn.textContent = isText ? 'SHOW' : 'HIDE';
  });
});

// ─── PASSWORD STRENGTH ──────────────────────────────────────────
function checkPasswordStrength(password) {
  let score = 0;
  if (password.length >= 8)  score++;
  if (/[A-Z]/.test(password)) score++;
  if (/[0-9]/.test(password)) score++;
  if (/[^A-Za-z0-9]/.test(password)) score++;
  return score; // 0-4
}

function updateStrengthMeter(password, meterId) {
  const meter = document.getElementById(meterId);
  if (!meter) return;
  const bars = meter.querySelectorAll('.strength-bar');
  const label = meter.querySelector('.strength-label');
  const score = checkPasswordStrength(password);

  bars.forEach((b, i) => {
    b.className = 'strength-bar';
    if (password.length === 0) return;
    if (score <= 1 && i < 1) b.classList.add('s1');
    else if (score === 2 && i < 2) b.classList.add('s2');
    else if (score >= 3 && i < score) b.classList.add('s3');
  });

  const labels = ['', 'WEAK', 'FAIR', 'GOOD', 'STRONG'];
  const colors = ['', 'var(--accent2)', 'var(--warn)', 'var(--accent3)', 'var(--accent3)'];
  if (label) {
    label.textContent = password.length ? labels[score] || labels[4] : '';
    label.style.color = colors[score] || '';
  }
}

// ─── FORM VALIDATION HELPERS ────────────────────────────────────
const validators = {
  username(val) {
    if (!val || val.trim().length < 3) return 'Username must be at least 3 characters';
    if (!/^[a-zA-Z0-9_]+$/.test(val)) return 'Only letters, numbers, underscores allowed';
    return null;
  },
  mobile(val) {
    const clean = val.replace(/\s/g, '');
    if (!/^[6-9]\d{9}$/.test(clean) && !/^\+91[6-9]\d{9}$/.test(clean)) return 'Enter valid 10-digit mobile number';
    return null;
  },
  password(val) {
    if (!val || val.length < 8) return 'Password must be at least 8 characters';
    return null;
  },
  confirmPassword(val, original) {
    if (val !== original) return 'Passwords do not match';
    return null;
  },
  adminKey(val) {
    if (!val || val.trim().length < 4) return 'Admin key is required';
    return null;
  }
};

function markField(inputId, errorMsg) {
  const input = document.getElementById(inputId);
  if (!input) return;
  if (errorMsg) {
    input.classList.add('error-field');
    const hint = input.closest('.form-group')?.querySelector('.field-hint');
    if (hint) { hint.textContent = '⚠ ' + errorMsg; hint.classList.add('show'); }
  } else {
    input.classList.remove('error-field');
    const hint = input.closest('.form-group')?.querySelector('.field-hint');
    if (hint) hint.classList.remove('show');
  }
}

// ─── API CALL WRAPPER ────────────────────────────────────────────
async function apiCall(endpoint, payload) {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  return { ok: res.ok, status: res.status, data };
}

// ─── BUTTON LOADING STATE ────────────────────────────────────────
function setBtnLoading(btn, loading, text = '') {
  if (loading) {
    btn.dataset.originalText = btn.textContent;
    btn.textContent = 'PROCESSING...';
    btn.classList.add('loading');
  } else {
    btn.textContent = text || btn.dataset.originalText || btn.textContent;
    btn.classList.remove('loading');
  }
}
