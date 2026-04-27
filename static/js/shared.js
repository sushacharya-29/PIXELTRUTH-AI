/**
 * PIXEL TRUTH — shared.js
 * Cursor, auth state helpers, utilities
 */

// ── Cursor dot ──────────────────────────────────────────────────
const cursorDot = document.getElementById('cursor-dot');
if (cursorDot) {
  document.addEventListener('mousemove', e => {
    cursorDot.style.left = e.clientX + 'px';
    cursorDot.style.top  = e.clientY + 'px';
  });
}

// ── Auth helpers ─────────────────────────────────────────────────
function getAuthToken()    { return localStorage.getItem('pt_token'); }
function getAuthUsername() { return localStorage.getItem('pt_username'); }
function getAuthRole()     { return localStorage.getItem('pt_role'); }

function handleLogout() {
  localStorage.removeItem('pt_token');
  localStorage.removeItem('pt_username');
  localStorage.removeItem('pt_role');
  window.location.href = '/';
}

// ── Nav auth state ────────────────────────────────────────────────
function updateNavAuth() {
  const token    = getAuthToken();
  const username = getAuthUsername();
  const authLinks= document.getElementById('nav-auth-links');
  const userInfo = document.getElementById('nav-user-info');
  const navUname = document.getElementById('nav-username') || document.getElementById('admin-username');

  if (authLinks) authLinks.style.display = (token && username) ? 'none' : 'flex';
  if (userInfo)  userInfo.style.display  = (token && username) ? 'flex' : 'none';
  if (navUname) {
  navUname.textContent = username ? username.toUpperCase() : '';
}
}
document.addEventListener('DOMContentLoaded', updateNavAuth);

// ── Number counter animation ─────────────────────────────────────
function animateCount(el, target, duration = 1200) {
  const start = parseInt(el.textContent) || 0;
  const startTime = performance.now();
  const step = (now) => {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(start + (target - start) * ease);
    if (progress < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ── Format timestamp ──────────────────────────────────────────────
function formatTime(date) {
  return date.toTimeString().slice(0, 8);
}