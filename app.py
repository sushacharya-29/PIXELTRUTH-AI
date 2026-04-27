"""
PIXEL TRUTH — app.py
Flask backend serving all frontend pages and stub API endpoints.

Routes:
  GET  /                    → Landing page (entry point)
  GET  /home                → Dashboard / image scanner
  GET  /login               → Login page
  GET  /register            → User registration page
  GET  /admin               → Admin monitor panel
  GET  /admin/register      → Admin registration page

  POST /api/auth/register          → User registration
  POST /api/auth/login             → User login
  POST /api/auth/admin/register    → Admin registration
  POST /api/auth/admin/login       → Admin login
  POST /api/detect                 → Image detection (stub/pluggable)
  GET  /api/admin/stats            → Admin statistics (stub/pluggable)
"""

import os
import json
import time
import random
import hashlib
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session
)

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'pixeltruth-dev-secret-2025')

# ─────────────────────────────────────────────
# In-memory stores (replace with DB in production)
# ─────────────────────────────────────────────
USERS  = {}   # username → { password_hash, mobile, role }
TOKENS = {}   # token    → { username, role, expires }
SCANS  = []   # list of scan records

# Admin key — set via env var or use this default for dev
ADMIN_KEY = os.environ.get('ADMIN_KEY', 'PIXELTRUTH_ADMIN_2025')


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()
def make_token(username: str, role: str) -> str:
    raw = f"{username}:{role}:{time.time()}:{random.random()}"
    token = hashlib.sha256(raw.encode()).hexdigest()

    TOKENS[token] = {
        'username': username,
        'role': role,
        'expires': time.time() + 3600  # 1 hour
    }

    return token

def get_token_from_request() -> str | None:
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        return auth[7:]
    return None

def verify_token(token: str) -> dict | None:
    data = TOKENS.get(token)

    if not data:
        return None

    if data.get('expires', 0) < time.time():
        TOKENS.pop(token, None)
        return None

    return data

# ─────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def landing():
    """Landing page — entry point of the application."""
    return render_template('landing.html')

@app.route('/home')
def homepage():
    """Dashboard / image analysis page."""
    return render_template('homepage.html')

@app.route('/login')
def login():
    """Login page."""
    return render_template('login.html')

@app.route('/register')
def register():
    """User registration page."""
    return render_template('register.html')

@app.route('/admin')
def admin():
    token = get_token_from_request()
    info = verify_token(token) if token else None

    if not info or info.get('role') != 'admin':
        return redirect('/login')

    return render_template('admin.html')

@app.route('/admin/register')
def admin_register():
    """Admin registration page."""
    return render_template('admin_register.html')

# Legacy alias — /landing also works
@app.route('/landing')
def landing_alias():
    return redirect(url_for('landing'))


# ─────────────────────────────────────────────
# AUTH API
# ─────────────────────────────────────────────

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register a new user account."""
    body = request.get_json(silent=True) or {}
    username = (body.get('username') or '').strip().lower()
    mobile   = (body.get('mobile')   or '').strip()
    password = (body.get('password') or '')

    # Validation
    if not username or len(username) < 3:
        return jsonify(success=False, error='Username must be at least 3 characters'), 400
    if not all(c.isalnum() or c == '_' for c in username):
        return jsonify(success=False, error='Username: only letters, numbers, underscores'), 400
    if username in USERS:
        return jsonify(success=False, error='Username already taken'), 409
    if not password or len(password) < 8:
        return jsonify(success=False, error='Password must be at least 8 characters'), 400

    USERS[username] = {
        'password_hash': hash_password(password),
        'mobile': mobile,
        'role': 'user',
        'created_at': datetime.utcnow().isoformat()
    }
    return jsonify(success=True, message='Account created successfully')


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Authenticate a user and return a token."""
    body = request.get_json(silent=True) or {}
    identifier = (body.get('identifier') or '').strip().lower()
    password   = (body.get('password')   or '')

    # Find user by username or mobile
    user = None
    uname_key = None
    for uname, udata in USERS.items():
        if uname == identifier or udata.get('mobile') == identifier:
            user = udata
            uname_key = uname
            break

    if not user or user['password_hash'] != hash_password(password):
        return jsonify(success=False, error='Invalid credentials'), 401

    token = make_token(uname_key, user['role'])
    TOKENS[token] = {'username': uname_key, 'role': user['role']}

    return jsonify(
        success=True,
        token=token,
        role=user['role'],
        username=uname_key,
        redirect='/home'
    )


@app.route('/api/auth/admin/register', methods=['POST'])
def api_admin_register():
    """Register a new admin account (requires admin key)."""
    body = request.get_json(silent=True) or {}
    username     = (body.get('username')     or '').strip().lower()
    mobile       = (body.get('mobile')       or '').strip()
    password     = (body.get('password')     or '')
    admin_key    = (body.get('admin_key')    or '').strip()
    access_level = (body.get('access_level') or 'moderator')

    if admin_key != ADMIN_KEY:
        return jsonify(success=False, error='Invalid admin authorization key'), 403
    if not username or len(username) < 3:
        return jsonify(success=False, error='Username must be at least 3 characters'), 400
    if username in USERS:
        return jsonify(success=False, error='Username already taken'), 409
    if not password or len(password) < 8:
        return jsonify(success=False, error='Password must be at least 8 characters'), 400

    USERS[username] = {
        'password_hash': hash_password(password),
        'mobile': mobile,
        'role': 'admin',
        'access_level': access_level,
        'created_at': datetime.utcnow().isoformat()
    }
    return jsonify(success=True, message='Admin account created successfully')


@app.route('/api/auth/admin/login', methods=['POST'])
def api_admin_login():
    """Authenticate an admin and return a token."""
    body = request.get_json(silent=True) or {}
    identifier = (body.get('identifier') or '').strip().lower()
    password   = (body.get('password')   or '')
    admin_key  = (body.get('admin_key')  or '').strip()

    if admin_key != ADMIN_KEY:
        return jsonify(success=False, error='Invalid admin authorization key'), 403

    user = None
    uname_key = None
    for uname, udata in USERS.items():
        if uname == identifier or udata.get('mobile') == identifier:
            user = udata
            uname_key = uname
            break

    if not user or user.get('role') != 'admin':
        return jsonify(success=False, error='No admin account found with those credentials'), 401
    if user['password_hash'] != hash_password(password):
        return jsonify(success=False, error='Invalid credentials'), 401

    token = make_token(uname_key, 'admin')
    TOKENS[token] = {'username': uname_key, 'role': 'admin'}

    return jsonify(
        success=True,
        token=token,
        role='admin',
        username=uname_key,
        redirect='/admin'
    )


# ─────────────────────────────────────────────
# DETECTION API  (stub — plug in your ML model)
# ─────────────────────────────────────────────

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """
    Image detection endpoint.
    Expects multipart/form-data with field 'image'.
    Returns JSON with verdict, probabilities, and model scores.

    Replace the stub logic below with your actual ML inference.
    """
    if 'image' not in request.files:
        return jsonify(success=False, error='No image file provided'), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify(success=False, error='Empty filename'), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        return jsonify(success=False, error='Invalid file type'), 400
    # ── Stub inference (replace with real model call) ──────────
    # Example: result = your_model.predict(file.read())
    real_prob = round(random.uniform(0.05, 0.95), 4)
    ai_prob   = round(1.0 - real_prob, 4)
    is_real   = real_prob >= 0.5

    result = {
        'success': True,
        'verdict': 'REAL' if is_real else 'FAKE',
        'real_probability': real_prob,
        'ai_probability':   ai_prob,
        'confidence': 'HIGH' if abs(real_prob - 0.5) > 0.35 else
                      'MEDIUM' if abs(real_prob - 0.5) > 0.15 else 'LOW',
        'method': 'ENSEMBLE',
        'models': {
            'frequency': {
                'real_probability': round(real_prob + random.uniform(-0.05, 0.05), 4),
                'verdict': 'REAL' if is_real else 'FAKE'
            },
            'spatial': {
                'real_probability': round(real_prob + random.uniform(-0.05, 0.05), 4),
                'verdict': 'REAL' if is_real else 'FAKE'
            }
        },
        'filename': file.filename,
        'timestamp': datetime.utcnow().isoformat()
    }

    # Save to scan log
    SCANS.insert(0, {
        'id': len(SCANS) + 1,
        'filename': file.filename,
        'verdict': result['verdict'],
        'real_probability': result['real_probability'],
        'ai_probability':   result['ai_probability'],
        'confidence':       result['confidence'],
        'method':           result['method'],
        'timestamp':        result['timestamp']
    })
    if len(SCANS) > 200:
        SCANS.pop()

    return jsonify(result)


# ─────────────────────────────────────────────
# ADMIN STATS API
# ─────────────────────────────────────────────

@app.route('/api/admin/stats', methods=['GET'])
def api_admin_stats():
    """
    Returns aggregate stats for the admin monitor.
    Token check is lightweight here — enforce strictly in production.
    """
    token = get_token_from_request()
    info  = verify_token(token) if token else None
    if not info or info.get('role') != 'admin':
        return jsonify(success=False, error='Unauthorized'), 401

    total  = len(SCANS)
    real_c = sum(1 for s in SCANS if s['verdict'] == 'REAL')
    fake_c = sum(1 for s in SCANS if s['verdict'] == 'FAKE')
    correct = round(total * 0.947) if total else 0

    return jsonify(
        success=True,
        total_scans=total,
        correct_predictions=correct,
        real_count=real_c,
        fake_count=fake_c,
        model_accuracy={
            'freq_sentinel': round(88.4 + random.uniform(0, 2), 1),
            'patch_oracle':  round(91.2 + random.uniform(0, 2), 1),
            'texture_judge': round(93.8 + random.uniform(0, 1), 1),
            'ensemble':      round(94.7 + random.uniform(0, 0.5), 1),
        },
        recent_scans=SCANS[:20],
        latest_scan=SCANS[0] if SCANS else None
    )


# ─────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify(error='Not found'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify(error='Internal server error'), 500


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  PIXEL TRUTH — Flask Development Server")
    print("="*55)
    print(f"  Landing page : http://127.0.0.1:5000/")
    print(f"  Dashboard    : http://127.0.0.1:5000/home")
    print(f"  Login        : http://127.0.0.1:5000/login")
    print(f"  Register     : http://127.0.0.1:5000/register")
    print(f"  Admin Panel  : http://127.0.0.1:5000/admin")
    print(f"  Admin Reg.   : http://127.0.0.1:5000/admin/register")
    print(f"\n  Admin Key    : {ADMIN_KEY}")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)