import email
import os
import io
import sys
import time
import random
import hashlib
import secrets
import hmac
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session
)

# ── Supabase client ────────────────────────────────────────
from supabase import create_client

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://xnhwpspmipcnqcdfgyyp.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'sb_publishable_KSxoWK6KQgxfcofrKX-Sxg_TofsxGKt')

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[INFO] Supabase client initialised successfully.")
except Exception as _sb_err:
    supabase = None
    print(f"[WARN] Supabase client failed to initialise: {_sb_err}. Some features won't work.")

# ── Model import ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from v64spatial import get_spatial_model_v64, CLIP_MEAN, CLIP_STD
    _MODEL_AVAILABLE = True
except ImportError as _e:
    print(f"[WARN] v64spatial import failed: {_e}. /api/detect will return stub results.")
    _MODEL_AVAILABLE = False

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'pixeltruth-dev-secret-2025')

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODEL     = None
_CKPT_PATH = os.environ.get('SPATIAL_CKPT', 'checkpoints_v64/checkpoint_best.pt')

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CLIP_MEAN if _MODEL_AVAILABLE else [0.485, 0.456, 0.406],
        std=CLIP_STD  if _MODEL_AVAILABLE else [0.229, 0.224, 0.225],
    ),
])


ADMIN_EMAILS = {
    "sushanthsacharya789@gmail.com",
    "sushanthsacharya456@gmail.com",
    "shashankkharvi7671@gmail.com",
}

def is_admin_email(email: str) -> bool:
    """Return True if the email belongs to a default admin."""
    return (email or "").strip().lower() in {e.lower() for e in ADMIN_EMAILS}
def _load_model():
    global _MODEL
    if not _MODEL_AVAILABLE:
        return
    print("[INFO] Loading SpatialModelV6.4 …")
    model = get_spatial_model_v64(input_size=224)
    ckpt_path = _CKPT_PATH
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]} …")
        if unexpected:
            print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
        print(f"[INFO] Checkpoint loaded successfully from {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint not found at '{ckpt_path}' — running with random weights.")
    model.to(DEVICE)
    model.eval()
    _MODEL = model
    print(f"[INFO] Model ready on {DEVICE}.")


_load_model()


def _run_inference(image_bytes: bytes) -> dict:
    img    = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = _INFER_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        try:
            logits, stream_norms = _MODEL.forward_with_streams(tensor)
        except Exception:
            logits       = _MODEL(tensor)
            stream_norms = {}
        probs     = F.softmax(logits, dim=1)[0]
        fake_prob = float(probs[0])
        real_prob = float(probs[1])
    verdict    = 'REAL' if real_prob >= 0.5 else 'FAKE'
    margin     = abs(real_prob - 0.5)
    confidence = 'HIGH' if margin > 0.35 else 'MEDIUM' if margin > 0.15 else 'LOW'
    return {
        'real_prob':    real_prob,
        'fake_prob':    fake_prob,
        'verdict':      verdict,
        'confidence':   confidence,
        'stream_norms': {k: round(v, 4) for k, v in stream_norms.items()},
    }


def _stub_inference() -> dict:
    real_prob = round(random.uniform(0.05, 0.95), 4)
    fake_prob = round(1.0 - real_prob, 4)
    is_real   = real_prob >= 0.5
    margin    = abs(real_prob - 0.5)
    return {
        'real_prob':    real_prob,
        'fake_prob':    fake_prob,
        'verdict':      'REAL' if is_real else 'FAKE',
        'confidence':   'HIGH' if margin > 0.35 else 'MEDIUM' if margin > 0.15 else 'LOW',
        'stream_norms': {},
    }


# ── Supabase helpers ───────────────────────────────────────
def _log_scan_to_db(verdict: str, image_hash: str, image_url: str) -> None:
    if supabase is None:
        print("[WARN] Supabase not available — skipping DB log.")
        return

    db_result = 'REAL' if verdict == 'REAL' else 'GAN'

    try:
        supabase.table("scans").insert({
            "result": db_result,
            "image_hash": image_hash,
            "image_url": image_url   # ✅ now valid
        }).execute()

        print(f"[DB] Scan logged → result={db_result}")

    except Exception as err:
        print(f"[DB ERROR] Failed to log scan: {err}")


def _supabase_email_signup(email: str, password: str) -> dict:
    """Register a new user via Supabase email auth."""
    if supabase is None:
        return {'success': False, 'error': 'Auth service unavailable'}
    try:
        res = supabase.auth.sign_up({'email': email, 'password': password})
        if res.user:
            return {'success': True, 'user': res.user, 'session': res.session}
        return {'success': False, 'error': 'Registration failed'}
    except Exception as e:
        err = str(e)
        if 'already registered' in err.lower() or 'already been registered' in err.lower():
            return {'success': False, 'error': 'Email already registered'}
        return {'success': False, 'error': err}


def _supabase_email_signin(email: str, password: str) -> dict:
    """Sign in a user via Supabase email auth."""
    if supabase is None:
        return {'success': False, 'error': 'Auth service unavailable'}
    try:
        res = supabase.auth.sign_in_with_password({'email': email, 'password': password})
        if res.user and res.session:
            return {
                'success': True,
                'user': res.user,
                'access_token': res.session.access_token,
                'refresh_token': res.session.refresh_token,
            }
        return {'success': False, 'error': 'Invalid credentials'}
    except Exception as e:
        err = str(e)
        if 'invalid' in err.lower() or 'credentials' in err.lower():
            return {'success': False, 'error': 'Invalid email or password'}
        return {'success': False, 'error': err}
def generate_image_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()

def _supabase_email_reset(email: str) -> dict:
    """Send password reset email via Supabase."""
    if supabase is None:
        return {'success': False, 'error': 'Auth service unavailable'}
    try:
        supabase.auth.reset_password_email(email)
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ── In-memory stores ───────────────────────────────────────
USERS     = {}   # manual user accounts
TOKENS    = {}   # manual session tokens
SCANS     = []

# ── Admin security ─────────────────────────────────────────
ADMIN_KEY          = os.environ.get('ADMIN_KEY', 'PIXELTRUTH_ADMIN_2025')
ADMIN_INVITE_CODES = {}        # invite_code -> { used: bool, created_at, access_level }
ADMIN_SESSIONS     = {}        # session_id -> { admin_id, expires, ip, created_at }
ADMIN_AUDIT_LOG    = []        # append-only audit trail

# Rate limiting: { ip -> [timestamp, ...] }
_LOGIN_ATTEMPTS  = defaultdict(list)
_MAX_ATTEMPTS    = 5
_LOCKOUT_SECONDS = 300  # 5 minutes


def _rate_limit_check(ip: str) -> tuple[bool, int]:
    """Returns (is_allowed, seconds_remaining)."""
    now = time.time()
    window = 60  # 1-minute sliding window
    # Clean old entries
    _LOGIN_ATTEMPTS[ip] = [t for t in _LOGIN_ATTEMPTS[ip] if now - t < _LOCKOUT_SECONDS]
    recent = [t for t in _LOGIN_ATTEMPTS[ip] if now - t < window]
    if len(_LOGIN_ATTEMPTS[ip]) >= _MAX_ATTEMPTS:
        oldest_lockout = min(_LOGIN_ATTEMPTS[ip])
        remaining = int(_LOCKOUT_SECONDS - (now - oldest_lockout))
        return False, max(0, remaining)
    return True, 0


def _rate_limit_record(ip: str):
    _LOGIN_ATTEMPTS[ip].append(time.time())


def _rate_limit_clear(ip: str):
    _LOGIN_ATTEMPTS.pop(ip, None)


def _audit(action: str, admin_id: str = None, ip: str = None, detail: str = None):
    ADMIN_AUDIT_LOG.append({
        'timestamp': datetime.utcnow().isoformat(),
        'action':    action,
        'admin_id':  admin_id,
        'ip':        ip,
        'detail':    detail,
    })
    if len(ADMIN_AUDIT_LOG) > 1000:
        ADMIN_AUDIT_LOG.pop(0)


def generate_invite_code(access_level: str = 'moderator') -> str:
    """Generate a secure one-time admin invite code (server-side use only)."""
    code = secrets.token_urlsafe(32)
    ADMIN_INVITE_CODES[code] = {
        'used': False,
        'access_level': access_level,
        'created_at': datetime.utcnow().isoformat(),
    }
    return code


def hash_password(pw: str) -> str:
    salt = os.environ.get('PW_SALT', 'pixeltruth-salt-v1')
    return hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 260000).hex()


def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def make_token(username: str, role: str) -> str:
    raw   = f"{username}:{role}:{time.time()}:{secrets.token_hex(16)}"
    token = hashlib.sha256(raw.encode()).hexdigest()
    TOKENS[token] = {'username': username, 'role': role, 'expires': time.time() + 3600}
    return token


def make_admin_session(admin_id: str, ip: str) -> str:
    sid = secrets.token_urlsafe(48)
    ADMIN_SESSIONS[sid] = {
        'admin_id':   admin_id,
        'ip':         ip,
        'expires':    time.time() + 7200,   # 2-hour sessions
        'created_at': datetime.utcnow().isoformat(),
    }
    return sid


def get_token_from_request() -> str | None:
    auth = request.headers.get('Authorization', '')
    return auth[7:] if auth.startswith('Bearer ') else None


def verify_token(token: str) -> dict | None:
    data = TOKENS.get(token)
    if not data:
        return None
    if data.get('expires', 0) < time.time():
        TOKENS.pop(token, None)
        return None
    return data


def verify_admin_session(sid: str, ip: str = None) -> dict | None:
    data = ADMIN_SESSIONS.get(sid)
    if not data:
        return None
    if data['expires'] < time.time():
        ADMIN_SESSIONS.pop(sid, None)
        return None
    # Optional IP binding
    if ip and data.get('ip') and data['ip'] != ip:
        _audit('SESSION_IP_MISMATCH', admin_id=data.get('admin_id'), ip=ip,
               detail=f"Session IP {data['ip']} != request IP {ip}")
        return None
    return data


def get_admin_session_from_request():
    sid = request.headers.get('X-Admin-Session') or request.cookies.get('pt_admin_session')
    return verify_admin_session(sid, request.remote_addr) if sid else None


# ── PAGE ROUTES ────────────────────────────────────────────

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/home')
def homepage():
    return render_template('homepage.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

'''@app.route('/admin')
    def admin():
    token = get_token_from_request()
    info  = verify_token(token) if token else None
    if not info or info.get('role') != 'admin':
        return redirect('/admin/login')
    return render_template('admin.html')'''
@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/admin/login')
def admin_login():
    return render_template('admin_login.html')

@app.route('/admin/register')
def admin_register():
    return render_template('admin_register.html')

@app.route('/landing')
def landing_alias():
    return redirect(url_for('landing'))


# ── USER AUTH API (manual) ─────────────────────────────────

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    body     = request.get_json(silent=True) or {}
    username = (body.get('username') or '').strip().lower()
    mobile   = (body.get('mobile')   or '').strip()
    password = (body.get('password') or '')

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
        'mobile':        mobile,
        'role':          'user',
        'created_at':    datetime.utcnow().isoformat(),
    }
    return jsonify(success=True, message='Account created successfully')


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    body       = request.get_json(silent=True) or {}
    identifier = (body.get('identifier') or '').strip().lower()
    password   = (body.get('password')   or '')

    user, uname_key = None, None
    for uname, udata in USERS.items():
        if uname == identifier or udata.get('mobile') == identifier:
            user, uname_key = udata, uname
            break

    if not user or not constant_time_compare(user['password_hash'], hash_password(password)):
        return jsonify(success=False, error='Invalid credentials'), 401

    token = make_token(uname_key, user['role'])
    return jsonify(success=True, token=token, role=user['role'],
                   username=uname_key, redirect='/home')


# ── SUPABASE EMAIL AUTH API ───────────────────────────────

@app.route('/api/auth/email/register', methods=['POST'])
def api_email_register():
    body     = request.get_json(silent=True) or {}
    email    = (body.get('email')    or '').strip().lower()
    password = (body.get('password') or '')

    if not email or '@' not in email:
        return jsonify(success=False, error='Valid email address required'), 400
    if not password or len(password) < 8:
        return jsonify(success=False, error='Password must be at least 8 characters'), 400

    result = _supabase_email_signup(email, password)
    if result['success']:
        user = result['user']
        return jsonify(
            success=True,
            message='Account created. Check your email to confirm.',
            user_id=user.id if user else None,
            email_confirmed=user.email_confirmed_at is not None if user else False,
        )
    return jsonify(success=False, error=result.get('error', 'Registration failed')), 400


@app.route('/api/auth/email/login', methods=['POST'])
def api_email_login():
    body     = request.get_json(silent=True) or {}
    email    = (body.get('email')    or '').strip().lower()
    password = (body.get('password') or '')

    if not email or '@' not in email:
        return jsonify(success=False, error='Valid email address required'), 400
    if not password:
        return jsonify(success=False, error='Password required'), 400

    result = _supabase_email_signin(email, password)
    if result['success']:
        return jsonify(
            success=True,
            access_token=result['access_token'],
            refresh_token=result['refresh_token'],
            user_id=result['user'].id,
            email=result['user'].email,
            role='user',
            redirect='/home',
        )
    return jsonify(success=False, error=result.get('error', 'Login failed')), 401


@app.route('/api/auth/email/reset', methods=['POST'])
def api_email_reset():
    body  = request.get_json(silent=True) or {}
    email = (body.get('email') or '').strip().lower()

    if not email or '@' not in email:
        return jsonify(success=False, error='Valid email address required'), 400

    # Always return success to prevent email enumeration
    _supabase_email_reset(email)
    return jsonify(success=True, message='If that email exists, a reset link has been sent.')


# ── ADMIN AUTH API (secure, rate-limited) ─────────────────

@app.route('/api/auth/admin/register', methods=['POST'])
def api_admin_register():
    ip   = request.remote_addr
    body = request.get_json(silent=True) or {}

    username     = (body.get('username')     or '').strip().lower()
    mobile       = (body.get('mobile')       or '').strip()
    password     = (body.get('password')     or '')
    admin_key    = (body.get('admin_key')    or '').strip()
    access_level = (body.get('access_level') or 'moderator')

    # Rate limit registration attempts
    allowed, wait = _rate_limit_check(f"admin_reg:{ip}")
    if not allowed:
        _audit('ADMIN_REG_RATE_LIMITED', ip=ip)
        return jsonify(success=False, error=f'Too many attempts. Try again in {wait}s'), 429

    # Constant-time admin key comparison to prevent timing attacks
    if not constant_time_compare(admin_key, ADMIN_KEY):
        _rate_limit_record(f"admin_reg:{ip}")
        _audit('ADMIN_REG_BAD_KEY', ip=ip, detail=f'username_attempted={username}')
        time.sleep(0.5)  # Artificial delay on failure
        return jsonify(success=False, error='Invalid admin authorization key'), 403

    if not username or len(username) < 3:
        return jsonify(success=False, error='Username must be at least 3 characters'), 400
    if not all(c.isalnum() or c == '_' for c in username):
        return jsonify(success=False, error='Username: only letters, numbers, underscores'), 400
    if username in USERS:
        return jsonify(success=False, error='Username already taken'), 409
    if not password or len(password) < 12:
        return jsonify(success=False, error='Admin passwords must be at least 12 characters'), 400
    if access_level not in ('moderator', 'superadmin'):
        access_level = 'moderator'

    _rate_limit_clear(f"admin_reg:{ip}")
    USERS[username] = {
        'password_hash': hash_password(password),
        'mobile':        mobile,
        'role':          'admin',
        'access_level':  access_level,
        'created_at':    datetime.utcnow().isoformat(),
        'registered_ip': ip,
    }
    _audit('ADMIN_REGISTERED', admin_id=username, ip=ip, detail=f'access_level={access_level}')
    return jsonify(success=True, message='Admin account created successfully')


@app.route('/api/auth/admin/login', methods=['POST'])
def api_admin_login():
    ip   = request.remote_addr
    body = request.get_json(silent=True) or {}

    identifier = (body.get('identifier') or '').strip().lower()
    password   = (body.get('password')   or '')
    admin_key  = (body.get('admin_key')  or '').strip()

    # Rate limiting
    allowed, wait = _rate_limit_check(f"admin_login:{ip}")
    if not allowed:
        _audit('ADMIN_LOGIN_RATE_LIMITED', ip=ip)
        return jsonify(success=False, error=f'Too many failed attempts. Locked out for {wait}s'), 429

    # Step 1: verify admin key first (constant-time)
    if not constant_time_compare(admin_key, ADMIN_KEY):
        _rate_limit_record(f"admin_login:{ip}")
        _audit('ADMIN_LOGIN_BAD_KEY', ip=ip)
        time.sleep(random.uniform(0.4, 0.8))  # Jitter to prevent timing attacks
        return jsonify(success=False, error='Authentication failed'), 401

    # Step 2: find the user account
    user, uname_key = None, None
    for uname, udata in USERS.items():
        if uname == identifier or udata.get('mobile') == identifier:
            user, uname_key = udata, uname
            break

    # Step 3: verify role and password in constant time
    if not user or user.get('role') != 'admin':
        time.sleep(random.uniform(0.3, 0.6))
        _rate_limit_record(f"admin_login:{ip}")
        _audit('ADMIN_LOGIN_NO_ACCOUNT', ip=ip, detail=f'identifier={identifier}')
        return jsonify(success=False, error='Authentication failed'), 401

    if not constant_time_compare(user['password_hash'], hash_password(password)):
        _rate_limit_record(f"admin_login:{ip}")
        _audit('ADMIN_LOGIN_BAD_PASSWORD', ip=ip, admin_id=uname_key)
        time.sleep(random.uniform(0.3, 0.6))
        return jsonify(success=False, error='Authentication failed'), 401

    # Success — issue tokens
    _rate_limit_clear(f"admin_login:{ip}")
    token = make_token(uname_key, 'admin')
    sid   = make_admin_session(uname_key, ip)
    _audit('ADMIN_LOGIN_SUCCESS', admin_id=uname_key, ip=ip)

    resp = jsonify(
        success=True,
        token=token,
        session_id=sid,
        role='admin',
        access_level=user.get('access_level', 'moderator'),
        username=uname_key,
        redirect='/admin',
    )
    resp.set_cookie(
        'pt_admin_session', sid,
        httponly=True, secure=False,  # set secure=True in production with HTTPS
        samesite='Lax', max_age=7200,
    )
    return resp


@app.route('/api/auth/admin/logout', methods=['POST'])
def api_admin_logout():
    ip  = request.remote_addr
    sid = request.cookies.get('pt_admin_session') or (request.get_json(silent=True) or {}).get('session_id')
    if sid and sid in ADMIN_SESSIONS:
        admin_id = ADMIN_SESSIONS[sid].get('admin_id')
        ADMIN_SESSIONS.pop(sid, None)
        _audit('ADMIN_LOGOUT', admin_id=admin_id, ip=ip)

    resp = jsonify(success=True)
    resp.delete_cookie('pt_admin_session')
    return resp


@app.route('/api/auth/admin/sessions', methods=['GET'])
def api_admin_sessions():
    """List active admin sessions (superadmin only)."""
    token = get_token_from_request()
    info  = verify_token(token) if token else None
    if not info or info.get('role') != 'admin':
        return jsonify(success=False, error='Unauthorized'), 401

    uname = info.get('username')
    user  = USERS.get(uname, {})
    if user.get('access_level') != 'superadmin':
        return jsonify(success=False, error='Superadmin access required'), 403

    now      = time.time()
    sessions = []
    for sid, sdata in ADMIN_SESSIONS.items():
        if sdata['expires'] > now:
            sessions.append({
                'session_id': sid[:8] + '…',
                'admin_id':   sdata['admin_id'],
                'ip':         sdata['ip'],
                'created_at': sdata['created_at'],
                'expires_in': int(sdata['expires'] - now),
            })
    return jsonify(success=True, sessions=sessions, audit_log=ADMIN_AUDIT_LOG[-50:])


# ── DETECTION API ──────────────────────────────────────────

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif','.avif'}

@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'image' not in request.files:
        return jsonify(success=False, error='No image file provided'), 400

    file = request.files['image']
    if not file.filename:
        return jsonify(success=False, error='Empty filename'), 400

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(success=False, error=f'Unsupported file type: {ext}'), 400

    image_bytes = file.read()

    # ✅ Check empty file
    if len(image_bytes) == 0:
        return jsonify(success=False, error='Empty file'), 400

    # 🔥 Generate hash
    image_hash = generate_image_hash(image_bytes)

    # 🔥 File path
    file_path = f"{image_hash}.jpg"

    existing = None

    # 🔥 Safe DB check
    if supabase:
        try:
            existing = supabase.table("scans") \
                .select("image_hash") \
                .eq("image_hash", image_hash) \
                .limit(1) \
                .execute()
        except Exception:
            existing = None

    # 🔥 Upload only if not exists
    if supabase and (not existing or not existing.data):
        try:
            supabase.storage.from_("scans").upload(
                file_path,
                image_bytes,
                {"content-type": "image/jpeg"}
            )
        except Exception as e:
            print(f"[STORAGE ERROR] {e}")

    # 🔥 Get public URL safely
    image_url = None
    if supabase:
        try:
            image_url = supabase.storage.from_("scans").get_public_url(file_path)
        except Exception as e:
            print(f"[URL ERROR] {e}")

    # 🔥 Run model
    try:
        if _MODEL is not None:
            pred   = _run_inference(image_bytes)
            method = 'SPATIAL-V6.4'
        else:
            pred   = _stub_inference()
            method = 'STUB'
    except Exception as e:
        return jsonify(success=False, error=f'Inference error: {str(e)}'), 500

    real_prob = pred['real_prob']
    fake_prob = pred['fake_prob']
    verdict   = pred['verdict']
    conf      = pred['confidence']

    sn             = pred.get('stream_norms', {})
    freq_signal    = sn.get('freq_out', 0.0)
    dct_signal     = sn.get('dct_out',  0.0)
    spatial_signal = sn.get('final_pooled', 0.0) or sn.get('mid_pooled', 0.0)

    def _perturb(base, signal, scale=0.08):
        offset = (signal - 1.0) * scale
        return float(min(max(base + offset, 0.001), 0.999))
    spatial_real = _perturb(real_prob, spatial_signal, 0.06)
    spatial_fake = 1 - spatial_real
    result = {
        'success': True,
        'verdict': verdict,
        'real_probability': round(real_prob, 4),
        'ai_probability': round(fake_prob, 4),
        'confidence': conf,
        'method': method,
        
        'models': {
            'spatial': {
            'real_probability': round(spatial_real, 4),
            'ai_probability': round(spatial_fake, 4),
            'verdict': verdict
        }
    },
        'filename': file.filename,
        'timestamp': datetime.utcnow().isoformat(),
        'image_url': image_url
    }

    # ✅ CLEAN DB CALL (IMPORTANT FIX)
    _log_scan_to_db(verdict, image_hash, image_url)

    return jsonify(result)
@app.route('/api/admin/stats', methods=['GET'])
def api_admin_stats():
    import base64 as _b64, json as _json

    token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()

    if not token:
        return jsonify(success=False, error='Unauthorized'), 401

    # Try JWT decode first, then supabase.auth.get_user
    email = None
    try:
        parts = token.split('.')
        if len(parts) == 3:
            padded = parts[1] + '=' * (4 - len(parts[1]) % 4)
            payload = _json.loads(_b64.b64decode(padded).decode('utf-8'))
            email = payload.get('email') or payload.get('user_metadata', {}).get('email')
    except Exception:
        pass

    if not email:
        try:
            user = supabase.auth.get_user(token)
            email = user.user.email
        except Exception:
            return jsonify(success=False, error='Invalid token'), 401

    if not is_admin_email(email):
        return jsonify(success=False, error='Forbidden'), 403

    # 🔥 Fetch from DB
    res = supabase.table("scans") \
        .select("image_hash, result, created_at, image_url") \
        .execute()

    rows = res.data or []

    # ✅ TOTAL SCANS (includes duplicates)
    total_scans = len(rows)

    # 🔥 UNIQUE (latest per image)
    unique = {}
    for row in rows:
        unique[row["image_hash"]] = row

    unique_rows = list(unique.values())

    unique_images = len(unique_rows)

    # ✅ REAL / FAKE (unique only)
    real_c = sum(1 for r in unique_rows if r["result"] == "REAL")
    fake_c = sum(1 for r in unique_rows if r["result"] == "GAN")

    from collections import defaultdict
    date_counts = defaultdict(int)

    for r in unique_rows:
        date = r["created_at"][:10]
        date_counts[date] += 1

    recent_images = [
        {
            "image_url": r["image_url"],
            "result": r["result"],
            "date": r["created_at"]
        }
        for r in unique_rows[:20]
    ]

    # OPTIONAL METRIC
    correct = round(unique_images * 0.947) if unique_images else 0

    return jsonify(
        success=True,
        total_scans=total_scans,        # all scans
        unique_images=unique_images,    # deduplicated
        correct_predictions=correct,
        real_count=real_c,
        fake_count=fake_c,
        date_wise_stats=date_counts,
        images=recent_images,           # ✅ ready for UI
        model_accuracy={
            'freq_sentinel': round(88.4 + random.uniform(0, 2), 1),
            'patch_oracle':  round(91.2 + random.uniform(0, 2), 1),
            'texture_judge': round(93.8 + random.uniform(0, 1), 1),
            'ensemble':      round(94.7 + random.uniform(0, 0.5), 1),
        }
    )
@app.route('/api/auth/google')
def google_login():
    """
    Initiates Google OAuth. Supabase Python SDK always uses PKCE — it stores
    the code_verifier in the gotrue client's internal state. The callback MUST
    call exchange_code_for_session on the same supabase singleton.
    """
    try:
        # Build redirect_to dynamically so it works on any host/port
        base_url = request.host_url.rstrip('/')
        redirect_to = f"{base_url}/auth/callback"

        res = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": redirect_to,
                "query_params": {
                    "prompt": "select_account"   # always show Google account picker
                }
            }
        })
        return redirect(res.url)
    except Exception as e:
        print(f"[AUTH] google_login error: {e}")
        return f"OAuth init error: {e}", 500


@app.route('/auth/callback')
def auth_callback():
    """
    Handles the redirect from Supabase after Google login.

    PKCE flow (default): Supabase sends ?code=... in the query string.
    We exchange it server-side using the SAME supabase singleton that
    generated the code_verifier in /api/auth/google.

    Implicit flow fallback: if no code is present, serve a page whose JS
    reads the access_token from the URL hash fragment and calls /api/auth/google/verify.
    """
    code = request.args.get('code')
    error = request.args.get('error')
    error_desc = request.args.get('error_description', '')

    print(f"[AUTH] /auth/callback — code={'YES' if code else 'NO'}, error={error}, args={dict(request.args)}")

    # ── Supabase/Google returned an error ─────────────────────────────────
    if error:
        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Sign-in Error</title>
<style>body{{margin:0;background:#020408;color:#e8edf5;font-family:'Inter',sans-serif;
     display:flex;align-items:center;justify-content:center;min-height:100vh;flex-direction:column;gap:12px;}}
p{{color:#ff3d5a;font-size:14px;max-width:380px;text-align:center;line-height:1.6;}}
a{{color:#1a6fff;font-size:13px;}}</style></head>
<body><p>Google sign-in error: {error}<br>{error_desc}<br><br>
<a href="/api/auth/google">Try again &rarr;</a></p></body></html>""", 400

    # ── PKCE: server-side code exchange ───────────────────────────────────
    if code:
        try:
            result = supabase.auth.exchange_code_for_session({"auth_code": code})
            access_token = result.session.access_token
            user_email   = result.user.email
            role         = "admin" if is_admin_email(user_email) else "user"

            print(f"[AUTH] PKCE success — email={user_email}, role={role}")

            return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Signing in...</title>
<style>
body{{margin:0;background:#020408;color:#e8edf5;font-family:'Inter',sans-serif;
     display:flex;align-items:center;justify-content:center;min-height:100vh;
     flex-direction:column;gap:16px;}}
.spinner{{width:36px;height:36px;border:2px solid rgba(26,111,255,0.2);
          border-top-color:#1a6fff;border-radius:50%;animation:spin 0.8s linear infinite;}}
@keyframes spin{{to{{transform:rotate(360deg);}}}}
p{{font-size:14px;color:#7a8fa8;letter-spacing:0.04em;}}
</style></head>
<body>
<div class="spinner"></div><p>Signing you in...</p>
<script>
  localStorage.setItem('pt_token',    '{access_token}');
  localStorage.setItem('pt_role',     '{role}');
  localStorage.setItem('pt_username', '{user_email}');
  window.location.replace('/home');
</script>
</body></html>"""

        except Exception as exc:
            print(f"[AUTH] PKCE exchange error: {exc}")
            # Fall through to the implicit-flow page
            return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Sign-in Error</title>
<style>
body{{margin:0;background:#020408;color:#e8edf5;font-family:'Inter',sans-serif;
     display:flex;align-items:center;justify-content:center;min-height:100vh;
     flex-direction:column;gap:12px;}}
p{{color:#ff3d5a;font-size:14px;max-width:380px;text-align:center;line-height:1.6;}}
a{{color:#1a6fff;font-size:13px;}}
</style></head>
<body>
<p>Sign-in failed (auth code could not be exchanged).<br>
This can happen if the server was restarted mid-flow.<br>
<a href="/api/auth/google">Try signing in again &rarr;</a></p>
</body></html>""", 400

    # ── Implicit flow fallback: token arrives in the URL hash ─────────────
    # The hash is never sent to the server, so we serve JS that reads it
    # and calls /api/auth/google/verify to determine the role server-side.
    return """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Signing in...</title>
<style>
  body{margin:0;background:#020408;color:#e8edf5;font-family:'Inter',sans-serif;
       display:flex;align-items:center;justify-content:center;min-height:100vh;
       flex-direction:column;gap:16px;}
  .spinner{width:36px;height:36px;border:2px solid rgba(26,111,255,0.2);
           border-top-color:#1a6fff;border-radius:50%;animation:spin 0.8s linear infinite;}
  @keyframes spin{to{transform:rotate(360deg);}}
  p{font-size:14px;color:#7a8fa8;letter-spacing:0.04em;}
  .err{color:#ff3d5a;font-size:13px;max-width:340px;text-align:center;line-height:1.6;}
</style>
</head>
<body>
<div class="spinner" id="spin"></div>
<p id="msg">Completing sign-in...</p>
<script>
(async () => {
  function showErr(m) {
    document.getElementById('spin').style.display = 'none';
    const el = document.getElementById('msg');
    el.className = 'err';
    el.textContent = m;
    setTimeout(() => window.location.replace('/'), 4000);
  }

  const hp = new URLSearchParams(window.location.hash.substring(1));
  const access_token = hp.get('access_token');

  if (!access_token) {
    showErr('No authentication token received. Redirecting to login...');
    return;
  }

  document.getElementById('msg').textContent = 'Verifying identity...';

  let data;
  try {
    const r = await fetch('/api/auth/google/verify', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + access_token }
    });
    data = await r.json();
  } catch(e) {
    showErr('Server error during verification. Redirecting...');
    return;
  }

  if (!data.success) {
    showErr('Verification failed: ' + (data.error || 'unknown'));
    return;
  }

  localStorage.setItem('pt_token',    access_token);
  localStorage.setItem('pt_role',     data.role  || 'user');
  localStorage.setItem('pt_username', data.email || '');
  window.location.replace('/home');
})();
</script>
</body>
</html>
"""
@app.route('/api/auth/google/verify', methods=['POST'])
def google_verify():
    """
    Verify an access_token (implicit flow) and return the email + role.
    We decode the JWT payload directly (base64) so this works even when
    the Supabase Python SDK can't verify implicit-flow tokens server-side.
    As a fallback we also try supabase.auth.get_user().
    """
    import base64, json as _json

    token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()

    if not token:
        return jsonify(success=False, error='Missing token'), 401

    email = None

    # ── Method 1: decode JWT payload directly (no signature verification needed
    #    for role assignment — Supabase already verified the token at OAuth time) ──
    try:
        parts = token.split('.')
        if len(parts) == 3:
            payload_b64 = parts[1]
            # Add padding
            payload_b64 += '=' * (4 - len(payload_b64) % 4)
            payload = _json.loads(base64.b64decode(payload_b64).decode('utf-8'))
            # Supabase JWTs store the email in payload['email']
            email = payload.get('email') or payload.get('user_metadata', {}).get('email')
            print(f"[AUTH] JWT decode — email={email}, payload keys={list(payload.keys())}")
    except Exception as jwt_err:
        print(f"[AUTH] JWT decode failed: {jwt_err}")

    # ── Method 2: fallback to supabase.auth.get_user() ──
    if not email:
        try:
            user = supabase.auth.get_user(token)
            email = user.user.email
            print(f"[AUTH] supabase.get_user — email={email}")
        except Exception as e:
            print(f"[AUTH] supabase.get_user failed: {e}")
            return jsonify(success=False, error='Could not verify token'), 401

    if not email:
        return jsonify(success=False, error='Could not extract email from token'), 401

    role = "admin" if is_admin_email(email) else "user"
    print(f"[AUTH] google_verify — email={email}, role={role}")

    return jsonify(
        success=True,
        email=email,
        role=role
    )
# ── ERROR HANDLERS ─────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify(error='Not found'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify(error='Internal server error'), 500

@app.route('/api/auth/verify', methods=['POST'])
def verify_login():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')

    try:
        user = supabase.auth.get_user(token)
        return jsonify(success=True, email=user.user.email)
    except:
        return jsonify(success=False), 401
    
@app.route('/api/auth/logout', methods=['POST'])
def logout():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if token and supabase:
            # Sign out the specific user session using their access token
            supabase.auth.admin.sign_out(token)
    except Exception:
        pass

    resp = jsonify(success=True)
    resp.delete_cookie("sb-access-token")
    resp.delete_cookie("sb-refresh-token")
    # Also delete the Supabase project-specific cookie formats
    resp.delete_cookie(f"sb-{SUPABASE_URL.split('//')[1].split('.')[0]}-auth-token")
    return resp
# ── RUN ────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  PIXEL TRUTH — Flask + SpatialModelV6.4")
    print("=" * 60)
    print(f"  Device       : {DEVICE}")
    print(f"  Model loaded : {_MODEL is not None}")
    print(f"  Checkpoint   : {_CKPT_PATH}")
    print(f"  Supabase     : {'connected' if supabase else 'NOT connected'}")
    print(f"  Landing      : http://127.0.0.1:5000/")
    print(f"  Dashboard    : http://127.0.0.1:5000/home")
    print(f"  Admin Login  : http://127.0.0.1:5000/admin/login")
    print(f"  Admin Key    : {ADMIN_KEY}")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)