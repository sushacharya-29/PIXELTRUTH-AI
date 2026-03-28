"""
Auth Routes — PIXEL TRUTH
==========================
User & Admin registration, login endpoints.
Drop this into app_INTEGRATED.py or include as a Blueprint.

Usage in app_INTEGRATED.py:
    from auth_routes import auth_bp
    app.register_blueprint(auth_bp)

Simple file-based user store (no DB dependency).
For production: swap _load_users / _save_users with a real DB.
"""

import os
import json
import hashlib
import secrets
import time
from pathlib import Path
from flask import Blueprint, request, jsonify

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# ─── User Store (JSON file — swap with DB in production) ────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"

# Admin key — set via env var in production
ADMIN_SECRET_KEY = os.environ.get("ADMIN_SECRET_KEY", "PIXELTRUTH_ADMIN_2025")

# Active session tokens (in-memory; use Redis/DB in production)
_sessions: dict = {}


def _load_users() -> dict:
    """Load users from JSON store."""
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    """Persist users to JSON store."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    """Hash password with SHA-256 + salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def _check_password(password: str, stored: str) -> bool:
    """Verify hashed password."""
    try:
        salt, hashed = stored.split(":", 1)
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False


def _create_token(username: str, role: str) -> str:
    """Create a simple session token."""
    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "username": username,
        "role": role,
        "created_at": time.time()
    }
    return token


def _resolve_identifier(identifier: str, users: dict):
    """Find user by username OR mobile number."""
    # Try direct username lookup first
    if identifier in users:
        return identifier, users[identifier]
    # Try mobile number lookup
    clean_mobile = identifier.replace("+91", "").replace(" ", "")
    for uname, udata in users.items():
        stored_mobile = udata.get("mobile", "").replace("+91", "").replace(" ", "")
        if stored_mobile == clean_mobile:
            return uname, udata
    return None, None


# ─── USER REGISTRATION ──────────────────────────────────────────
@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}

    username = (data.get("username") or "").strip()
    mobile   = (data.get("mobile")   or "").strip().replace(" ", "")
    password = data.get("password", "")

    # Basic validation
    if not username or len(username) < 3:
        return jsonify({"success": False, "error": "Username must be at least 3 characters"}), 400
    if not mobile or len(mobile) < 10:
        return jsonify({"success": False, "error": "Enter a valid 10-digit mobile number"}), 400
    if not password or len(password) < 8:
        return jsonify({"success": False, "error": "Password must be at least 8 characters"}), 400

    users = _load_users()

    # Duplicate check
    if username in users:
        return jsonify({"success": False, "error": "Username already exists"}), 409

    # Mobile duplicate check
    clean_mobile = mobile.replace("+91", "")
    for u in users.values():
        if u.get("mobile", "").replace("+91", "") == clean_mobile:
            return jsonify({"success": False, "error": "Mobile number already registered"}), 409

    # Save user
    users[username] = {
        "username":   username,
        "mobile":     mobile,
        "password":   _hash_password(password),
        "role":       "user",
        "created_at": time.time()
    }
    _save_users(users)

    return jsonify({"success": True, "message": "Account created successfully"})


# ─── USER LOGIN ─────────────────────────────────────────────────
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    identifier = (data.get("identifier") or "").strip()
    password   = data.get("password", "")

    if not identifier or not password:
        return jsonify({"success": False, "error": "Username/mobile and password are required"}), 400

    users = _load_users()
    username, user = _resolve_identifier(identifier, users)

    if not user or not _check_password(password, user.get("password", "")):
        return jsonify({"success": False, "error": "Invalid credentials"}), 401

    if user.get("role") == "admin" or user.get("role") == "superadmin":
        return jsonify({"success": False, "error": "Use admin login for admin accounts"}), 403

    token = _create_token(username, user["role"])
    return jsonify({
        "success":  True,
        "token":    token,
        "role":     user["role"],
        "username": username,
        "redirect": "/"
    })


# ─── ADMIN REGISTRATION ─────────────────────────────────────────
@auth_bp.route("/admin/register", methods=["POST"])
def admin_register():
    data = request.get_json(silent=True) or {}

    username     = (data.get("username")     or "").strip()
    mobile       = (data.get("mobile")       or "").strip().replace(" ", "")
    password     = data.get("password", "")
    admin_key    = (data.get("admin_key")    or "").strip()
    access_level = data.get("access_level", "moderator")

    # Verify admin key
    if admin_key != ADMIN_SECRET_KEY:
        return jsonify({"success": False, "error": "Invalid admin authorization key"}), 403

    if not username or len(username) < 3:
        return jsonify({"success": False, "error": "Username must be at least 3 characters"}), 400
    if not mobile or len(mobile) < 10:
        return jsonify({"success": False, "error": "Enter a valid 10-digit mobile number"}), 400
    if not password or len(password) < 8:
        return jsonify({"success": False, "error": "Password must be at least 8 characters"}), 400

    role = "superadmin" if access_level == "superadmin" else "admin"

    users = _load_users()
    if username in users:
        return jsonify({"success": False, "error": "Username already exists"}), 409

    users[username] = {
        "username":     username,
        "mobile":       mobile,
        "password":     _hash_password(password),
        "role":         role,
        "access_level": access_level,
        "created_at":   time.time()
    }
    _save_users(users)

    return jsonify({"success": True, "message": f"Admin account created with {access_level} access"})


# ─── ADMIN LOGIN ────────────────────────────────────────────────
@auth_bp.route("/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json(silent=True) or {}

    identifier = (data.get("identifier") or "").strip()
    password   = data.get("password", "")
    admin_key  = (data.get("admin_key")  or "").strip()

    if not identifier or not password or not admin_key:
        return jsonify({"success": False, "error": "All fields are required for admin login"}), 400

    if admin_key != ADMIN_SECRET_KEY:
        return jsonify({"success": False, "error": "Invalid admin authorization key"}), 403

    users = _load_users()
    username, user = _resolve_identifier(identifier, users)

    if not user or not _check_password(password, user.get("password", "")):
        return jsonify({"success": False, "error": "Invalid credentials"}), 401

    if user.get("role") not in ("admin", "superadmin"):
        return jsonify({"success": False, "error": "Account does not have admin privileges"}), 403

    token = _create_token(username, user["role"])
    return jsonify({
        "success":      True,
        "token":        token,
        "role":         user["role"],
        "access_level": user.get("access_level", "moderator"),
        "username":     username,
        "redirect":     "/"
    })


# ─── TOKEN VALIDATION HELPER ────────────────────────────────────
def require_auth(roles=None):
    """
    Decorator to protect routes requiring authentication.
    
    Usage:
        @app.route('/protected')
        @require_auth(roles=['admin', 'superadmin'])
        def protected():
            ...
    """
    from functools import wraps

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            session = _sessions.get(token)
            if not session:
                return jsonify({"error": "Authentication required"}), 401
            if roles and session["role"] not in roles:
                return jsonify({"error": "Insufficient privileges"}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator
