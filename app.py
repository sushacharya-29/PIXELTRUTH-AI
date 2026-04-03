"""
PRODUCTION Flask Application - AI Image Detection
==================================================

v5.0 — integrated with v5inference.py (standalone functions, dynamic ensemble).

Integration notes:
  - Uses v5inference.py functions directly (get_spatial_model_v5, get_frequency_model_v5,
    predict_spatial, predict_frequency, dynamic_ensemble_predict).
  - No AIImageDetectorNN wrapper class — v5inference.py exposes plain functions.
  - Checkpoint filenames align with v5training.py output:
      spatial_model_v5_best.pth   (or spatial_model_v5_calibrated.pth if --calibrate was used)
      frequency_model_v5_best.pth (or frequency_model_v5_calibrated.pth)
  - Dynamic ensemble: frequency model's own confidence score drives its weight (10-40%).
  - Fallback chain: neural ensemble → spatial-only → frequency-only → heuristic → 503.

Environment variables for tuning REAL identification:
  SPATIAL_TEMPERATURE   (float, default 1.0) — raise to 1.5-2.0 if REAL is mis-classified
  FREQ_TEMPERATURE      (float, default 1.0) — same for frequency model
  REAL_THRESHOLD        (float, default 0.5) — lower to 0.45 for more REAL predictions

Author: AI Forensics Team
Version: 5.0 (Production)
"""

import os
import uuid
import logging
import torch
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pathlib import Path

try:
    from auth_routes import auth_bp
    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

if HAS_AUTH:
    app.register_blueprint(auth_bp)
else:
    logger.warning("auth_routes not found — auth endpoints disabled")

UPLOAD_FOLDER  = os.path.join(BASE_DIR, "uploads")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["SECRET_KEY"]         = os.environ.get("SECRET_KEY", "pixel-truth-secret-prod-v5")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================
# Tuning Parameters
# =========================

def _get_tuning_params() -> dict:
    """
    Read tuning parameters from environment variables.
    Raise SPATIAL_TEMPERATURE / FREQ_TEMPERATURE to 1.5-2.0 if real images
    are being misclassified as FAKE. Lower REAL_THRESHOLD to 0.45 for the same effect.
    """
    return {
        'spatial_temperature': float(os.environ.get('SPATIAL_TEMPERATURE', '1.0')),
        'freq_temperature':    float(os.environ.get('FREQ_TEMPERATURE',    '1.0')),
        'real_threshold':      float(os.environ.get('REAL_THRESHOLD',      '0.5')),
    }

# =========================
# Model Loading (Lazy, singleton)
# =========================

_spatial_model   = None
_frequency_model = None
_device          = None
_traditional_detector = None

# Checkpoint filenames produced by v5training.py
# Priority: calibrated (post-calibration) > best (raw best epoch)
_SPATIAL_CKPT_NAMES   = ['spatial_model_v5_calibrated.pth',   'spatial_model_v5_best.pth']
_FREQUENCY_CKPT_NAMES = ['frequency_model_v5_calibrated.pth', 'frequency_model_v5_best.pth']


def _resolve_checkpoint(names: list) -> str | None:
    """Return the first checkpoint path that exists, or None."""
    for name in names:
        path = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(path):
            return path
    return None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Torch device: %s", _device)
    return _device


def get_spatial_model():
    """Lazy-load the spatial (CLIP-semantic) model."""
    global _spatial_model
    if _spatial_model is not None:
        return _spatial_model

    ckpt_path = _resolve_checkpoint(_SPATIAL_CKPT_NAMES)
    if not ckpt_path:
        logger.warning(
            "No spatial checkpoint found in %s  (tried: %s)",
            CHECKPOINT_DIR, _SPATIAL_CKPT_NAMES
        )
        return None

    try:
        from v5spatial import get_spatial_model_v5
        params = _get_tuning_params()
        device = _get_device()

        logger.info("Loading spatial model from %s  (temperature=%.2f)", ckpt_path, params['spatial_temperature'])
        model = get_spatial_model_v5(temperature=params['spatial_temperature']).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        model.eval()
        _spatial_model = model
        logger.info("Spatial model ready")
        return _spatial_model

    except Exception as e:
        logger.error("Failed to load spatial model: %s", e, exc_info=True)
        return None


def get_frequency_model():
    """Lazy-load the frequency (upsampling-artifact) model."""
    global _frequency_model
    if _frequency_model is not None:
        return _frequency_model

    ckpt_path = _resolve_checkpoint(_FREQUENCY_CKPT_NAMES)
    if not ckpt_path:
        logger.warning(
            "No frequency checkpoint found in %s  (tried: %s)",
            CHECKPOINT_DIR, _FREQUENCY_CKPT_NAMES
        )
        return None

    try:
        from v5frequency import get_frequency_model_v5
        params = _get_tuning_params()
        device = _get_device()

        logger.info("Loading frequency model from %s  (temperature=%.2f)", ckpt_path, params['freq_temperature'])
        model = get_frequency_model_v5(temperature=params['freq_temperature']).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        model.eval()
        _frequency_model = model
        logger.info("Frequency model ready")
        return _frequency_model

    except Exception as e:
        logger.error("Failed to load frequency model: %s", e, exc_info=True)
        return None


def get_traditional_detector():
    """Lazy-load the traditional (heuristic) fallback detector."""
    global _traditional_detector
    if _traditional_detector is not None:
        return _traditional_detector

    try:
        from detector import AIImageDetector
        _traditional_detector = AIImageDetector()
        logger.info("Traditional heuristic detector loaded")
    except ImportError as e:
        logger.error("Cannot import detector module: %s", e)
        _traditional_detector = None
    except Exception as e:
        logger.error("Error loading traditional detector: %s", e, exc_info=True)
        _traditional_detector = None

    return _traditional_detector

# =========================
# Routes — Pages
# =========================

@app.route("/")
def landing():
    try:
        return render_template("landing.html")
    except Exception as e:
        logger.error("Error loading landing page: %s", e)
        return jsonify({"error": "Landing page not found"}), 500


@app.route("/home")
def index():
    try:
        return render_template("homepage.html")
    except Exception as e:
        logger.error("Error loading homepage: %s", e)
        return jsonify({"error": "Homepage not found"}), 500


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/admin/register")
def admin_register_page():
    return render_template("admin_register.html")

# =========================
# Routes — Health / Info
# =========================

@app.route("/health")
def health_check():
    """Health check — shows which detection path is active and tuning params."""
    spatial_loaded   = get_spatial_model()   is not None
    frequency_loaded = get_frequency_model() is not None
    trad_loaded      = get_traditional_detector() is not None

    if spatial_loaded and frequency_loaded:
        active_method = "neural_ensemble (spatial + frequency, dynamic weights)"
    elif spatial_loaded:
        active_method = "neural_spatial_only"
    elif frequency_loaded:
        active_method = "neural_frequency_only"
    elif trad_loaded:
        active_method = "traditional_heuristic_fallback"
    else:
        active_method = "none — no models available"

    params = _get_tuning_params()

    # List checkpoints present in the checkpoint directory
    found_checkpoints = []
    if os.path.exists(CHECKPOINT_DIR):
        found_checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]

    return jsonify({
        "status":                  "healthy",
        "active_detection_method": active_method,
        "models": {
            "spatial_loaded":   spatial_loaded,
            "frequency_loaded": frequency_loaded,
            "traditional_loaded": trad_loaded,
        },
        "checkpoints_found":    found_checkpoints,
        "checkpoint_directory": CHECKPOINT_DIR,
        "tuning_parameters": {
            "spatial_temperature": params['spatial_temperature'],
            "freq_temperature":    params['freq_temperature'],
            "real_threshold":      params['real_threshold'],
            "tip": (
                "If real images are classified as FAKE, try: "
                "export SPATIAL_TEMPERATURE=1.5 && export REAL_THRESHOLD=0.45"
            ),
        },
        "upload_folder_exists": os.path.exists(app.config["UPLOAD_FOLDER"]),
    })


@app.route("/api/models")
def model_info():
    """Get information about loaded models and available checkpoints."""
    found_checkpoints = []
    if os.path.exists(CHECKPOINT_DIR):
        found_checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]

    return jsonify({
        "spatial_model": {
            "loaded":             get_spatial_model() is not None,
            "checkpoint_tried":   _SPATIAL_CKPT_NAMES,
            "checkpoint_found":   _resolve_checkpoint(_SPATIAL_CKPT_NAMES),
        },
        "frequency_model": {
            "loaded":             get_frequency_model() is not None,
            "checkpoint_tried":   _FREQUENCY_CKPT_NAMES,
            "checkpoint_found":   _resolve_checkpoint(_FREQUENCY_CKPT_NAMES),
        },
        "traditional_detector": {
            "loaded": get_traditional_detector() is not None,
        },
        "checkpoint_directory": CHECKPOINT_DIR,
        "checkpoints_present":  found_checkpoints,
    })

# =========================
# Routes — Detection API
# =========================

@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Main detection endpoint.

    Detection priority:
      1. Dynamic ensemble  — both spatial + frequency models loaded
      2. Spatial-only      — only spatial model loaded
      3. Frequency-only    — only frequency model loaded
      4. Traditional heuristic fallback (detector.py)
      5. 503 if nothing available

    All neural paths use v5inference.py functions directly.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    filename    = secure_filename(file.filename)
    ext         = filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath    = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    try:
        file.save(filepath)
        logger.info("Processing upload: %s", filename)

        # ── Load models ──────────────────────────────────────────
        spatial_model   = get_spatial_model()
        frequency_model = get_frequency_model()
        device          = _get_device()

        # ── Neural detection paths ────────────────────────────────
        if spatial_model or frequency_model:
            try:
                from v5inference import (
                    get_spatial_inference_transform,
                    get_frequency_inference_transform,
                    predict_spatial,
                    predict_frequency,
                    dynamic_ensemble_predict,
                )

                spatial_transform   = get_spatial_inference_transform(image_size=224)
                frequency_transform = get_frequency_inference_transform(image_size=128)

                # ── Path 1: full dynamic ensemble ─────────────────
                if spatial_model and frequency_model:
                    logger.info("Running dynamic ensemble detection...")
                    r = dynamic_ensemble_predict(
                        filepath,
                        spatial_model,
                        frequency_model,
                        spatial_transform,
                        frequency_transform,
                        device,
                    )
                    params = _get_tuning_params()
                    real_prob_pct = r['real_prob'] * 100
                    ai_prob_pct   = r['fake_prob'] * 100

                    # Apply real_threshold env-var override
                    threshold = params['real_threshold']
                    verdict   = 'REAL' if r['real_prob'] >= threshold else 'FAKE'

                    confidence_str = (
                        'High'   if r['confidence'] >= 80 else
                        'Medium' if r['confidence'] >= 60 else
                        'Low'
                    )
                    logger.info(
                        "Ensemble done: verdict=%s  confidence=%.1f%%  "
                        "real=%.1f%%  fake=%.1f%%  agree=%s",
                        verdict, r['confidence'], real_prob_pct, ai_prob_pct, r['models_agree']
                    )
                    return jsonify({
                        "filename":           filename,
                        "verdict":            verdict,
                        "confidence":         confidence_str,
                        "real_probability":   real_prob_pct,
                        "ai_probability":     ai_prob_pct,
                        "method":             "neural_ensemble_dynamic",
                        "dominant_signal":    r.get('dominant_signal'),
                        "models_agree":       r['models_agree'],
                        "spatial_weight":     round(r['spatial_weight'], 3),
                        "frequency_weight":   round(r['frequency_weight'], 3),
                        "models": {
                            "spatial": {
                                "prediction": r['spatial_pred'],
                                "confidence": round(r['spatial_conf'], 2),
                            },
                            "frequency": {
                                "prediction":      r['frequency_pred'],
                                "confidence":      round(r['frequency_conf'], 2),
                                "self_confidence": round(r['frequency_self_confidence'], 3),
                            },
                        },
                        "success": True,
                    })

                # ── Path 2: spatial-only ──────────────────────────
                elif spatial_model:
                    logger.info("Running spatial-only detection...")
                    r      = predict_spatial(filepath, spatial_model, spatial_transform, device)
                    params = _get_tuning_params()
                    threshold = params['real_threshold']
                    verdict   = 'REAL' if r['real_prob'] >= threshold else 'FAKE'
                    confidence_str = (
                        'High'   if r['confidence'] >= 80 else
                        'Medium' if r['confidence'] >= 60 else
                        'Low'
                    )
                    logger.info("Spatial-only done: verdict=%s  confidence=%.1f%%", verdict, r['confidence'])
                    return jsonify({
                        "filename":         filename,
                        "verdict":          verdict,
                        "confidence":       confidence_str,
                        "real_probability": r['real_prob'] * 100,
                        "ai_probability":   r['fake_prob'] * 100,
                        "method":           "neural_spatial_only",
                        "dominant_signal":  r.get('signal'),
                        "warning":          "Frequency model not loaded — spatial model only.",
                        "success":          True,
                    })

                # ── Path 3: frequency-only ────────────────────────
                else:
                    logger.info("Running frequency-only detection...")
                    r      = predict_frequency(filepath, frequency_model, frequency_transform, device)
                    params = _get_tuning_params()
                    threshold = params['real_threshold']
                    verdict   = 'REAL' if r['real_prob'] >= threshold else 'FAKE'
                    confidence_str = (
                        'High'   if r['confidence'] >= 80 else
                        'Medium' if r['confidence'] >= 60 else
                        'Low'
                    )
                    logger.info("Frequency-only done: verdict=%s  confidence=%.1f%%", verdict, r['confidence'])
                    return jsonify({
                        "filename":                  filename,
                        "verdict":                   verdict,
                        "confidence":                confidence_str,
                        "real_probability":          r['real_prob'] * 100,
                        "ai_probability":            r['fake_prob'] * 100,
                        "method":                    "neural_frequency_only",
                        "frequency_self_confidence": round(r['freq_confidence_score'], 3),
                        "warning":                   "Spatial model not loaded — frequency model only.",
                        "success":                   True,
                    })

            except Exception as e:
                logger.error("Neural detection error: %s", e, exc_info=True)
                logger.warning("Falling back to traditional detector...")

        # ── Path 4: traditional heuristic fallback ────────────────
        trad_det = get_traditional_detector()
        if trad_det:
            try:
                logger.info("Using traditional (heuristic) detector...")
                trad_results = trad_det.analyze(filepath, filename)

                if 'error' in trad_results:
                    raise RuntimeError(trad_results['error'])

                verdict_info = trad_results['verdict']
                return jsonify({
                    "filename":         filename,
                    "verdict":          verdict_info['label'],
                    "confidence":       verdict_info.get('confidence', 'Medium'),
                    "real_probability": verdict_info['real_probability'],
                    "ai_probability":   verdict_info['ai_probability'],
                    "method":           "traditional_heuristic",
                    "metadata":         trad_results.get('metadata'),
                    "visual":           trad_results.get('visual'),
                    "warning": (
                        "Neural models not loaded — using heuristic fallback. "
                        "Accuracy is significantly lower. Train models with v5training.py "
                        f"and place checkpoints in {CHECKPOINT_DIR}."
                    ),
                    "success": True,
                })
            except Exception as e:
                logger.error("Traditional detector error: %s", e, exc_info=True)
                return jsonify({
                    "error":   "Both detectors failed",
                    "details": str(e),
                    "success": False,
                }), 500

        # ── Path 5: nothing available ─────────────────────────────
        return jsonify({
            "error": "No detection methods available",
            "message": (
                "No trained checkpoints found and traditional detector unavailable. "
                f"Train with v5training.py and place checkpoints in {CHECKPOINT_DIR}. "
                f"Expected filenames: {_SPATIAL_CKPT_NAMES[0]}, {_FREQUENCY_CKPT_NAMES[0]}"
            ),
            "success": False,
        }), 503

    except Exception as e:
        logger.error("Detection pipeline error: %s", e, exc_info=True)
        return jsonify({
            "error":   "Detection failed",
            "details": str(e),
            "success": False,
        }), 500

    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                logger.warning("Could not delete temp file %s: %s", unique_name, e)

# =========================
# Error Handlers
# =========================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large (max 16 MB)"}), 413

@app.errorhandler(500)
def server_error(e):
    logger.error("Server error: %s", e)
    return jsonify({"error": "Internal server error"}), 500

# =========================
# Startup
# =========================

def print_startup_info():
    print("\n" + "=" * 70)
    print(" " * 15 + "PIXEL TRUTH — AI IMAGE DETECTOR v5.0")
    print("=" * 70)
    print(f"\n  Server:      http://localhost:5000")
    print(f"  Uploads:     {UPLOAD_FOLDER}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")

    for label, names in [("Spatial",   _SPATIAL_CKPT_NAMES),
                          ("Frequency", _FREQUENCY_CKPT_NAMES)]:
        found = _resolve_checkpoint(names)
        status = f"FOUND  →  {os.path.basename(found)}" if found else f"MISSING  (tried: {names})"
        print(f"\n  {label} checkpoint: {status}")

    params = _get_tuning_params()
    print(f"\n  Tuning (env vars):")
    print(f"    SPATIAL_TEMPERATURE = {params['spatial_temperature']}")
    print(f"    FREQ_TEMPERATURE    = {params['freq_temperature']}")
    print(f"    REAL_THRESHOLD      = {params['real_threshold']}")
    print(f"\n  Tip: If real images are classified as FAKE, try:")
    print(f"    export SPATIAL_TEMPERATURE=1.5")
    print(f"    export FREQ_TEMPERATURE=1.5")
    print(f"    export REAL_THRESHOLD=0.45")

    print("\n" + "=" * 70)
    print("  Endpoints:")
    print("    GET  /               Landing page")
    print("    GET  /home           Detection dashboard")
    print("    GET  /health         Health + active detection method + tuning")
    print("    GET  /api/models     Model load status + checkpoint paths")
    print("    POST /api/detect     Image detection (ensemble → spatial → freq → heuristic)")
    if HAS_AUTH:
        print("    POST /api/auth/register        User registration")
        print("    POST /api/auth/login           User login")
        print("    POST /api/auth/admin/register  Admin registration")
        print("    POST /api/auth/admin/login     Admin login")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_startup_info()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)