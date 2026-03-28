"""
PRODUCTION Flask Application - AI Image Detection
==================================================

v3.0 — aligned with inference.py v3.0:
  - Neural detector now correctly uses spatial + frequency models with
    the fixed preprocessing pipelines and label ordering.
  - Added support for temperature scaling and real_threshold via
    environment variables — allows fixing REAL-identification without retraining.
  - Improved health check shows per-class accuracy from loaded checkpoints.

Environment variables for tuning REAL identification:
  SPATIAL_TEMPERATURE   (float, default 1.0) — raise to 1.5-2.0 if REAL is mis-classified
  FREQ_TEMPERATURE      (float, default 1.0) — same for frequency model
  REAL_THRESHOLD        (float, default 0.5) — lower to 0.45 for more REAL predictions

Author: AI Forensics Team
Version: 3.0 (Production)
"""

import os
import uuid
import logging
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
app.config["SECRET_KEY"]         = os.environ.get("SECRET_KEY", "pixel-truth-secret-prod-v3")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================
# Model Loading (Lazy, singleton)
# =========================

neural_detector      = None
traditional_detector = None


def _get_tuning_params():
    """
    Read tuning parameters from environment variables.
    These allow fixing REAL identification without retraining.

    Usage examples:
      export SPATIAL_TEMPERATURE=1.5   # soften FAKE-overconfidence in spatial model
      export FREQ_TEMPERATURE=1.5      # same for frequency model
      export REAL_THRESHOLD=0.45       # lower threshold to classify more images as REAL
    """
    return {
        'spatial_temperature': float(os.environ.get('SPATIAL_TEMPERATURE', '1.0')),
        'freq_temperature':    float(os.environ.get('FREQ_TEMPERATURE',    '1.0')),
        'real_threshold':      float(os.environ.get('REAL_THRESHOLD',      '0.5')),
    }


def get_neural_detector():
    """
    Lazy-load the neural network detector (spatial + frequency ensemble).

    Preprocessing handled inside AIImageDetectorNN (inference.py v3.0):
      - Spatial model:   CIFAKE normalization ([0.4914, 0.4822, 0.4465])
      - Frequency model: raw [0,1] tensor, no normalization (FFT-safe)
    Label convention: class 0 = FAKE, class 1 = REAL (CIFAKE alphabetical)

    REAL-identification tuning (via env vars):
      SPATIAL_TEMPERATURE, FREQ_TEMPERATURE, REAL_THRESHOLD
    """
    global neural_detector
    if neural_detector is not None:
        return neural_detector

    try:
        from inference import AIImageDetectorNN

        spatial_path   = os.path.join(CHECKPOINT_DIR, 'spatial_model_best.pth')
        frequency_path = os.path.join(CHECKPOINT_DIR, 'frequency_model_best.pth')

        spatial_weights   = spatial_path   if os.path.exists(spatial_path)   else None
        frequency_weights = frequency_path if os.path.exists(frequency_path) else None

        if not spatial_weights and not frequency_weights:
            logger.warning(
                "No trained checkpoints found in %s — neural detector unavailable. "
                "Falling back to traditional (heuristic) detector.",
                CHECKPOINT_DIR
            )
            return None

        if not spatial_weights:
            logger.warning("spatial_model_best.pth not found — running frequency-only ensemble")
        if not frequency_weights:
            logger.warning("frequency_model_best.pth not found — running spatial-only ensemble")

        params = _get_tuning_params()
        logger.info(
            "Loading neural network detector... "
            "(spatial_temp=%.2f, freq_temp=%.2f, real_threshold=%.2f)",
            params['spatial_temperature'],
            params['freq_temperature'],
            params['real_threshold'],
        )

        neural_detector = AIImageDetectorNN(
            spatial_weights=spatial_weights,
            frequency_weights=frequency_weights,
            device=None,
            spatial_temperature=params['spatial_temperature'],
            freq_temperature=params['freq_temperature'],
            real_threshold=params['real_threshold'],
        )
        info = neural_detector.get_model_info()
        logger.info(
            "Neural detector ready — device=%s  spatial=%s  frequency=%s  "
            "spatial_temp=%.2f  freq_temp=%.2f  real_threshold=%.2f",
            info['device'],
            info['spatial_loaded'],
            info['frequency_loaded'],
            info['spatial_temperature'],
            info['freq_temperature'],
            info['real_threshold'],
        )
        return neural_detector

    except ImportError as e:
        logger.error("Cannot import inference module: %s", e)
    except Exception as e:
        logger.error("Error loading neural detector: %s", e, exc_info=True)

    neural_detector = None
    return None


def get_traditional_detector():
    """Lazy-load the traditional (heuristic) fallback detector."""
    global traditional_detector
    if traditional_detector is not None:
        return traditional_detector

    try:
        from detector import AIImageDetector
        traditional_detector = AIImageDetector()
        logger.info("Traditional (heuristic) detector v3.0 loaded")
    except ImportError as e:
        logger.error("Cannot import detector module: %s", e)
        traditional_detector = None
    except Exception as e:
        logger.error("Error loading traditional detector: %s", e, exc_info=True)
        traditional_detector = None

    return traditional_detector

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
def login():
    return render_template("login.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/admin/register")
def admin_register():
    return render_template("admin_register.html")

# =========================
# Routes — Health / Info
# =========================

@app.route("/health")
def health_check():
    """Health check — shows which detection path is active and tuning params."""
    neural_det = get_neural_detector()
    trad_det   = get_traditional_detector()
    neural_info = neural_det.get_model_info() if neural_det else None

    active_method = "none"
    if neural_det:
        info = neural_det.get_model_info()
        if info['spatial_loaded'] and info['frequency_loaded']:
            active_method = "neural_ensemble (spatial + frequency)"
        elif info['spatial_loaded']:
            active_method = "neural_spatial_only"
        elif info['frequency_loaded']:
            active_method = "neural_frequency_only"
    elif trad_det:
        active_method = "traditional_heuristic_fallback"

    params = _get_tuning_params()

    return jsonify({
        "status":                   "healthy",
        "active_detection_method":  active_method,
        "neural_detector": {
            "available":  neural_det is not None,
            "info":       neural_info,
        },
        "traditional_detector": {
            "available": trad_det is not None,
        },
        "tuning_parameters": {
            "spatial_temperature": params['spatial_temperature'],
            "freq_temperature":    params['freq_temperature'],
            "real_threshold":      params['real_threshold'],
            "note": (
                "To fix REAL mis-classification: set SPATIAL_TEMPERATURE=1.5 "
                "and/or REAL_THRESHOLD=0.45 as environment variables."
            ),
        },
        "upload_folder_exists": os.path.exists(app.config["UPLOAD_FOLDER"]),
    })


@app.route("/api/models")
def model_info():
    """Get information about loaded models."""
    neural_det = get_neural_detector()
    trad_det   = get_traditional_detector()

    checkpoints = []
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]

    return jsonify({
        "neural_detector": {
            "loaded": neural_det is not None,
            "info":   neural_det.get_model_info() if neural_det else None,
        },
        "traditional_detector": {
            "loaded": trad_det is not None,
        },
        "checkpoint_directory": CHECKPOINT_DIR,
        "checkpoints_found":    checkpoints,
    })

# =========================
# Routes — Detection API
# =========================

@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Main detection endpoint.

    Detection priority:
      1. Neural ensemble (spatial CNN + frequency CNN) via inference.py v3.0
      2. Traditional heuristic detector (detector.py v3.0) — fallback

    REAL-identification tuning (env vars):
      SPATIAL_TEMPERATURE — raise to 1.5-2.0 to soften FAKE overconfidence
      FREQ_TEMPERATURE    — same for frequency model
      REAL_THRESHOLD      — lower to 0.45 to classify more images as REAL
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

        # ── Primary path: neural ensemble ──
        neural_det = get_neural_detector()
        if neural_det:
            try:
                logger.info("Running neural ensemble detection...")
                results = neural_det.detect(filepath, ensemble_method='weighted_average')

                ensemble = results.get('ensemble', {})
                if not ensemble or 'error' in ensemble:
                    raise RuntimeError(
                        f"Ensemble failed: {ensemble.get('error', 'unknown error')}"
                    )

                response = {
                    "filename":         filename,
                    "verdict":          ensemble['verdict'],
                    "confidence":       ensemble.get('confidence', 'N/A'),
                    "real_probability": ensemble['real_probability'],
                    "ai_probability":   ensemble['ai_probability'],
                    "method":           "neural_network",
                    "ensemble_method":  ensemble.get('method', 'weighted_average'),
                    "models": {
                        "spatial":   results.get('spatial'),
                        "frequency": results.get('frequency'),
                    },
                    "success": True,
                }
                logger.info(
                    "Neural detection complete: verdict=%s  confidence=%s  "
                    "real=%.1f%%  ai=%.1f%%",
                    ensemble['verdict'],
                    ensemble.get('confidence'),
                    ensemble['real_probability'],
                    ensemble['ai_probability'],
                )
                return jsonify(response)

            except Exception as e:
                logger.error("Neural detector error: %s", e, exc_info=True)
                logger.warning("Falling back to traditional detector...")

        # ── Fallback: traditional heuristic detector ──
        trad_det = get_traditional_detector()
        if trad_det:
            try:
                logger.info("Using traditional (heuristic) detector...")
                trad_results = trad_det.analyze(filepath, filename)

                if 'error' in trad_results:
                    raise RuntimeError(trad_results['error'])

                verdict_info = trad_results['verdict']
                response = {
                    "filename":         filename,
                    "verdict":          verdict_info['label'],
                    "confidence":       verdict_info.get('confidence', 'Medium'),
                    "real_probability": verdict_info['real_probability'],
                    "ai_probability":   verdict_info['ai_probability'],
                    "method":           "traditional_heuristic",
                    "metadata":         trad_results.get('metadata'),
                    "visual":           trad_results.get('visual'),
                    "success":          True,
                    "warning": (
                        "Neural models not loaded — using heuristic fallback. "
                        "Accuracy is significantly lower. Train and place "
                        "spatial_model_best.pth and frequency_model_best.pth "
                        f"in {CHECKPOINT_DIR} for full accuracy."
                    ),
                }
                logger.info("Traditional detection complete: %s", verdict_info['label'])
                return jsonify(response)

            except Exception as e:
                logger.error("Traditional detector error: %s", e, exc_info=True)
                return jsonify({
                    "error":   "Both detectors failed",
                    "details": str(e),
                    "success": False,
                }), 500

        return jsonify({
            "error":   "No detection methods available",
            "message": (
                "No trained checkpoints found and traditional detector unavailable. "
                f"Place spatial_model_best.pth and frequency_model_best.pth in {CHECKPOINT_DIR}."
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
    print(" " * 15 + "AI IMAGE DETECTOR - PRODUCTION v3.0")
    print("=" * 70)
    print(f"\n  Server:      http://localhost:5000")
    print(f"  Uploads:     {UPLOAD_FOLDER}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")

    spatial_path   = os.path.join(CHECKPOINT_DIR, 'spatial_model_best.pth')
    frequency_path = os.path.join(CHECKPOINT_DIR, 'frequency_model_best.pth')
    print(f"\n  Spatial checkpoint:   {'FOUND' if os.path.exists(spatial_path)   else 'MISSING'}")
    print(f"  Frequency checkpoint: {'FOUND' if os.path.exists(frequency_path) else 'MISSING'}")

    params = _get_tuning_params()
    print(f"\n  REAL-identification tuning (env vars):")
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
    print("    GET  /health         Health + active detection method + tuning params")
    print("    POST /api/detect     Image detection (neural -> heuristic fallback)")
    print("    GET  /api/models     Model info")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_startup_info()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)