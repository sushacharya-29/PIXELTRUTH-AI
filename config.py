"""
Configuration Module - AI Image Detection System
================================================

Centralized configuration for all components.
Supports environment variables for production deployment.
"""

import os
from pathlib import Path

# ============================================
# Base Paths
# ============================================

BASE_DIR = Path(__file__).parent.resolve()
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# ============================================
# Model Configuration
# ============================================

class ModelConfig:
    """Neural network model configuration"""
    
    # Model Architecture
    IMAGE_SIZE = 128
    NUM_CLASSES = 2
    SPATIAL_DROPOUT = 0.5
    FREQUENCY_DROPOUT = 0.4
    
    # Checkpoint Paths
    SPATIAL_CHECKPOINT = CHECKPOINT_DIR / "spatial_model_best.pth"
    FREQUENCY_CHECKPOINT = CHECKPOINT_DIR / "frequency_model_best.pth"
    
    # Device Configuration
    DEVICE = os.environ.get("DEVICE", "auto")  # 'auto', 'cuda', or 'cpu'
    USE_AMP = os.environ.get("USE_AMP", "true").lower() == "true"
    
    # Ensemble Configuration
    ENSEMBLE_METHOD = "weighted_average"  # 'weighted_average', 'voting', 'max_confidence'
    SPATIAL_WEIGHT = 0.55
    FREQUENCY_WEIGHT = 0.45

# ============================================
# Training Configuration
# ============================================

class TrainingConfig:
    """Training pipeline configuration"""
    
    # Data
    DATA_ROOT = BASE_DIR / "CIFAKE"
    TRAIN_DIR = DATA_ROOT / "train"
    TEST_DIR = DATA_ROOT / "test"
    
    # Hyperparameters
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
    NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "30"))
    LEARNING_RATE = float(os.environ.get("LR", "0.001"))
    WEIGHT_DECAY = 1e-4
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # Optimization
    USE_MIXED_PRECISION = True
    NUM_WORKERS = 0  # Windows compatibility
    PIN_MEMORY = False  # Windows compatibility
    
    # Scheduling
    SCHEDULER_TYPE = "cosine"  # 'cosine' or 'step'
    WARMUP_EPOCHS = 3
    
    # Regularization
    EARLY_STOPPING_PATIENCE = 5
    MIN_DELTA = 0.001
    
    # Checkpointing
    SAVE_DIR = CHECKPOINT_DIR
    SAVE_BEST_ONLY = False  # Save both best and latest
    LOG_INTERVAL = 50
    VAL_INTERVAL = 1

# ============================================
# Flask Application Configuration
# ============================================

class FlaskConfig:
    """Flask web server configuration"""
    
    # Server
    HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
    PORT = int(os.environ.get("FLASK_PORT", "5000"))
    DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    
    # Upload
    UPLOAD_FOLDER = UPLOAD_DIR
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}
    
    # Security
    SECRET_KEY = os.environ.get("SECRET_KEY", "pixel-truth-production-secret-v2")
    
    # Directories
    TEMPLATE_FOLDER = TEMPLATE_DIR
    STATIC_FOLDER = STATIC_DIR

# ============================================
# Logging Configuration
# ============================================

class LoggingConfig:
    """Logging configuration"""
    
    LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Log Files
    LOG_DIR = BASE_DIR / "logs"
    APP_LOG = LOG_DIR / "app.log"
    ERROR_LOG = LOG_DIR / "error.log"

# ============================================
# Production Settings
# ============================================

class ProductionConfig:
    """Production deployment configuration"""
    
    # Performance
    ENABLE_CACHING = os.environ.get("ENABLE_CACHING", "false").lower() == "true"
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "3600"))  # seconds
    
    # Monitoring
    ENABLE_METRICS = os.environ.get("ENABLE_METRICS", "false").lower() == "true"
    METRICS_PORT = int(os.environ.get("METRICS_PORT", "9090"))
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT", "false").lower() == "true"
    RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))

# ============================================
# Helper Functions
# ============================================

def create_directories():
    """Create required directories if they don't exist"""
    dirs = [
        CHECKPOINT_DIR,
        UPLOAD_DIR,
        TEMPLATE_DIR,
        STATIC_DIR,
        LoggingConfig.LOG_DIR
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

def get_config_summary():
    """Get configuration summary for logging"""
    return {
        "model": {
            "device": ModelConfig.DEVICE,
            "amp_enabled": ModelConfig.USE_AMP,
            "image_size": ModelConfig.IMAGE_SIZE
        },
        "training": {
            "batch_size": TrainingConfig.BATCH_SIZE,
            "epochs": TrainingConfig.NUM_EPOCHS,
            "learning_rate": TrainingConfig.LEARNING_RATE
        },
        "flask": {
            "host": FlaskConfig.HOST,
            "port": FlaskConfig.PORT,
            "debug": FlaskConfig.DEBUG
        }
    }

# ============================================
# Initialization
# ============================================

# Create directories on import
create_directories()

if __name__ == "__main__":
    import json
    print("="*70)
    print(" "*20 + "CONFIGURATION SUMMARY")
    print("="*70)
    print(json.dumps(get_config_summary(), indent=2))
    print("="*70)
