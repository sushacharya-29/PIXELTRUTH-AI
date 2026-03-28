"""
PRODUCTION Inference Module for AI Image Detection
===================================================

v3.0 — Fixes REAL-image identification failure + PyTorch 2.x compatibility.

Fixes over v2.1:
  FIX 1 — torch.cuda.amp.autocast() deprecated (PyTorch 2.x):
    Old: with torch.cuda.amp.autocast():
    New: with torch.amp.autocast(device_type=device.type):
    The old API silently falls back to CPU mode on PyTorch 2.x, which means
    AMP is effectively disabled but no error is raised. This can cause subtle
    numerical differences that shift the decision boundary.

  FIX 2 — Temperature scaling support:
    Models trained with high REAL class weights can be over-confident on FAKE
    (because FAKE artifacts are very distinctive). After training, if REAL images
    still get classified as FAKE with high confidence, apply temperature > 1.0
    to soften predictions. This is set via environment variables or constructor
    args — no retraining required.

  FIX 3 — Per-class probability logging:
    Added FAKE/REAL probability logging at INFO level so you can see if the
    model is collapsed (e.g., always outputs FAKE=99%, REAL=1%).

  FIX 4 — weights_only=False for checkpoint compatibility:
    PyTorch 2.6+ changed default to weights_only=True which breaks loading
    checkpoints that include config dicts. Explicitly set weights_only=False.

  FIX 5 — Ensemble threshold calibration:
    Old ensemble treated REAL if real_prob > fake_prob (threshold=50%).
    For models that tend to under-predict REAL, a lower threshold (e.g., 40%)
    gives better REAL recall. Added configurable real_threshold parameter.

Author: AI Forensics Team
Version: 3.0 (Production — Real-image fix)
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import logging
from typing import Optional, Dict, Union, List
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# CIFAKE normalization constants (matches training)
# NOT ImageNet — these are CIFAR-10 pixel statistics.
# ──────────────────────────────────────────────────────────────
CIFAKE_MEAN = [0.4914, 0.4822, 0.4465]
CIFAKE_STD  = [0.2470, 0.2435, 0.2616]


class AIImageDetectorNN:
    """
    Production-ready AI Image Detector using Neural Networks.

    Label convention (CIFAKE ImageFolder alphabetical order):
        class 0 -> FAKE  (AI-Generated)
        class 1 -> REAL

    Two separate preprocessing pipelines:
        spatial_transform   -- CIFAKE normalization (not ImageNet)
        frequency_transform -- ToTensor() only, NO normalize
                               (FFT must see raw [0,1] pixel values)

    Temperature scaling:
        If the model under-predicts REAL (real images classified as FAKE
        with high confidence), set spatial_temperature or freq_temperature > 1.0.
        Try 1.5 or 2.0. This requires no retraining.

    Real threshold:
        If models are balanced but still slightly biased toward FAKE,
        lower real_threshold below 0.5 (e.g., 0.45) to classify more images as REAL.
    """

    def __init__(
        self,
        spatial_weights:    Optional[str] = None,
        frequency_weights:  Optional[str] = None,
        device:             Optional[str] = None,
        use_amp:            bool = True,
        spatial_temperature:  float = 1.0,
        freq_temperature:     float = 1.0,
        real_threshold:       float = 0.5,
    ):
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # FIX 1: AMP only active on CUDA
        self.use_amp = use_amp and (self.device.type == 'cuda')
        self.spatial_temperature  = spatial_temperature
        self.freq_temperature     = freq_temperature
        # FIX 5: Configurable threshold — lower to improve REAL recall
        self.real_threshold       = real_threshold

        logger.info(f"Initializing AI Image Detector on {self.device}")
        if self.use_amp:
            logger.info("Using automatic mixed precision (FP16)")
        if real_threshold != 0.5:
            logger.info(f"Real threshold: {real_threshold} (lowered from 0.5 to improve REAL recall)")

        self.spatial_model    = None
        self.frequency_model  = None

        if spatial_weights:
            self.spatial_model = self._load_model(
                'spatial', spatial_weights,
                lambda: self._get_spatial_model(spatial_temperature)
            )

        if frequency_weights:
            self.frequency_model = self._load_model(
                'frequency', frequency_weights,
                lambda: self._get_frequency_model(freq_temperature)
            )

        if self.spatial_model is None and self.frequency_model is None:
            logger.warning("No models loaded! Detector will fail on inference.")
            logger.warning("Provide checkpoint paths or train models first.")

        # Spatial: CIFAKE normalization, NOT ImageNet
        self.spatial_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAKE_MEAN, std=CIFAKE_STD),
        ])

        # Frequency: raw [0,1] — NO Normalize (FFT operates on raw pixels)
        self.frequency_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        logger.info("Detector initialized successfully")

    def _get_spatial_model(self, temperature: float = 1.0):
        """Create spatial model with training-matching dropout and temperature."""
        try:
            from spatial_model import get_spatial_model
            return get_spatial_model(num_classes=2, dropout=0.3, temperature=temperature)
        except ImportError as e:
            logger.error(f"Cannot import spatial_model: {e}")
            return None

    def _get_frequency_model(self, temperature: float = 1.0):
        """Create frequency model with training-matching dropout and temperature."""
        try:
            from frequency_model import get_frequency_model
            return get_frequency_model(
                num_classes=2,
                dropout=0.25,
                process_in_model=True,
                temperature=temperature,
            )
        except ImportError as e:
            logger.error(f"Cannot import frequency_model: {e}")
            return None

    def _load_model(
        self,
        model_name: str,
        weights_path: str,
        model_factory,
    ) -> Optional[torch.nn.Module]:
        """Safely load a model with comprehensive error handling."""
        try:
            if not os.path.exists(weights_path):
                logger.error(f"[{model_name}] checkpoint not found: {weights_path}")
                return None

            model = model_factory()
            if model is None:
                logger.error(f"[{model_name}] Failed to create model instance")
                return None

            logger.info(f"[{model_name}] Loading from {weights_path}...")
            # FIX 4: weights_only=False — needed for checkpoints with config dicts
            checkpoint = torch.load(
                weights_path,
                map_location=self.device,
                weights_only=False,
            )

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    logger.warning(
                        f"[{model_name}] No 'model_state_dict' key found. "
                        f"Available keys: {list(checkpoint.keys())}. "
                        f"Attempting to treat checkpoint as raw state_dict."
                    )
                    state_dict = checkpoint
            else:
                logger.error(f"[{model_name}] Unexpected checkpoint type: {type(checkpoint)}")
                return None

            model.load_state_dict(state_dict, strict=True)
            model = model.to(self.device)
            model.eval()

            if isinstance(checkpoint, dict):
                if 'metrics' in checkpoint:
                    m = checkpoint['metrics']
                    fake_acc = m.get('fake_accuracy', 'N/A')
                    real_acc = m.get('real_accuracy', 'N/A')
                    bal_acc  = m.get('balanced_accuracy', 'N/A')
                    logger.info(
                        f"[{model_name}] Checkpoint — "
                        f"FAKE_acc={fake_acc}  REAL_acc={real_acc}  balanced={bal_acc}"
                    )
                    # Warn if loaded from an old v2.0 checkpoint without per-class metrics
                    if fake_acc == 'N/A' and real_acc == 'N/A':
                        logger.warning(
                            f"[{model_name}] Checkpoint has no per-class accuracy metrics. "
                            f"This checkpoint was likely saved by trainingScript v2.0 or earlier. "
                            f"If REAL identification fails, retrain with v3.0."
                        )
                    elif isinstance(real_acc, (int, float)) and real_acc < 70:
                        logger.warning(
                            f"[{model_name}] REAL accuracy in checkpoint is low ({real_acc}%). "
                            f"Consider retraining or using a higher real_threshold / temperature."
                        )
                if 'epoch' in checkpoint:
                    logger.info(f"[{model_name}] Trained for {checkpoint['epoch'] + 1} epoch(s)")

            logger.info(f"[{model_name}] Loaded successfully")
            return model

        except Exception as e:
            logger.error(f"[{model_name}] Error loading: {e}")
            logger.exception("Full traceback:")
            return None

    def _preprocess_spatial(
        self, image_input: Union[str, Path, Image.Image]
    ) -> torch.Tensor:
        """Preprocess for spatial model: CIFAKE normalization."""
        try:
            if isinstance(image_input, (str, Path)):
                image = Image.open(image_input).convert('RGB')
            else:
                image = image_input.convert('RGB')
            return self.spatial_transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Spatial preprocessing failed: {e}")

    def _preprocess_frequency(
        self, image_input: Union[str, Path, Image.Image]
    ) -> torch.Tensor:
        """Preprocess for frequency model: raw [0,1] — NO normalization."""
        try:
            if isinstance(image_input, (str, Path)):
                image = Image.open(image_input).convert('RGB')
            else:
                image = image_input.convert('RGB')
            return self.frequency_transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Frequency preprocessing failed: {e}")

    @torch.no_grad()
    def _predict_single_model(
        self,
        image_tensor: torch.Tensor,
        model: torch.nn.Module,
    ) -> tuple:
        """
        Get prediction from a single model.

        CIFAKE label order (ImageFolder alphabetical):
            index 0 -> FAKE  (AI-Generated)
            index 1 -> REAL

        Returns:
            (real_probability, fake_probability) as floats in [0, 1]
        """
        try:
            # FIX 1: device_type-aware autocast (PyTorch 2.x compatible)
            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type):
                    logits = model(image_tensor)
            else:
                logits = model(image_tensor)

            probabilities = F.softmax(logits, dim=1)

            # CIFAKE: class 0 = FAKE, class 1 = REAL
            fake_prob = probabilities[0, 0].item()
            real_prob = probabilities[0, 1].item()

            # FIX 3: Log both probabilities so collapse is visible in logs
            logger.debug(f"Model output — FAKE: {fake_prob:.3f}  REAL: {real_prob:.3f}")

            return real_prob, fake_prob

        except Exception as e:
            logger.error(f"Model inference error: {e}")
            raise RuntimeError(f"Model inference failed: {e}")

    def _verdict_from_probs(self, real_prob: float, fake_prob: float) -> str:
        """
        FIX 5: Apply configurable real_threshold instead of hardcoded 0.5.
        If real_prob >= real_threshold → REAL, else → AI-GENERATED.
        """
        return 'REAL' if real_prob >= self.real_threshold else 'AI-GENERATED'

    def detect(
        self,
        image_path: Union[str, Path],
        ensemble_method: str = 'weighted_average',
    ) -> Dict:
        """
        Detect if image is AI-generated.

        Args:
            image_path:      Path to image file
            ensemble_method: 'weighted_average', 'voting', or 'max_confidence'

        Returns:
            Dict with keys: 'spatial', 'frequency', 'ensemble', 'success'
        """
        if not os.path.exists(str(image_path)):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not self.spatial_model and not self.frequency_model:
            raise RuntimeError("No models loaded! Cannot perform detection.")

        results = {
            'spatial':   None,
            'frequency': None,
            'ensemble':  None,
            'success':   True,
        }

        # ── Spatial model prediction ──
        if self.spatial_model:
            try:
                spatial_tensor = self._preprocess_spatial(image_path)
                real_prob, fake_prob = self._predict_single_model(
                    spatial_tensor, self.spatial_model
                )
                verdict = self._verdict_from_probs(real_prob, fake_prob)
                results['spatial'] = {
                    'real_probability': round(real_prob * 100, 2),
                    'ai_probability':   round(fake_prob * 100, 2),
                    'verdict':          verdict,
                }
                logger.info(
                    f"[spatial] real={real_prob*100:.1f}%  "
                    f"fake={fake_prob*100:.1f}%  verdict={verdict}"
                )
                # FIX 3: Warn on potential collapse
                if fake_prob > 0.95:
                    logger.warning(
                        f"[spatial] Very high FAKE confidence ({fake_prob*100:.1f}%). "
                        f"If this is a real image, model may be collapsed. "
                        f"Try spatial_temperature > 1.0 or retrain with higher real_weight."
                    )
            except Exception as e:
                logger.error(f"Spatial model prediction failed: {e}")
                results['spatial'] = {'error': str(e)}

        # ── Frequency model prediction ──
        if self.frequency_model:
            try:
                freq_tensor = self._preprocess_frequency(image_path)
                real_prob, fake_prob = self._predict_single_model(
                    freq_tensor, self.frequency_model
                )
                verdict = self._verdict_from_probs(real_prob, fake_prob)
                results['frequency'] = {
                    'real_probability': round(real_prob * 100, 2),
                    'ai_probability':   round(fake_prob * 100, 2),
                    'verdict':          verdict,
                }
                logger.info(
                    f"[frequency] real={real_prob*100:.1f}%  "
                    f"fake={fake_prob*100:.1f}%  verdict={verdict}"
                )
                if fake_prob > 0.95:
                    logger.warning(
                        f"[frequency] Very high FAKE confidence ({fake_prob*100:.1f}%). "
                        f"If this is a real image, try freq_temperature > 1.0."
                    )
            except Exception as e:
                logger.error(f"Frequency model prediction failed: {e}")
                results['frequency'] = {'error': str(e)}

        # ── Ensemble ──
        results['ensemble'] = self._compute_ensemble(
            results['spatial'],
            results['frequency'],
            ensemble_method,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _compute_ensemble(
        self,
        spatial_result:   Optional[Dict],
        frequency_result: Optional[Dict],
        method: str,
    ) -> Dict:
        """
        Compute ensemble prediction from individual model results.
        Weights: spatial=0.55, frequency=0.45.
        FIX 5: Uses self.real_threshold for final verdict.
        """
        has_spatial   = bool(spatial_result)   and 'error' not in spatial_result
        has_frequency = bool(frequency_result) and 'error' not in frequency_result

        if not has_spatial and not has_frequency:
            return {
                'error':            'Both models failed',
                'real_probability': 50.0,
                'ai_probability':   50.0,
                'verdict':          'UNCERTAIN',
                'confidence':       'None',
            }

        if not has_spatial:
            logger.warning("Ensemble: using only frequency model (spatial failed/unavailable)")
            out = dict(frequency_result)
            out['confidence'] = self._confidence_label(
                max(frequency_result['real_probability'], frequency_result['ai_probability'])
            )
            out['method'] = 'frequency_only'
            return out

        if not has_frequency:
            logger.warning("Ensemble: using only spatial model (frequency failed/unavailable)")
            out = dict(spatial_result)
            out['confidence'] = self._confidence_label(
                max(spatial_result['real_probability'], spatial_result['ai_probability'])
            )
            out['method'] = 'spatial_only'
            return out

        # Both models available
        if method == 'weighted_average':
            sw, fw = 0.55, 0.45
            ensemble_real = (
                spatial_result['real_probability'] * sw +
                frequency_result['real_probability'] * fw
            )
            ensemble_fake = (
                spatial_result['ai_probability'] * sw +
                frequency_result['ai_probability'] * fw
            )

        elif method == 'voting':
            sv = 1 if spatial_result['verdict']   == 'REAL' else 0
            fv = 1 if frequency_result['verdict'] == 'REAL' else 0
            if (sv + fv) >= 1:
                ensemble_real, ensemble_fake = 75.0, 25.0
            else:
                ensemble_real, ensemble_fake = 25.0, 75.0

        elif method == 'max_confidence':
            sc = max(spatial_result['real_probability'],   spatial_result['ai_probability'])
            fc = max(frequency_result['real_probability'], frequency_result['ai_probability'])
            if sc >= fc:
                ensemble_real = spatial_result['real_probability']
                ensemble_fake = spatial_result['ai_probability']
            else:
                ensemble_real = frequency_result['real_probability']
                ensemble_fake = frequency_result['ai_probability']

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        confidence_score = max(ensemble_real, ensemble_fake)

        # FIX 5: Apply real_threshold for final verdict
        threshold_pct = self.real_threshold * 100
        verdict = 'REAL' if ensemble_real >= threshold_pct else 'AI-GENERATED'

        logger.info(
            f"[ensemble/{method}] real={ensemble_real:.1f}%  "
            f"fake={ensemble_fake:.1f}%  threshold={threshold_pct:.0f}%  verdict={verdict}"
        )
        return {
            'real_probability': round(ensemble_real, 2),
            'ai_probability':   round(ensemble_fake, 2),
            'verdict':          verdict,
            'confidence':       self._confidence_label(confidence_score),
            'confidence_score': round(confidence_score, 2),
            'method':           method,
        }

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 90:   return 'Very High'
        elif score >= 75: return 'High'
        elif score >= 60: return 'Medium'
        else:             return 'Low'

    def batch_detect(
        self,
        image_paths: List[Union[str, Path]],
        ensemble_method: str = 'weighted_average',
    ) -> List[Dict]:
        """Detect multiple images in batch."""
        results = []
        for image_path in image_paths:
            try:
                result = self.detect(image_path, ensemble_method)
                result['filename'] = os.path.basename(str(image_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'filename': os.path.basename(str(image_path)),
                    'error':   str(e),
                    'success': False,
                })
        return results

    def calibrate_temperature(
        self,
        val_image_paths: List[Union[str, Path]],
        val_labels: List[int],
        search_temps: Optional[List[float]] = None,
    ) -> Dict:
        """
        Post-hoc temperature calibration on a validation set.

        Finds the temperature value that maximises balanced accuracy
        (average of FAKE recall + REAL recall) on the provided validation set.

        Args:
            val_image_paths: List of image file paths
            val_labels:      List of ground truth labels (0=FAKE, 1=REAL)
            search_temps:    Temperatures to try (default: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

        Returns:
            Dict with best_spatial_temp, best_freq_temp, per_temp_results
        """
        if search_temps is None:
            search_temps = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]

        logger.info(f"Temperature calibration: {len(val_image_paths)} images, temps={search_temps}")
        results = {}

        for temp in search_temps:
            if self.spatial_model:
                self.spatial_model.set_temperature(temp)
            if self.frequency_model:
                self.frequency_model.set_temperature(temp)

            preds  = []
            labels = []
            for path, label in zip(val_image_paths, val_labels):
                try:
                    result = self.detect(path)
                    ens = result.get('ensemble', {})
                    if 'verdict' not in ens:
                        continue
                    pred = 1 if ens['verdict'] == 'REAL' else 0
                    preds.append(pred)
                    labels.append(label)
                except Exception:
                    continue

            if not preds:
                continue

            preds  = np.array(preds)
            labels = np.array(labels)
            fake_mask = labels == 0
            real_mask = labels == 1
            fake_acc = np.mean(preds[fake_mask] == labels[fake_mask]) if fake_mask.any() else 0.0
            real_acc = np.mean(preds[real_mask] == labels[real_mask]) if real_mask.any() else 0.0
            bal_acc  = (fake_acc + real_acc) / 2
            results[temp] = {
                'balanced_accuracy': bal_acc,
                'fake_accuracy': fake_acc,
                'real_accuracy': real_acc,
            }
            logger.info(
                f"  temp={temp}  balanced={bal_acc*100:.1f}%  "
                f"FAKE={fake_acc*100:.1f}%  REAL={real_acc*100:.1f}%"
            )

        best_temp = max(results, key=lambda t: results[t]['balanced_accuracy'])
        logger.info(f"Best temperature: {best_temp}  ({results[best_temp]})")

        # Apply best temperature
        if self.spatial_model:
            self.spatial_model.set_temperature(best_temp)
        if self.frequency_model:
            self.frequency_model.set_temperature(best_temp)
        self.spatial_temperature = best_temp
        self.freq_temperature    = best_temp

        return {
            'best_temperature': best_temp,
            'per_temp_results': results,
        }

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            'spatial_loaded':         self.spatial_model is not None,
            'frequency_loaded':       self.frequency_model is not None,
            'device':                 str(self.device),
            'amp_enabled':            self.use_amp,
            'cuda_available':         torch.cuda.is_available(),
            'spatial_transform':      'CIFAKE normalization (mean=[0.4914,0.4822,0.4465])',
            'frequency_transform':    'Raw [0,1] — no normalization (FFT-safe)',
            'label_convention':       'class 0=FAKE (AI-Generated), class 1=REAL',
            'spatial_temperature':    self.spatial_temperature,
            'freq_temperature':       self.freq_temperature,
            'real_threshold':         self.real_threshold,
        }


def get_detector(
    spatial_weights:   Optional[str] = None,
    frequency_weights: Optional[str] = None,
    device:            Optional[str] = None,
    spatial_temperature: float = 1.0,
    freq_temperature:    float = 1.0,
    real_threshold:      float = 0.5,
) -> AIImageDetectorNN:
    """
    Factory function — backwards compatible.

    Tip: If real images are being classified as FAKE, try:
        detector = get_detector(..., spatial_temperature=1.5, freq_temperature=1.5)
    Or lower the threshold:
        detector = get_detector(..., real_threshold=0.45)
    """
    return AIImageDetectorNN(
        spatial_weights=spatial_weights,
        frequency_weights=frequency_weights,
        device=device,
        spatial_temperature=spatial_temperature,
        freq_temperature=freq_temperature,
        real_threshold=real_threshold,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("PRODUCTION INFERENCE MODULE v3.0 — SELF-TEST")
    print("=" * 70)
    detector = AIImageDetectorNN()
    info = detector.get_model_info()
    print(f"\n  Device:              {info['device']}")
    print(f"  CUDA:                {info['cuda_available']}")
    print(f"  AMP:                 {info['amp_enabled']}")
    print(f"  Spatial transform:   {info['spatial_transform']}")
    print(f"  Frequency transform: {info['frequency_transform']}")
    print(f"  Label convention:    {info['label_convention']}")
    print(f"  Spatial temperature: {info['spatial_temperature']}")
    print(f"  Freq temperature:    {info['freq_temperature']}")
    print(f"  Real threshold:      {info['real_threshold']}")
    print("\nSelf-test complete.")
    print("=" * 70)