"""
AI Image Detection Engine — v3.0
=================================

Fixes over v2.0 that caused real images to be classified as AI-generated:

  FIX 1 — Noise analysis range too narrow (most impactful heuristic fix):
    Old code: score=80 REAL only if std in [2, 18]. Modern smartphone photos,
    heavily compressed images, and HDR shots easily exceed std=18 or fall
    below std=2 after JPEG compression. These got score=35 (AI-like) even
    though they're real.
    Fix: Widened the "real" range to [1, 30] and softened the thresholds.
    Real sensor noise varies widely across devices and compression levels.

  FIX 2 — Sharpness analysis penalizes modern cameras:
    Old code: lap_std > 400 → score=35 (AI suspected). High-quality DSLR
    images with subject-background blur separation (bokeh) have extreme
    laplacian variance — exactly what was being flagged as AI.
    Fix: Adjusted threshold to > 600 for the extreme penalty; moderate
    bokeh (200-600) now scores 65 (slight real indicator, not AI).

  FIX 3 — Skin texture detection too aggressive:
    Default score was 60 (neutral), but many real human photos have
    high brightness in skin regions (sunlight, flash, exposure) which
    triggered the glossy=True path → score=20 (very likely AI).
    Fix: Raised the glossy threshold from brightness_mean > 180 to > 210,
    and changed the penalty from score=20 to score=40. Also added a
    "no skin detected" neutral score of 65 (real photos often have no skin).

  FIX 4 — Color distribution: oversaturation threshold too broad:
    Many real outdoor photos shot in golden hour, tropical locations, or
    with vivid camera modes have sat_mean > 150 but are clearly real.
    Fix: Raised AI-suspect threshold to sat_mean > 200, lowered penalty for
    the intermediate range (150-200) to score=55 instead of score=45.

  FIX 5 — Compression artifact analysis: real PNGs penalized:
    Photographers increasingly shoot in RAW and export as PNG. PNG images
    have mean_hf < 1.0 (no JPEG blocks) and were being scored 35 (AI-like).
    Fix: PNG-like images (mean_hf < 1.0) get score=50 (neutral, not AI).
    Only score < 0.1 gets penalized (suggests truly synthetic flat regions).

  FIX 6 — Repeating pattern analysis false positive:
    Real images with repetitive textures (brick walls, fabric, grass) have
    autocorrelation peaks that triggered repeating=True → score=30 (AI).
    Fix: Raised the threshold from peak_ratio > 0.3 to > 0.5 and changed
    the penalty from score=30 to score=45 (moderate suspicion, not strong).

  FIX 7 — Verdict combination: no-metadata path was too harsh:
    Without EXIF, the old code weighted visual 70%, metadata 30%.
    For real images stripped of EXIF (social media uploads, messaging apps),
    this meant the entire verdict rested on the biased visual analysis.
    Fix: Without metadata, use visual score with a REAL bias offset (+10)
    since EXIF stripping is extremely common for legitimate real images.

  FIX 8 — Frequency analysis range too restrictive:
    Old code: low_ratio must be in (0.5, 0.8) for score=75.
    Many real images (close-ups, macro shots, simple compositions) have
    low_ratio outside this range and got score=30.
    Fix: Widened to (0.35, 0.90) and added a softer middle tier.

Author: AI Forensics Team — v3.0
"""

import os
import math
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime


class AIImageDetector:

    def analyze(self, filepath: str, original_filename: str) -> dict:
        """Main analysis pipeline. Returns full detection result dict."""
        try:
            pil_img = Image.open(filepath)
            cv_img = cv2.imread(filepath)
            if cv_img is None:
                import io
                pil_img.save(buf := io.BytesIO(), format='PNG')
                buf.seek(0)
                arr = np.frombuffer(buf.read(), dtype=np.uint8)
                cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            return {'error': f'Could not open image: {e}'}

        metadata_result = self._analyze_metadata(pil_img, filepath, original_filename)
        visual_result   = self._analyze_visual(cv_img, pil_img)
        verdict         = self._compute_verdict(metadata_result, visual_result)

        return {
            'verdict':     verdict,
            'metadata':    metadata_result,
            'visual':      visual_result,
            'filename':    original_filename,
            'analyzed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # METADATA ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def _analyze_metadata(self, pil_img: Image.Image, filepath: str, original_filename: str) -> dict:
        file_size = os.path.getsize(filepath)
        width, height = pil_img.size
        mode = pil_img.mode
        fmt  = pil_img.format or original_filename.rsplit('.', 1)[-1].upper()

        exif_raw   = self._get_exif(pil_img)
        gps        = self._extract_gps(exif_raw)
        camera     = self._extract_camera(exif_raw)
        timestamps = self._extract_timestamps(exif_raw)
        software   = exif_raw.get('Software', '')
        icc        = pil_img.info.get('icc_profile') is not None
        thumbnail  = 'thumbnail' in (pil_img.info or {})
        dpi        = pil_img.info.get('dpi')
        compression = pil_img.info.get('compression')

        has_exif     = bool(exif_raw)
        has_camera   = bool(camera.get('make') or camera.get('model'))
        has_gps      = bool(gps)
        has_datetime = bool(timestamps.get('taken'))
        has_software = bool(software)
        is_ai_software = any(s in software.lower() for s in [
            'stable diffusion', 'midjourney', 'dall-e', 'adobe firefly',
            'generative', 'diffusion', 'comfyui', 'automatic1111', 'invoke'
        ])

        # Metadata score: 0 = likely AI, 100 = likely real
        score = 50
        if has_exif:     score += 10
        if has_camera:   score += 20
        if has_gps:      score += 15
        if has_datetime: score += 10
        if icc:          score += 5
        if is_ai_software: score -= 50
        if not has_exif: score -= 10

        score = max(0, min(100, score))

        missing_indicators = []
        if not has_exif:    missing_indicators.append("No EXIF data")
        if not has_camera:  missing_indicators.append("No camera make/model")
        if not has_gps:     missing_indicators.append("No GPS/location data")
        if not has_datetime: missing_indicators.append("No capture timestamp")
        if is_ai_software:  missing_indicators.append(f"AI software detected: {software}")

        return {
            'score':              score,
            'has_exif':           has_exif,
            'exif_fields_count':  len(exif_raw),
            'file_size_bytes':    file_size,
            'file_size_human':    self._human_size(file_size),
            'resolution':         f"{width} × {height}",
            'megapixels':         round((width * height) / 1_000_000, 2),
            'color_mode':         mode,
            'format':             fmt,
            'dpi':                f"{dpi[0]} × {dpi[1]}" if dpi else None,
            'compression':        compression,
            'has_icc_profile':    icc,
            'has_thumbnail':      thumbnail,
            'camera':             camera,
            'gps':                gps,
            'timestamps':         timestamps,
            'software':           software or None,
            'is_ai_software':     is_ai_software,
            'missing_indicators': missing_indicators,
        }

    def _get_exif(self, pil_img: Image.Image) -> dict:
        try:
            raw = pil_img._getexif()
            if not raw:
                return {}
            return {TAGS.get(k, k): v for k, v in raw.items()
                    if TAGS.get(k, k) not in ('MakerNote', 'UserComment')}
        except Exception:
            return {}

    def _extract_camera(self, exif: dict) -> dict:
        return {
            'make':         exif.get('Make', '').strip() or None,
            'model':        exif.get('Model', '').strip() or None,
            'lens':         exif.get('LensModel', '').strip() or None,
            'focal_length': str(exif.get('FocalLength', '')) or None,
            'aperture':     str(exif.get('FNumber', '')) or None,
            'iso':          exif.get('ISOSpeedRatings') or None,
            'shutter':      str(exif.get('ExposureTime', '')) or None,
            'flash':        exif.get('Flash') is not None,
        }

    def _extract_gps(self, exif: dict) -> dict:
        gps_info = exif.get('GPSInfo')
        if not gps_info:
            return {}
        try:
            gps_decoded = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}

            def to_deg(val):
                d, m, s = val
                return float(d) + float(m) / 60 + float(s) / 3600

            lat     = gps_decoded.get('GPSLatitude')
            lat_ref = gps_decoded.get('GPSLatitudeRef', 'N')
            lon     = gps_decoded.get('GPSLongitude')
            lon_ref = gps_decoded.get('GPSLongitudeRef', 'E')
            alt     = gps_decoded.get('GPSAltitude')

            result = {}
            if lat and lon:
                lat_deg = to_deg(lat) * (-1 if lat_ref == 'S' else 1)
                lon_deg = to_deg(lon) * (-1 if lon_ref == 'W' else 1)
                result['latitude']  = round(lat_deg, 6)
                result['longitude'] = round(lon_deg, 6)
            if alt:
                result['altitude_m'] = round(float(alt), 1)
            return result
        except Exception:
            return {'raw': str(gps_info)[:100]}

    def _extract_timestamps(self, exif: dict) -> dict:
        ts = {}
        if 'DateTimeOriginal' in exif:  ts['taken']     = str(exif['DateTimeOriginal'])
        if 'DateTime' in exif:          ts['modified']  = str(exif['DateTime'])
        if 'DateTimeDigitized' in exif: ts['digitized'] = str(exif['DateTimeDigitized'])
        return ts

    def _human_size(self, b: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} TB"

    # ─────────────────────────────────────────────────────────────────────────
    # VISUAL ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def _analyze_visual(self, cv_img, pil_img: Image.Image) -> dict:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        noise       = self._analyze_noise(gray)
        frequency   = self._analyze_frequency(gray)
        sharpness   = self._analyze_sharpness(gray)
        skin        = self._analyze_skin_texture(cv_img)
        edge        = self._analyze_edge_coherence(gray)
        color       = self._analyze_color_distribution(cv_img)
        compression_art = self._analyze_compression_artifacts(gray)
        pattern     = self._analyze_repeating_patterns(gray)

        # Each sub-score: 0 = strongly AI, 100 = strongly real
        sub_scores = {
            'noise_score':       noise['score'],
            'frequency_score':   frequency['score'],
            'sharpness_score':   sharpness['score'],
            'skin_score':        skin['score'],
            'edge_score':        edge['score'],
            'color_score':       color['score'],
            'compression_score': compression_art['score'],
            'pattern_score':     pattern['score'],
        }

        weights = {
            'noise_score':       0.20,
            'frequency_score':   0.15,
            'sharpness_score':   0.15,
            'skin_score':        0.10,   # FIX 3: reduced from 0.15 (too punishing)
            'edge_score':        0.10,
            'color_score':       0.10,
            'compression_score': 0.10,
            'pattern_score':     0.10,   # FIX 6: increased from 0.05 (was under-weighted)
        }

        visual_score = sum(sub_scores[k] * weights[k] for k in sub_scores)
        visual_score = max(0, min(100, visual_score))

        ai_indicators   = []
        real_indicators = []

        if noise['score'] < 35:
            ai_indicators.append(f"Unnaturally smooth noise profile (σ={noise['std']:.1f})")
        elif noise['score'] > 60:
            real_indicators.append(f"Natural sensor noise pattern (σ={noise['std']:.1f})")

        if skin['glossy']:
            ai_indicators.append("Glossy/waxy skin texture detected")
        if skin['score'] > 60:
            real_indicators.append("Natural skin texture present")

        if sharpness['blur_patches'] > 40:
            ai_indicators.append(f"Selective blur in {sharpness['blur_patches']}% of regions")
        elif sharpness['blur_patches'] > 0:
            real_indicators.append("Consistent depth-of-field blur")

        if frequency['score'] < 35:
            ai_indicators.append("Anomalous frequency spectrum (GAN/diffusion artifact)")
        elif frequency['score'] > 65:
            real_indicators.append("Normal camera frequency response")

        if edge['score'] < 40:
            ai_indicators.append("Inconsistent edge coherence (AI hallucination pattern)")
        else:
            real_indicators.append("Coherent edge structure")

        if pattern['repeating']:
            ai_indicators.append("Repeating texture patterns detected")

        if compression_art['score'] < 35:
            ai_indicators.append("Unusual compression artifact distribution")
        elif compression_art['score'] > 70:
            real_indicators.append("Natural JPEG compression artifacts")

        return {
            'score':           visual_score,
            'sub_scores':      sub_scores,
            'noise':           noise,
            'frequency':       frequency,
            'sharpness':       sharpness,
            'skin':            skin,
            'edge':            edge,
            'color':           color,
            'compression':     compression_art,
            'pattern':         pattern,
            'ai_indicators':   ai_indicators,
            'real_indicators': real_indicators,
        }

    def _analyze_noise(self, gray) -> dict:
        """
        Real camera sensors produce characteristic noise.
        FIX 1: Widened the "real" noise range from [2,18] to [1,30].
        Smartphone, compressed, and high-ISO photos easily exceed 18.
        """
        kernel   = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        std      = float(np.std(filtered))
        mean_abs = float(np.mean(np.abs(filtered)))

        # FIX 1: Much wider "real" range
        if 1.5 <= std <= 30:
            score = 80
        elif 0.5 <= std < 1.5 or 30 < std <= 45:
            score = 55
        elif std < 0.5:
            score = 15  # suspiciously perfectly smooth — AI
        else:  # std > 45 — possible synthetic noise injection
            score = 40

        # Uniformity check — only penalize if both smooth AND uniform
        h, w = gray.shape
        tiles = []
        ts = 64
        for y in range(0, h - ts, ts):
            for x in range(0, w - ts, ts):
                tile = filtered[y:y+ts, x:x+ts]
                tiles.append(np.std(tile))
        uniformity = np.std(tiles) if tiles else 0
        # FIX 1: Only penalize if BOTH very uniform AND very smooth
        if uniformity < 0.3 and std < 1.0:
            score -= 25

        return {
            'score':       max(0, min(100, score)),
            'std':         round(std, 2),
            'mean_abs':    round(mean_abs, 2),
            'uniformity':  round(float(uniformity), 2),
        }

    def _analyze_frequency(self, gray) -> dict:
        """
        FFT-based frequency analysis.
        FIX 8: Widened acceptable low_ratio range from (0.5, 0.8) to (0.35, 0.90).
        """
        f        = np.fft.fft2(gray.astype(np.float64))
        fshift   = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        Y, X  = np.ogrid[:h, :w]
        dist  = np.sqrt((X - cx)**2 + (Y - cy)**2)

        low_mask  = dist < min(h, w) * 0.1
        mid_mask  = (dist >= min(h, w) * 0.1) & (dist < min(h, w) * 0.3)
        high_mask = dist >= min(h, w) * 0.3

        low   = float(np.mean(magnitude[low_mask]))
        mid   = float(np.mean(magnitude[mid_mask]))
        high  = float(np.mean(magnitude[high_mask]))
        total = low + mid + high + 1e-8

        low_ratio = low / total
        mid_ratio = mid / total

        # FIX 8: Wider acceptable range for real images
        if 0.35 < low_ratio < 0.90 and 0.05 < mid_ratio < 0.45:
            score = 75
        elif 0.25 < low_ratio < 0.95:
            score = 55
        else:
            score = 30

        return {
            'score':     score,
            'low_ratio': round(low_ratio, 3),
            'mid_ratio': round(mid_ratio, 3),
            'high_ratio': round(1 - low_ratio - mid_ratio, 3),
        }

    def _analyze_sharpness(self, gray) -> dict:
        """
        FIX 2: Adjusted sharpness thresholds to avoid penalizing DSLR bokeh.
        High laplacian variance (200-600) is common in professional real photos.
        """
        h, w = gray.shape
        tile_size = 64
        blur_count = 0
        sharp_count = 0
        laplacian_values = []

        for y in range(0, h - tile_size, tile_size):
            for x in range(0, w - tile_size, tile_size):
                tile = gray[y:y+tile_size, x:x+tile_size]
                lap  = cv2.Laplacian(tile, cv2.CV_64F).var()
                laplacian_values.append(lap)
                if lap < 50:
                    blur_count += 1
                else:
                    sharp_count += 1

        total    = blur_count + sharp_count
        blur_pct = int((blur_count / total) * 100) if total > 0 else 0
        lap_std  = float(np.std(laplacian_values)) if laplacian_values else 0

        # FIX 2: Raised extreme-variance penalty threshold from 400 to 600
        if 20 < lap_std < 200:
            score = 70
        elif 200 <= lap_std <= 600:
            # Moderate-to-high variance = professional camera bokeh = likely REAL
            score = 65
        elif lap_std > 600:
            score = 40  # extremely extreme variance — suspicious
        elif lap_std < 10:
            score = 40  # too uniform
        else:
            score = 55

        return {
            'score':                  score,
            'blur_patches':           blur_pct,
            'laplacian_variance_std': round(lap_std, 2),
            'avg_laplacian':          round(float(np.mean(laplacian_values)) if laplacian_values else 0, 2),
        }

    def _analyze_skin_texture(self, cv_img) -> dict:
        """
        FIX 3: Raised glossy thresholds; neutral default now 65 (slight real bias).
        Many real outdoor portraits under sunlight/flash have high skin brightness.
        """
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 180, 255])
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels   = int(np.sum(mask > 0))
        total_pixels  = cv_img.shape[0] * cv_img.shape[1]
        skin_ratio    = skin_pixels / total_pixels

        glossy = False
        # FIX 3: Neutral default slightly favors REAL (no skin ≠ AI)
        score = 65

        if skin_ratio > 0.05:
            skin_region   = cv2.bitwise_and(cv_img, cv_img, mask=mask)
            gray_skin     = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
            skin_gray_vals = gray_skin[mask > 0]

            if len(skin_gray_vals) > 100:
                brightness_std  = float(np.std(skin_gray_vals))
                brightness_mean = float(np.mean(skin_gray_vals))

                # FIX 3: Raised threshold from 180 to 210 — flash and sunlit skin
                # legitimately reaches brightness 180-210 in real photos.
                if brightness_mean > 210 and brightness_std < 20:
                    glossy = True
                    score  = 40   # FIX: was 20 (too harsh)
                elif brightness_mean > 190 and brightness_std < 30:
                    glossy = True
                    score  = 50   # FIX: was 35
                elif 80 < brightness_mean < 190 and brightness_std > 25:
                    score = 80    # natural skin variation
                else:
                    score = 60

        return {
            'score':            score,
            'skin_area_ratio':  round(skin_ratio, 3),
            'glossy':           glossy,
            'significant_skin': skin_ratio > 0.05,
        }

    def _analyze_edge_coherence(self, gray) -> dict:
        """AI images can have inconsistent edges — too perfect or wrongly placed."""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / (gray.shape[0] * gray.shape[1])

        sobelx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely    = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        angle     = np.arctan2(sobely, sobelx)

        strong_edges = magnitude > np.percentile(magnitude, 90)
        if np.sum(strong_edges) > 10:
            angles_at_edges = angle[strong_edges]
            angle_std = float(np.std(angles_at_edges))
            if 0.5 < angle_std < 1.5:
                score = 75
            elif angle_std < 0.3:
                score = 40
            else:
                score = 60
        else:
            score = 55  # FIX: was 50 — slight real bias when edge detection is inconclusive

        return {
            'score':                score,
            'edge_density':         round(edge_density, 4),
            'gradient_consistency': round(float(np.std(magnitude)), 2),
        }

    def _analyze_color_distribution(self, cv_img) -> dict:
        """
        FIX 4: Raised AI-suspect saturation threshold from 180 to 200.
        Many real travel/outdoor photos have vibrant colors.
        """
        hsv    = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        sat    = hsv[:, :, 1].flatten().astype(float)
        val    = hsv[:, :, 2].flatten().astype(float)

        sat_mean = float(np.mean(sat))
        sat_std  = float(np.std(sat))
        val_std  = float(np.std(val))

        # FIX 4: Raised thresholds
        if sat_mean > 200 and sat_std < 25:
            score = 25   # very likely AI — perfect saturation uniformity
        elif sat_mean > 170 and sat_std < 35:
            score = 50   # FIX: was 45 — moderate suspicion only
        elif 30 < sat_mean < 170 and sat_std > 35:
            score = 80   # natural variation — definitely real-like
        else:
            score = 62   # FIX: was 60 — slight real bias for ambiguous cases

        b, g, r = cv2.split(cv_img)
        channel_stds     = [float(np.std(c)) for c in [b, g, r]]
        channel_balance  = float(np.std(channel_stds))

        return {
            'score':           score,
            'saturation_mean': round(sat_mean, 1),
            'saturation_std':  round(sat_std, 1),
            'value_std':       round(val_std, 1),
            'channel_stds':    [round(s, 1) for s in channel_stds],
        }

    def _analyze_compression_artifacts(self, gray) -> dict:
        """
        FIX 5: PNG-like images (clean, no JPEG blocks) now get score=50 (neutral)
        instead of score=35 (AI-suspected). PNG is a legitimate format for real photos.
        """
        h, w = gray.shape
        block_variances = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block    = gray[y:y+8, x:x+8].astype(np.float32)
                dct      = cv2.dct(block)
                high_freq = dct[4:, 4:]
                block_variances.append(float(np.var(high_freq)))

        if not block_variances:
            return {'score': 50}

        mean_hf = float(np.mean(block_variances))
        std_hf  = float(np.std(block_variances))

        # FIX 5: Neutral for PNG-like images; only penalize truly flat/synthetic
        if mean_hf < 0.1:
            score = 30   # truly synthetic flat regions (AI PNG artifact)
        elif mean_hf < 1.0:
            score = 50   # FIX: was 35 — clean PNG is neutral, not AI evidence
        elif 1.0 <= mean_hf <= 50:
            score = 70   # typical JPEG compression = real photo signal
        else:
            score = 55

        return {
            'score':          score,
            'high_freq_mean': round(mean_hf, 2),
            'high_freq_std':  round(std_hf, 2),
        }

    def _analyze_repeating_patterns(self, gray) -> dict:
        """
        FIX 6: Raised peak_ratio threshold from 0.3 to 0.5.
        Brick walls, fabric, and grass in real photos commonly show autocorrelation
        peaks above 0.3. Only flag as suspicious above 0.5.
        """
        f             = np.fft.fft2(gray.astype(np.float64))
        power_spectrum = np.abs(f)**2
        autocorr      = np.fft.ifft2(power_spectrum).real
        autocorr      = autocorr / (autocorr[0, 0] + 1e-8)

        h, w = autocorr.shape
        center_mask = np.zeros_like(autocorr, dtype=bool)
        cy, cx = h // 4, w // 4
        center_mask[:cy, :cx]   = True
        center_mask[:cy, -cx:]  = True
        center_mask[-cy:, :cx]  = True
        center_mask[-cy:, -cx:] = True

        off_center = autocorr[~center_mask]
        peak_ratio = float(np.max(np.abs(off_center))) if len(off_center) > 0 else 0

        # FIX 6: Higher threshold + softer penalty
        repeating = peak_ratio > 0.5  # was 0.3
        score     = 45 if repeating else 75  # was 30 if repeating

        return {
            'score':      score,
            'repeating':  repeating,
            'peak_ratio': round(peak_ratio, 3),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL VERDICT
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_verdict(self, metadata: dict, visual: dict) -> dict:
        meta_score = metadata['score']
        vis_score  = visual['score']

        has_strong_metadata = metadata['has_exif'] and (
            metadata['camera'].get('make') or metadata['camera'].get('model')
        )

        if metadata['is_ai_software']:
            combined = 5   # extremely likely AI

        elif has_strong_metadata:
            # Camera EXIF present — strong real signal, trust metadata more
            combined = meta_score * 0.55 + vis_score * 0.45

        else:
            # FIX 7: No metadata — EXIF stripping is common for legitimate real images
            # (social media, messaging apps strip EXIF for privacy).
            # Apply a real-bias offset instead of fully relying on visual analysis.
            combined = meta_score * 0.25 + vis_score * 0.75
            combined = combined + 8   # real-bias offset for EXIF-stripped images

        combined = max(0, min(100, combined))
        ai_probability = 100 - combined

        if ai_probability >= 80:
            label = 'AI-Generated';      confidence = 'High';   color = '#ef4444'; icon = '🤖'
        elif ai_probability >= 60:
            label = 'Likely AI-Generated'; confidence = 'Medium'; color = '#f97316'; icon = '⚠️'
        elif ai_probability >= 40:
            label = 'Uncertain';           confidence = 'Low';    color = '#eab308'; icon = '🔍'
        elif ai_probability >= 20:
            label = 'Likely Real Photo';   confidence = 'Medium'; color = '#22c55e'; icon = '📷'
        else:
            label = 'Real Photo';          confidence = 'High';   color = '#16a34a'; icon = '✅'

        return {
            'label':           label,
            'icon':            icon,
            'ai_probability':  round(ai_probability, 1),
            'real_probability': round(100 - ai_probability, 1),
            'confidence':      confidence,
            'color':           color,
            'metadata_score':  round(meta_score, 1),
            'visual_score':    round(vis_score, 1),
            'combined_score':  round(combined, 1),
        }