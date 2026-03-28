# AI Image Detector

A Python web application that detects whether an image is real or AI-generated using two complementary methods:

1. **Metadata / EXIF Analysis** — real cameras embed rich metadata (GPS, make/model, timestamps, ICC profiles). AI tools typically strip or omit this.
2. **Visual Pattern Analysis** — AI images show distinctive artifacts: unnaturally smooth noise, glossy skin, selective blur (hair/eyes/hands), unusual frequency spectra, and repeating textures.

---

## Project Structure

```
ai_detector/
├── app.py              ← Flask web server & API routes
├── detector.py         ← Detection engine (all analysis logic)
├── requirements.txt    ← Python dependencies
├── templates/
│   └── index.html      ← Standalone HTML frontend (NO Django template tags)
├── uploads/            ← Temp folder (auto-created, files deleted after analysis)
└── README.md
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install Flask Pillow numpy opencv-python-headless Werkzeug
```

### 2. Run the server

```bash
python app.py
```

### 3. Open in browser

```
http://localhost:5000
```

---

## How Detection Works

### Metadata Analysis (contributes ~35–55% of final score)

| Signal | Real Photo | AI Image |
|--------|-----------|----------|
| EXIF data | ✅ Present (dozens of fields) | ❌ Usually absent |
| Camera make/model | ✅ e.g. "Canon EOS R5" | ❌ None |
| GPS coordinates | ✅ Often present | ❌ None |
| Capture timestamp | ✅ Exact date/time | ❌ None |
| ICC color profile | ✅ Usually embedded | ❌ Often absent |
| AI software tag | — | ❌ "Stable Diffusion", "DALL-E", etc. |

### Visual Analysis (contributes ~45–70% of final score)

| Check | What we look for | Why it matters |
|-------|-----------------|----------------|
| **Noise Profile** | Sensor noise std deviation & uniformity | Real cameras have consistent shot noise; AI images are too smooth or have synthetic noise |
| **Frequency Spectrum** | FFT magnitude distribution (low/mid/high ratios) | Camera optics create characteristic frequency falloff; diffusion models deviate |
| **Sharpness Distribution** | Tile-by-tile Laplacian variance | AI often has extreme selective blur — sharp faces, blurred hands/hair |
| **Skin Texture** | Brightness mean/std in skin-colored regions | AI produces waxy/glossy skin with low local variance |
| **Edge Coherence** | Sobel gradient direction consistency | Real objects have physically consistent edges; AI hallucinations don't |
| **Color Distribution** | HSV saturation stats | AI oversaturates with low variance; real photos have natural variation |
| **Compression Artifacts** | DCT high-frequency block analysis | Real JPEGs have characteristic compression patterns |
| **Repeating Patterns** | Autocorrelation peak analysis | Diffusion models sometimes tile textures in backgrounds |

### Scoring

- Each check produces a **0–100 score** (100 = strongly real, 0 = strongly AI)
- Weighted combination produces a final `combined_score`
- `AI probability = 100 − combined_score`

---

## API

### POST `/api/detect`

Upload an image for analysis.

**Request:** `multipart/form-data` with field `image`

**Response:**
```json
{
  "verdict": {
    "label": "Likely AI-Generated",
    "icon": "⚠️",
    "ai_probability": 72.4,
    "real_probability": 27.6,
    "confidence": "Medium",
    "color": "#f97316",
    "metadata_score": 30.0,
    "visual_score": 38.5,
    "combined_score": 27.6
  },
  "metadata": {
    "score": 30.0,
    "has_exif": false,
    "exif_fields_count": 0,
    "file_size_human": "2.1 MB",
    "resolution": "1024 × 1024",
    "camera": { "make": null, "model": null },
    "gps": {},
    "timestamps": {},
    "missing_indicators": ["No EXIF data", "No camera make/model", "No GPS/location data"]
  },
  "visual": {
    "score": 38.5,
    "ai_indicators": ["Unnaturally smooth noise profile", "Glossy/waxy skin texture detected"],
    "real_indicators": [],
    "sub_scores": { ... }
  },
  "filename": "portrait.png",
  "analyzed_at": "2026-02-23 14:30:22"
}
```

---

## Notes

- Images are **never stored permanently** — temp files are deleted immediately after analysis
- No external AI API calls — all analysis runs locally
- The HTML frontend uses **zero Django/Jinja template syntax** — it's pure HTML/CSS/JS
- Works offline after initial install
