# 🔍 PIXEL TRUTH — AI Image Forgery Detector

> **Upload an image. Know the truth.**  
> A deep learning powered web app that detects whether an image is **AI-generated or real** — using a dual neural network ensemble (spatial + frequency analysis).

---

## ✨ What It Does

| Feature | Details |
|---|---|
| 🧠 Neural Detection | CLIP-based spatial model + lightweight frequency CNN |
| 🔁 Smart Ensemble | Dynamic weighting based on model confidence |
| 🛡️ Auth System | User + Admin registration/login with role-based access |
| 🌐 Web Interface | Clean, cyberpunk-themed UI with real-time detection |
| ⚡ Fallback Mode | Heuristic detector runs if neural models aren't loaded |

---

## 🗂️ Project Structure

```
pixel-truth/
│
├── app.py                  # Flask server & API routes
├── auth_routes.py          # User & admin auth (register/login)
├── config.py               # Centralized configuration
│
├── v5spatial.py            # Spatial model (CLIP ViT backbone)
├── v5frequency.py          # Frequency model (FFT-based CNN)
├── v5training.py           # Training pipeline
├── v5inference.py          # Inference / detection logic
│
├── templates/
│   ├── landing.html        # Landing page
│   ├── homepage.html       # Detection dashboard
│   ├── login.html
│   ├── register.html
│   └── admin_register.html
│
├── static/
│   ├── css/auth.css
│   └── js/auth.js
│
├── checkpoints/            # Place trained .pth files here
├── uploads/                # Temp upload folder (auto-created)
└── data/                   # User store (auto-created)
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/pixel-truth.git
cd pixel-truth
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **PyTorch note:** For GPU support, install PyTorch separately from [pytorch.org](https://pytorch.org/get-started/locally/) matching your CUDA version.

### 4. Run the app

```bash
python app.py
```

Open your browser at → **http://localhost:5000**

---

## 🧠 Using the Neural Models

The app works in two modes:

| Mode | When | Accuracy |
|---|---|---|
| 🟢 Neural Ensemble | When `.pth` checkpoints are present | High |
| 🟡 Heuristic Fallback | When no checkpoints found | Lower |

### Train your own models

```bash
# Train both models on CIFAKE dataset
python v5training.py --data_root CIFAKE --model both --epochs 50 --calibrate

# Place the output files in checkpoints/
# → checkpoints/spatial_model_best.pth
# → checkpoints/frequency_model_best.pth
```

---

## 🔐 Authentication

| Role | Access |
|---|---|
| `user` | Upload & analyze images |
| `admin` / `superadmin` | Elevated access, requires Admin Key |

The default admin key is `PIXELTRUTH_ADMIN_2025`.  
**Change it in production** by setting the environment variable:

```bash
export ADMIN_SECRET_KEY=your_secret_key_here
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `pixel-truth-secret-prod-v3` | Flask session secret |
| `ADMIN_SECRET_KEY` | `PIXELTRUTH_ADMIN_2025` | Admin registration key |
| `SPATIAL_TEMPERATURE` | `1.0` | Raise to 1.5–2.0 if real images are misclassified |
| `FREQ_TEMPERATURE` | `1.0` | Same for frequency model |
| `REAL_THRESHOLD` | `0.5` | Lower to ~0.45 to bias towards REAL predictions |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET` | `/home` | Detection dashboard |
| `POST` | `/api/detect` | **Upload image for detection** |
| `GET` | `/api/models` | Model info |
| `GET` | `/health` | Health check |
| `POST` | `/api/auth/register` | User registration |
| `POST` | `/api/auth/login` | User login |
| `POST` | `/api/auth/admin/register` | Admin registration |
| `POST` | `/api/auth/admin/login` | Admin login |

### Example — detect an image

```bash
curl -X POST http://localhost:5000/api/detect \
  -F "image=@your_photo.jpg"
```

---

## 🖼️ Supported Image Formats

`PNG` · `JPG / JPEG` · `WEBP` · `BMP` · `TIFF`  
**Max file size: 16 MB**

---

## 📦 Requirements

- Python 3.9+
- PyTorch 2.0+
- 4 GB+ VRAM recommended for GPU inference (CPU also works)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<p align="center">Made with ⚡ by the PIXEL TRUTH Team</p>