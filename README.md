# 🛰️ Satellite Image Classifier — Deployment Guide

REST API that classifies satellite images into **cloudy | fire | floods | normal**  
Built with FastAPI + TensorFlow. Runs fully locally — no cloud account needed.

---

## 📁 Project Structure

```
satellite_classifier/
│
├── Neural_Dataset/                      # Training dataset
│   ├── cloudy/
│   ├── fire/
│   ├── floods/
│   └── normal/
│
├── app/                                 # FastAPI app
│   ├── __pycache__/
│   │   └── main.cpython-313.pyc
│   ├── static/                          # Frontend files
│   │   ├── favicon.ico
│   │   └── index.html
│   └── main.py                          # API entry point
│
├── scripts/
│   ├── export_model.py                  # Model export / packaging
│   └── test_api.py                      # API testing script
│
├── Satellite_image_classification.ipynb # Training notebook
├── best_model.keras                     # Saved model (best)
├── final_model.keras                    # Final trained model
├── class_mapping.json                   # Label mapping
├── satellite_classifierrun.bat          # Run script (Windows)
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Step 1 — Train the model (run the notebook)

Open `Satellite_image_classification.ipynb` in Jupyter or VS Code.

The notebook expects your dataset in this structure, **in the same folder as the notebook**:

```
Neural_Dataset/
├── cloudy/
├── fire/
├── floods/
└── normal/
```

If your dataset folder has a different name or is in a different location,
update `dataset_path` in the first code cell:

```python
dataset_path = "/absolute/path/to/your/Neural_Dataset"
```

Run all cells. When training finishes, the notebook saves:
- `best_model.keras` — best weights (saved automatically by ModelCheckpoint)
- `final_model.keras` — final weights
- `class_mapping.json` — label index → class name

---

## Step 2 — Verify the model export

From the same folder as the notebook:

```bash
python scripts/export_model.py
# or if your file is named differently:
python scripts/export_model.py --model final_model.keras
```

---

## Step 3 — Copy model into the deployment folder

```
satellite_classifier/
├── best_model.keras      ← copy here
├── class_mapping.json    ← copy here
├── app/main.py
└── Dockerfile
```

---

## Step 4 — Run the API

### With Docker (recommended — no dependency conflicts)

```bash
cd satellite_classifier/
docker compose up --build
```

### Without Docker

```bash
cd satellite_classifier/
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Status check |
| GET | `/health` | Health + model loaded flag |
| GET | `/docs` | Swagger UI (interactive) |
| POST | `/predict` | Classify a single image |
| POST | `/predict/batch` | Classify up to 20 images |

### Single image

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/satellite.jpg"
```

```json
{
  "predicted_class": "fire",
  "confidence": 0.9731,
  "all_probabilities": {
    "cloudy": 0.0089,
    "fire": 0.9731,
    "floods": 0.0124,
    "normal": 0.0056
  }
}
```

### Batch

```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg"
```

---

## Step 5 — Smoke test

```bash
python scripts/test_api.py --url http://localhost:8000
```

---

## Optional — Deploy to the cloud

Once working locally, push to the cloud:

| Platform | Command |
|---|---|
| **Render** | Connect GitHub repo → New Web Service → Docker |
| **Railway** | `railway up` |
| **GCP Cloud Run** | `gcloud run deploy --source .` |

---

## Model info

| Property | Value |
|----------|-------|
| Input | 64 × 64 RGB image |
| Output | 4-class softmax |
| Classes | cloudy, fire, floods, normal |
| Format | `.keras` (TensorFlow 2.x) |
