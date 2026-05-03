"""
Satellite Image Classification API
Classes: cloudy, fire, floods, normal
"""

import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import tensorflow as tf
import logging

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Init ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Satellite Image Classifier",
    description="Classifies satellite images into: Cloudy, Fire, Floods, Normal",
    version="1.0.0",
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Enable CORS (optional but useful)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE = 64
CLASSES = ["cloudy", "fire", "floods", "normal"]
MODEL_PATH = "best_model.keras"

# ── Load model once ────────────────────────────────────────────────────────────
model: tf.keras.Model | None = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise RuntimeError(f"Cannot start API — model file not found at '{MODEL_PATH}'") from e


# ── Helper ─────────────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    """Open, resize, normalize image → (1, 64, 64, 3)"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Routes ─────────────────────────────────────────────────────────────────────

# 🔥 MAIN PAGE (Frontend)
@app.get("/")
def home():
    return FileResponse("app/static/index.html")


# Health check
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": model is not None}


# Single prediction
@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    try:
        image_bytes = await file.read()
        input_tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    probs = model.predict(input_tensor, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))

    return {
        "predicted_class": CLASSES[predicted_idx],
        "confidence": round(float(probs[predicted_idx]), 4),
        "all_probabilities": {
            cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)
        },
    }


# Batch prediction
@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(files: list[UploadFile] = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Max 20 images allowed")

    results = []

    for f in files:
        try:
            image_bytes = await f.read()
            tensor = preprocess(image_bytes)
            probs = model.predict(tensor, verbose=0)[0]
            idx = int(np.argmax(probs))

            results.append({
                "filename": f.filename,
                "predicted_class": CLASSES[idx],
                "confidence": round(float(probs[idx]), 4),
            })

        except Exception as e:
            results.append({
                "filename": f.filename,
                "error": str(e)
            })

    return {"results": results, "total": len(results)}