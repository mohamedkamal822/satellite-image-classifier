# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps (Pillow needs libjpeg; TF needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App ────────────────────────────────────────────────────────────────────────
COPY app/ ./app/
# Copy your trained model into the image
COPY best_model.keras .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
