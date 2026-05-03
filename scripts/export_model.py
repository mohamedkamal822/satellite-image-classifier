"""
Run this script ONCE after training completes to verify and package
the model file for deployment.

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --model path/to/best_model.keras
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

CLASSES = ["cloudy", "fire", "floods", "normal"]

def main(model_path: str):
    # 1. Check file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Make sure training completed and the notebook's 'model.save()' cell ran.\n"
            "Expected file: best_model.keras or final_model.keras"
        )
    print(f"✅ Model file found: {model_path}")

    # 2. Load and sanity-check
    model = tf.keras.models.load_model(model_path)
    dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (1, 4), f"Unexpected output shape: {out.shape}"
    print("✅ Model loads and outputs shape (1, 4) — OK")

    # 3. Save class label mapping
    encoder = LabelEncoder()
    encoder.fit(CLASSES)
    mapping = {int(i): cls for i, cls in enumerate(encoder.classes_)}
    out_path = os.path.join(os.path.dirname(model_path) or ".", "class_mapping.json")
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ class_mapping.json saved: {mapping}")

    # 4. Print next steps
    print("\n── Export complete ──────────────────────────────────────────────────")
    print("Copy these two files into the satellite_classifier/ deployment folder:")
    print(f"  {os.path.abspath(model_path)}")
    print(f"  {os.path.abspath(out_path)}")
    print("Then run:  docker compose up --build")
    print("────────────────────────────────────────────────────────────────────")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_model.keras",
                        help="Path to the trained .keras model file")
    args = parser.parse_args()
    main(args.model)
