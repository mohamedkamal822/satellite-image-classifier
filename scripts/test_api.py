"""
Test the running API with a synthetic image.
Usage: python scripts/test_api.py [--url http://localhost:8000]
"""

import argparse
import io
import json
import sys
import urllib.request
from PIL import Image
import numpy as np

def make_dummy_image() -> bytes:
    """Create a small random RGB image and return it as JPEG bytes."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_health(base_url: str):
    url = f"{base_url}/health"
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read())
    assert data["status"] == "ok", f"Health check failed: {data}"
    assert data["model_loaded"] is True, "Model not loaded!"
    print(f"✅ /health  →  {data}")

def test_predict(base_url: str):
    import urllib.request, urllib.parse
    image_bytes = make_dummy_image()

    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{base_url}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())

    assert "predicted_class" in data
    assert data["predicted_class"] in ["cloudy", "fire", "floods", "normal"]
    print(f"✅ /predict →  class={data['predicted_class']}, confidence={data['confidence']}")
    print(f"   all_probabilities: {data['all_probabilities']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    print(f"\nTesting API at {args.url}\n")
    try:
        test_health(args.url)
        test_predict(args.url)
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
