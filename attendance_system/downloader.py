"""
Download required TFLite models into their target directories.
Run once before starting the system:
    python downloader.py
"""
import hashlib
import logging
import sys
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
#
# face_detection.tflite  — BlazeFace short-range (MediaPipe / Google)
#   Input : 128×128 RGB float32, normalized to [-1, 1]
#   Output: [1,896,16] regressors + [1,896,1] scores
#
# face_embedding.tflite  — MediaPipe MobileNetV3 face embedder
#   Input : 112×112 RGB float32, normalized to [-1, 1]
#   Output: [1, 128] L2-normalized embedding vector
# ---------------------------------------------------------------------------
MODELS: list[dict] = [
    {
        "filename": "face_detection.tflite",
        "dest": "detection/models/face_detection.tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite",
        "sha256": "",
        "description": "BlazeFace short-range face detector",
    },
    # {
    #     "filename": "face_embedding.tflite",
    #     "dest": "recognition/models/face_embedding.tflite",
    #     # URL đã bị Google xóa — cần tìm link thay thế
    #     "url": "https://storage.googleapis.com/mediapipe-models/face_embedder/mobilenet_v3_small/float32/latest/face_embedder.tflite",
    #     "sha256": "",
    #     "description": "MediaPipe MobileNetV3 face embedder (128-dim)",
    # },
]


def ensure_models(base_dir: str = ".") -> None:
    """Download any missing models. Call this at app startup."""
    root = Path(base_dir)
    for spec in MODELS:
        dest = root / spec["dest"]
        if dest.exists():
            logger.info("Model already present: %s", dest)
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s …", spec["description"])
        _download(spec["url"], dest)

        expected = spec.get("sha256", "").strip()
        if expected and _sha256(dest) != expected:
            dest.unlink()
            raise ValueError(f"SHA256 mismatch after downloading {dest.name} — deleted")

        logger.info("Saved: %s (%.2f MB)", dest, dest.stat().st_size / 1_048_576)


def _download(url: str, dest: Path, chunk_size: int = 8192) -> None:
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  {dest.name}: {pct}%", end="", flush=True)
        print()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    # Run from attendance_system/ directory
    base = Path(__file__).parent
    try:
        ensure_models(str(base))
        print("\nAll models ready.")
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
