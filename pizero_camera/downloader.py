import hashlib
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# face_detection.tflite  — BlazeFace short-range from Google MediaPipe assets
# face_embedding.tflite  — MobileFaceNet (find a pre-converted .tflite on GitHub,
#   e.g. search "mobilefacenet tflite" or convert from:
#   https://github.com/sirius-ai/MobileFaceNet_TF)
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "face_detection.tflite": {
        "url": "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite",
        "sha256": "",
        "description": "BlazeFace short-range face detector (MobileNet-based)",
    },
    "face_embedding.tflite": {
        "url": "https://storage.googleapis.com/mediapipe-models/face_embedder/mobilenet_v3_small/float32/latest/face_embedder.tflite",
        "sha256": "",
        "description": "MediaPipe MobileNetV3 face embedder",
        "optional": True,   # system still runs without it — all faces shown as Unknown
    },
}


def ensure_models(models_dir: str = "models") -> None:
    """Download missing models into models_dir. Skip if already present and valid."""
    dest = Path(models_dir)
    dest.mkdir(parents=True, exist_ok=True)

    for filename, meta in MODELS.items():
        path = dest / filename
        url = meta.get("url", "").strip()

        if not url:
            if not path.exists():
                optional = meta.get("optional", False)
                if optional:
                    logger.warning("Optional model '%s' not found — skipping", filename)
                else:
                    raise FileNotFoundError(
                        f"Model '{filename}' not found and no URL configured.\n"
                        f"  → Add the download URL in downloader.py MODELS['{filename}']['url']"
                    )
            logger.debug("No URL for %s — skipping download check", filename)
            continue

        if path.exists():
            expected_sha = meta.get("sha256", "").strip()
            if not expected_sha or _sha256(path) == expected_sha:
                logger.info("Model already present: %s", filename)
                continue
            logger.warning("SHA256 mismatch for %s — re-downloading", filename)
            path.unlink()

        logger.info("Downloading %s …", meta["description"])
        _download(url, path)

        expected_sha = meta.get("sha256", "").strip()
        if expected_sha and _sha256(path) != expected_sha:
            path.unlink()
            raise ValueError(f"SHA256 mismatch after downloading {filename} — file deleted")

        logger.info("Saved %s (%.1f MB)", filename, path.stat().st_size / 1_048_576)


def _download(url: str, dest: Path, chunk_size: int = 8192) -> None:
    with requests.get(url, stream=True, timeout=30) as res:
        res.raise_for_status()
        total = int(res.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in res.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {downloaded * 100 // total}%", end="", flush=True)
        print()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
