"""
Crop faces from images in known_faces/ using the BlazeFace detector.
Run once on PC or Pi Zero before starting main.py.

Usage:
    python tools/crop_faces.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from detector import FaceDetector

KNOWN_FACES_DIR  = "data/known_faces"
DETECTOR_MODEL   = "models/face_detection.tflite"
PADDING          = 0.2   # extra margin around detected face (20%)
MIN_FACE_PX      = 40    # skip detections smaller than this


def crop_with_padding(image: Image.Image, det, padding: float) -> Image.Image:
    w, h = image.size
    fw = det.x2 - det.x1
    fh = det.y2 - det.y1
    pad_x = int(fw * padding)
    pad_y = int(fh * padding)
    x1 = max(0, det.x1 - pad_x)
    y1 = max(0, det.y1 - pad_y)
    x2 = min(w, det.x2 + pad_x)
    y2 = min(h, det.y2 + pad_y)
    return image.crop((x1, y1, x2, y2))


def process_directory(detector: FaceDetector, faces_dir: Path) -> None:
    image_exts = {".jpg", ".jpeg", ".png"}

    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        print(f"\n[{person_dir.name}]")

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in image_exts:
                continue
            if "_face" in img_path.stem:
                print(f"  skip (already cropped): {img_path.name}")
                continue

            image = Image.open(img_path).convert("RGB")
            detections = detector.detect(image)

            if not detections:
                print(f"  NO FACE: {img_path.name} — keep original")
                continue

            # use the largest detected face
            det = max(detections, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
            face_w = det.x2 - det.x1
            face_h = det.y2 - det.y1

            if face_w < MIN_FACE_PX or face_h < MIN_FACE_PX:
                print(f"  TOO SMALL ({face_w}x{face_h}px): {img_path.name}")
                continue

            cropped = crop_with_padding(image, det, PADDING)

            out_path = img_path.with_stem(img_path.stem + "_face")
            cropped.save(out_path)
            print(f"  OK: {img_path.name} → {out_path.name}  ({face_w}x{face_h}px face)")


def main() -> None:
    faces_dir = Path(KNOWN_FACES_DIR)
    if not faces_dir.exists():
        print(f"Directory not found: {faces_dir}")
        return

    print(f"Loading detector: {DETECTOR_MODEL}")
    detector = FaceDetector(model_path=DETECTOR_MODEL, confidence_threshold=0.4)

    process_directory(detector, faces_dir)

    print("\nDone. Rename or delete original files if needed.")
    print("Then delete data/embeddings.cache and restart main.py.")


if __name__ == "__main__":
    main()
