"""
Enroll a single person into the face database (1 image per person).

Usage (run from attendance_system/ directory):
    python scripts/enroll_single.py --id NV001 --name "Nguyen Van A" --image path/to/photo.jpg
    python scripts/enroll_single.py --id NV001 --name "Nguyen Van A" --camera

Options:
    --id       Employee ID (e.g. NV001)
    --name     Full name
    --image    Path to a reference photo (no camera needed)
    --camera   Capture from live camera instead of loading a file
    --config   Path to config.yaml (default: ../config.yaml)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ExifTags

# Allow running from attendance_system/ or attendance_system/scripts/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from detection.face_detector import FaceDetector
from recognition.face_recognizer import FaceRecognizer
from recognition.face_db import FaceDB
from utils.image_enhancer import ImageEnhancer
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_image_exif(path: str) -> np.ndarray:
    """Load image applying EXIF rotation — phone photos are often stored rotated."""
    pil_img = Image.open(path).convert("RGB")
    try:
        exif = pil_img._getexif()
        if exif:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    if value == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif value == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif value == 8:
                        pil_img = pil_img.rotate(90, expand=True)
                    break
    except Exception:
        pass
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def capture_from_camera(cfg: dict) -> np.ndarray | None:
    """Warm up camera, capture a single frame, and apply enhancement."""
    from camera.camera_manager import CameraManager
    import time

    cam_cfg = cfg.get("camera", {})
    enh_cfg = cfg.get("enhance", {})

    enhancer: ImageEnhancer | None = None
    if enh_cfg.get("enabled", False):
        enhancer = ImageEnhancer(
            clahe_clip_limit=enh_cfg.get("clahe_clip_limit", 2.0),
            clahe_tile_grid=enh_cfg.get("clahe_tile_grid", 8),
            sharpen_strength=enh_cfg.get("sharpen_strength", 0.6),
            denoise=enh_cfg.get("denoise", False),
        )

    with CameraManager(
        width=cam_cfg.get("width", 320),
        height=cam_cfg.get("height", 240),
        cv2_device_index=cam_cfg.get("cv2_device_index", 0),
        isp_controls=cam_cfg.get("isp_controls"),
    ) as cam:
        print("Warming up camera (2s)...")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            cam.capture()
        print("Capturing...")
        frame = cam.capture()

    if frame is not None and enhancer is not None:
        frame = enhancer.enhance(frame)
    return frame


def verify_self_similarity(
    recognizer: FaceRecognizer,
    db: FaceDB,
    employee_id: str,
    face_crop: np.ndarray,
) -> None:
    """Sanity check: newly enrolled embedding should score ≥ 0.95 against itself."""
    embedding = recognizer.embed(face_crop)
    result = db.find_match(embedding)
    if result.name != db._db[employee_id]["name"]:
        logger.warning(
            "Self-similarity check: matched '%s' instead of expected '%s' — re-enroll?",
            result.name, db._db[employee_id]["name"],
        )
    else:
        logger.info(
            "Self-similarity check OK: %.4f (expected ≥ 0.95)", result.confidence
        )
        if result.confidence < 0.95:
            logger.warning("Score %.4f is below 0.95 — the crop quality may be poor", result.confidence)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll one person into the face DB")
    parser.add_argument("--id",     required=True,  help="Employee ID, e.g. NV001")
    parser.add_argument("--name",   required=True,  help="Full name")
    parser.add_argument("--image",  default=None,   help="Path to reference photo")
    parser.add_argument("--camera", action="store_true", help="Capture from live camera")
    parser.add_argument("--config", default=None,   help="Path to config.yaml")
    args = parser.parse_args()

    if not args.image and not args.camera:
        parser.error("Provide either --image <path> or --camera")

    config_path = Path(args.config) if args.config else _ROOT / "config.yaml"
    cfg = load_config(config_path)

    setup_logger(cfg.get("debug", {}).get("log_level", "INFO"))
    logger.info("Enrolling: %s (%s)", args.name, args.id)

    det_cfg = cfg.get("detection", {})
    rec_cfg = cfg.get("recognition", {})

    detector   = FaceDetector(
        model_path=str(_ROOT / det_cfg.get("model_path", "detection/models/face_detection.tflite")),
        confidence_threshold=det_cfg.get("confidence_threshold", 0.5),
        min_face_size_px=det_cfg.get("min_face_size_px", 40),
    )
    recognizer = FaceRecognizer(
        model_path=str(_ROOT / rec_cfg.get("model_path", "recognition/models/face_embedding.tflite")),
    )
    db = FaceDB(
        embeddings_path=str(_ROOT / rec_cfg.get("embeddings_path", "data/embeddings.pkl")),
        similarity_threshold=rec_cfg.get("similarity_threshold", 0.75),
    )

    # ---- Acquire source frame ----
    if args.camera:
        frame = capture_from_camera(cfg)
        if frame is None:
            logger.info("Enroll cancelled")
            sys.exit(0)
    else:
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error("Image not found: %s", img_path)
            sys.exit(1)
        frame = load_image_exif(str(img_path))
        if frame is None:
            logger.error("Cannot read image: %s", img_path)
            sys.exit(1)
        logger.info("Loaded image: %dx%d px", frame.shape[1], frame.shape[0])

    # ---- Detect face in frame ----
    detections = detector.detect(frame)
    if not detections:
        logger.error("No face detected in the image — try a clearer photo")
        sys.exit(1)

    det = max(detections, key=lambda d: d.width * d.height)
    logger.info("Face bbox: %dx%d px at (%d,%d)", det.width, det.height, det.x1, det.y1)

    if det.keypoints:
        re, le = det.keypoints[0], det.keypoints[1]
        eye_dist = ((le[0]-re[0])**2 + (le[1]-re[1])**2) ** 0.5
        logger.info("Eye keypoints: kp[0]=%s  kp[1]=%s  dist=%.1f  face_w=%d",
                    re, le, eye_dist, det.width)

    face_crop = det.align_bgr(frame)
    face_h, face_w = face_crop.shape[:2]
    logger.info("Face crop: %dx%d px", face_w, face_h)

    # Save crop preview to file — open it in any image viewer to verify
    preview_path = _ROOT / "data" / "enroll_preview.jpg"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), face_crop)

    # Draw all keypoints on original frame for debug — saves to enroll_preview_kps.jpg
    _KP_COLORS = [
        (0, 255, 0),    # 0 right_eye — green
        (255, 128, 0),  # 1 left_eye  — blue-orange
        (0, 0, 255),    # 2 nose      — red
        (0, 255, 255),  # 3 mouth     — yellow
        (255, 0, 255),  # 4 right_ear — magenta
        (255, 255, 0),  # 5 left_ear  — cyan
    ]
    _KP_LABELS = ["0:RE", "1:LE", "2:nose", "3:mouth", "4:rear", "5:lear"]
    kp_frame = frame.copy()
    # Draw bbox
    cv2.rectangle(kp_frame, (det.x1, det.y1), (det.x2, det.y2), (200, 200, 200), 1)
    for idx, kp in enumerate(det.keypoints):
        color = _KP_COLORS[idx % len(_KP_COLORS)]
        cv2.circle(kp_frame, kp, 5, color, -1)
        cv2.putText(kp_frame, _KP_LABELS[idx], (kp[0] + 6, kp[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    kp_preview_path = _ROOT / "data" / "enroll_preview_kps.jpg"
    cv2.imwrite(str(kp_preview_path), kp_frame)
    print(f"Keypoint debug image saved to: {kp_preview_path}")
    print(f"\nFace crop saved to: {preview_path}")
    print("Open that file to verify the crop quality.")
    print("Press ENTER to confirm and save, or type 'q' + ENTER to cancel: ", end="", flush=True)

    answer = input().strip().lower()
    if answer == "q":
        logger.info("Enroll cancelled by user")
        sys.exit(0)

    # ---- Compute embedding and save ----
    embedding = recognizer.embed(face_crop)
    db.add_person(args.id, args.name, embedding)

    # ---- Self-similarity sanity check ----
    verify_self_similarity(recognizer, db, args.id, face_crop)

    print(f"\nEnrolled successfully: {args.name} ({args.id})")
    print(f"Total in DB: {len(db.list_people())} person(s)")
    for eid, name in db.list_people():
        print(f"  {eid}: {name}")


if __name__ == "__main__":
    main()
