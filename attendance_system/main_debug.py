"""
Phase 1 debug entry point.

Serves a live MJPEG stream viewable in any browser:
    http://<PI_IP>:5000

Run from attendance_system/ directory:
    python main_debug.py
"""
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, Response

from camera.camera_manager import CameraManager
from detection.face_detector import Detection, FaceDetector
from recognition.face_db import FaceDB, MatchResult
from recognition.face_recognizer import FaceRecognizer
from utils.logger import setup_logger

logger = logging.getLogger(__name__)

# BGR colors
_COLOR_STABLE   = (0, 255, 255)   # yellow
_COLOR_MATCH    = (0, 255, 0)     # green
_COLOR_UNKNOWN  = (0, 0, 255)     # red
_COLOR_LOW_CONF = (0, 165, 255)   # orange

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS  = 1

# Shared state between pipeline thread and Flask thread
_frame_lock   = threading.Lock()
_latest_frame: np.ndarray | None = None


# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------

def draw_bbox(frame: np.ndarray, det: Detection, color: tuple, label: str) -> None:
    x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
    arm = max(8, min((x2 - x1) // 4, (y2 - y1) // 4))
    for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px, py), (px + dx * arm, py), color, 2)
        cv2.line(frame, (px, py), (px, py + dy * arm), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _THICKNESS)
        lx, ly = max(0, x1), max(th + 4, y1 - 4)
        cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly), color, -1)
        cv2.putText(frame, label, (lx + 2, ly - 2), _FONT, _FONT_SCALE, (0, 0, 0), _THICKNESS)


def draw_status_bar(frame: np.ndarray, text: str, color: tuple) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 22), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, text, (4, h - 6), _FONT, 0.45, color, 1)


# ------------------------------------------------------------------
# Pipeline thread — runs detection + recognition, writes to _latest_frame
# ------------------------------------------------------------------

def _pipeline_wrapper(*args, **kwargs):
    try:
        pipeline_loop(*args, **kwargs)
    except Exception:
        logger.exception("Pipeline thread crashed")


def pipeline_loop(
    cam: CameraManager,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    db: FaceDB,
    cfg: dict,
    stop_event: threading.Event,
) -> None:
    global _latest_frame

    det_cfg = cfg.get("detection", {})
    rec_cfg = cfg.get("recognition", {})

    stable_required   = det_cfg.get("stable_frames_required", 3)
    low_conf_thresh   = rec_cfg.get("low_confidence_threshold", 0.60)
    log_latency       = cfg.get("debug", {}).get("log_latency", True)

    stable_count       = 0
    last_result: MatchResult | None = None
    result_frames_left = 0
    RESULT_HOLD_FRAMES = 30

    if db.is_empty():
        logger.warning("Face DB is empty — enroll someone first")

    logger.info("Warming up camera (2s)...")
    warmup_start = time.perf_counter()
    while time.perf_counter() - warmup_start < 2.0:
        cam.capture()

    logger.info("Pipeline running. Open http://<PI_IP>:5000 in browser.")

    while not stop_event.is_set():
        t0 = time.perf_counter()

        frame = cam.capture()
        if frame is None:
            continue

        t_det = time.perf_counter()
        detections = detector.detect(frame)
        det_ms = (time.perf_counter() - t_det) * 1000

        annotated = frame.copy()

        if not detections:
            stable_count = 0
            result_frames_left = max(0, result_frames_left - 1)
            if result_frames_left == 0:
                last_result = None
            draw_status_bar(annotated, "Scanning...", (180, 180, 180))
        else:
            det = detections[0]
            stable_count += 1
            recog_ms = 0.0

            if result_frames_left > 0:
                result_frames_left -= 1
                result = last_result
            elif stable_count >= stable_required:
                face_crop = det.align_bgr(frame)
                if log_latency and det.keypoints:
                    re, le = det.keypoints[0], det.keypoints[1]
                    import math
                    logger.debug(
                        "kp right=%s left=%s eye_dist=%.1f face_w=%d",
                        re, le,
                        math.hypot(le[0]-re[0], le[1]-re[1]),
                        det.width,
                    )
                t_rec = time.perf_counter()
                embedding  = recognizer.embed(face_crop)
                result     = db.find_match(embedding)
                recog_ms   = (time.perf_counter() - t_rec) * 1000

                stable_count       = 0
                last_result        = result
                result_frames_left = RESULT_HOLD_FRAMES

                logger.info(
                    "Recognition: %s  conf=%.3f  matched=%s  det=%.0fms recog=%.0fms",
                    result.name, result.confidence, result.matched, det_ms, recog_ms,
                )
            else:
                result = None

            if result is None:
                draw_bbox(annotated, det, _COLOR_STABLE, f"Stable {stable_count}/{stable_required}")
            elif result.matched:
                draw_bbox(annotated, det, _COLOR_MATCH, f"{result.name} {result.confidence:.0%}")
            else:
                color = _COLOR_LOW_CONF if result.confidence >= low_conf_thresh else _COLOR_UNKNOWN
                draw_bbox(annotated, det, color, f"Unknown ({result.confidence:.0%})")

            if result and result.matched:
                draw_status_bar(annotated, f"MATCH: {result.name}", _COLOR_MATCH)
            elif result:
                draw_status_bar(annotated, "No match", _COLOR_UNKNOWN)
            else:
                draw_status_bar(annotated,
                    f"Face detected — stable {stable_count}/{stable_required}", _COLOR_STABLE)

        if log_latency:
            logger.debug("frame=%.0fms  det=%.0fms", (time.perf_counter() - t0) * 1000, det_ms)

        with _frame_lock:
            _latest_frame = annotated


# ------------------------------------------------------------------
# Flask MJPEG stream
# ------------------------------------------------------------------

app = Flask(__name__)


def _generate_mjpeg():
    while True:
        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        time.sleep(0.05)   # ~20fps cap


@app.route("/")
def index():
    return (
        "<html><body style='background:#111;margin:0'>"
        "<img src='/stream' style='width:100%;max-width:960px;display:block;margin:auto'>"
        "</body></html>"
    )


@app.route("/stream")
def stream():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    root = Path(__file__).resolve().parent
    cfg  = load_config(root / "config.yaml")
    setup_logger(cfg.get("debug", {}).get("log_level", "INFO"))

    logger.info("Checking models...")
    from downloader import ensure_models
    ensure_models(str(root))

    det_cfg = cfg.get("detection", {})
    rec_cfg = cfg.get("recognition", {})
    cam_cfg = cfg.get("camera", {})

    logger.info("Loading face detector...")
    detector = FaceDetector(
        model_path=str(root / det_cfg["model_path"]),
        confidence_threshold=det_cfg.get("confidence_threshold", 0.6),
        nms_iou_threshold=det_cfg.get("nms_iou_threshold", 0.3),
        min_face_size_px=det_cfg.get("min_face_size_px", 60),
    )

    logger.info("Loading face recognizer...")
    recognizer = FaceRecognizer(model_path=str(root / rec_cfg["model_path"]))

    logger.info("Loading face DB...")
    db = FaceDB(
        embeddings_path=str(root / rec_cfg["embeddings_path"]),
        similarity_threshold=rec_cfg.get("similarity_threshold", 0.75),
        low_confidence_threshold=rec_cfg.get("low_confidence_threshold", 0.60),
    )

    logger.info("Starting camera...")
    stop_event = threading.Event()

    with CameraManager(
        width=cam_cfg.get("width", 320),
        height=cam_cfg.get("height", 240),
        cv2_device_index=cam_cfg.get("cv2_device_index", 0),
    ) as cam:
        pipeline_thread = threading.Thread(
            target=_pipeline_wrapper,
            args=(cam, detector, recognizer, db, cfg, stop_event),
            daemon=True,
        )
        pipeline_thread.start()

        try:
            # Flask runs in main thread; pipeline runs in background
            app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            stop_event.set()

    logger.info("Done.")


if __name__ == "__main__":
    main()
