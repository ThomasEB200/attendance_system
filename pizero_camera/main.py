import logging
import time

from camera import Camera
from client import ServerClient
from detector import FaceDetector
from downloader import ensure_models
from recognizer import FaceRecognizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
SERVER_URL = "http://192.168.1.x:5000"   # change to your PC's IP
                                          # set to "" to run offline (no web)
MODELS_DIR         = "models"
EMBEDDING_MODEL    = "models/face_embedding.tflite"
KNOWN_FACES_DIR    = "data/known_faces"
EMBEDDING_CACHE    = "data/embeddings.cache"

CAMERA_WIDTH       = 640
CAMERA_HEIGHT      = 480
DETECTION_CONFIDENCE  = 0.5
RECOGNITION_THRESHOLD = 0.45
LOOP_INTERVAL_SEC     = 1.0
# ------------------------------------------------------------------


def process_frame(
    camera: Camera,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    client: ServerClient | None,
) -> None:
    frame = camera.capture()
    detections = detector.detect(frame)

    if not detections:
        if client:
            client.post_result([])
        return

    faces = []
    for det in detections:
        face_crop = det.crop(frame)
        result = recognizer.identify(face_crop)
        faces.append(result)
        logger.info("Face: %s (%.0f%%)", result["name"], result["confidence"] * 100)

    if client:
        client.post_result(faces)


def main() -> None:
    logger.info("Checking models…")
    ensure_models(MODELS_DIR)

    logger.info("Starting camera…")
    camera = Camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

    logger.info("Loading face detector…")
    detector = FaceDetector(confidence_threshold=DETECTION_CONFIDENCE)

    logger.info("Loading recognizer + embeddings…")
    recognizer = FaceRecognizer(
        EMBEDDING_MODEL,
        KNOWN_FACES_DIR,
        similarity_threshold=RECOGNITION_THRESHOLD,
        cache_path=EMBEDDING_CACHE,
    )

    client = ServerClient(SERVER_URL) if SERVER_URL else None
    if client:
        logger.info("Results will be sent to %s", SERVER_URL)
    else:
        logger.info("Offline mode — results printed locally only")

    logger.info("Ready.")
    try:
        while True:
            try:
                process_frame(camera, detector, recognizer, client)
            except Exception as e:
                logger.error("Error in main loop: %s", e, exc_info=True)
            time.sleep(LOOP_INTERVAL_SEC)
    finally:
        camera.close()
        detector.close()


if __name__ == "__main__":
    main()
