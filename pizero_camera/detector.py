import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_INPUT_SIZE = 128   # BlazeFace short-range input resolution
_NUM_ANCHORS = 896  # total anchors: 16×16×2 + 8×8×6 = 512 + 384


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop((self.x1, self.y1, self.x2, self.y2))


class FaceDetector:
    """
    BlazeFace TFLite face detector (MobileNet-based).
    Model file: models/face_detection.tflite (downloaded by downloader.py)

    BlazeFace outputs raw regressors + classificators that need post-processing:
      1. decode boxes from anchor offsets
      2. sigmoid on scores
      3. non-maximum suppression (NMS)
    """

    def __init__(
        self,
        model_path: str = "models/face_detection.tflite",
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
    ):
        self._threshold = confidence_threshold
        self._nms_iou = nms_iou_threshold

        self._interpreter = tflite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_idx = self._interpreter.get_input_details()[0]["index"]
        self._output_details = self._interpreter.get_output_details()

        self._anchors = self._build_anchors()

    def detect(self, image: Image.Image) -> list[Detection]:
        w, h = image.size
        input_tensor = self._preprocess(image)
        self._interpreter.set_tensor(self._input_idx, input_tensor)
        self._interpreter.invoke()

        # BlazeFace outputs: [0] regressors [1,896,16], [1] classificators [1,896,1]
        raw_boxes  = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        raw_scores = self._interpreter.get_tensor(self._output_details[1]["index"])[0]

        clipped = np.clip(raw_scores[:, 0], -88, 88)       # prevent overflow in exp
        scores = 1.0 / (1.0 + np.exp(-clipped))            # sigmoid
        boxes  = self._decode_boxes(raw_boxes)              # normalized [y1,x1,y2,x2]

        keep = self._nms(boxes, scores)

        detections = []
        for i in keep:
            y1, x1, y2, x2 = boxes[i]
            detections.append(Detection(
                x1=max(0, int(x1 * w)),
                y1=max(0, int(y1 * h)),
                x2=min(w, int(x2 * w)),
                y2=min(h, int(y2 * h)),
                score=float(scores[i]),
            ))

        logger.debug("Detected %d face(s)", len(detections))
        return detections

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        arr = np.array(image.resize((_INPUT_SIZE, _INPUT_SIZE)), dtype=np.float32)
        arr = (arr - 127.5) / 127.5  # normalize to [-1, 1]
        return np.expand_dims(arr, axis=0)

    def _build_anchors(self) -> np.ndarray:
        """
        Generate BlazeFace short-range anchors.
        Config: strides [8,16], anchors/cell [2,6], input 128×128.
        """
        anchors = []
        for stride, num_a in zip([8, 16], [2, 6]):
            grid = _INPUT_SIZE // stride
            for y in range(grid):
                for x in range(grid):
                    cx = (x + 0.5) / grid
                    cy = (y + 0.5) / grid
                    for _ in range(num_a):
                        anchors.append([cy, cx])  # [y, x] to match box format
        return np.array(anchors, dtype=np.float32)  # [896, 2]

    def _decode_boxes(self, raw_boxes: np.ndarray) -> np.ndarray:
        """Decode raw regressors into normalized [y1, x1, y2, x2] boxes."""
        # raw_boxes[:, 0:4] = [dy, dx, dh, dw] offsets relative to anchor
        cy = raw_boxes[:, 0] / _INPUT_SIZE + self._anchors[:, 0]
        cx = raw_boxes[:, 1] / _INPUT_SIZE + self._anchors[:, 1]
        h  = raw_boxes[:, 2] / _INPUT_SIZE
        w  = raw_boxes[:, 3] / _INPUT_SIZE
        return np.stack([cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2], axis=1)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        """Non-maximum suppression — returns indices of kept boxes."""
        candidates = np.where(scores >= self._threshold)[0].tolist()
        candidates.sort(key=lambda i: scores[i], reverse=True)

        kept = []
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [
                i for i in candidates
                if self._iou(boxes[best], boxes[i]) < self._nms_iou
            ]
        return kept

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter_y1 = max(a[0], b[0])
        inter_x1 = max(a[1], b[1])
        inter_y2 = min(a[2], b[2])
        inter_x2 = min(a[3], b[3])
        inter = max(0.0, inter_y2 - inter_y1) * max(0.0, inter_x2 - inter_x1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
