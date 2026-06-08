import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_INPUT_SIZE = 128   # BlazeFace short-range input resolution
_NUM_ANCHORS = 896  # 16×16×2 (stride 8) + 8×8×6 (stride 16) = 512 + 384


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    # 6 keypoints (x, y) pixel coords: [right_eye, left_eye, nose, mouth, right_ear, left_ear]
    keypoints: list = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def crop_bgr(self, frame: np.ndarray, padding: float = 0.0) -> np.ndarray:
        """Crop detected face region with optional fractional padding on each side."""
        h, w = frame.shape[:2]
        if padding > 0:
            pw = int(self.width * padding)
            ph = int(self.height * padding)
            x1 = max(0, self.x1 - pw)
            y1 = max(0, self.y1 - ph)
            x2 = min(w, self.x2 + pw)
            y2 = min(h, self.y2 + ph)
        else:
            x1, y1 = max(0, self.x1), max(0, self.y1)
            x2, y2 = min(w, self.x2), min(h, self.y2)
        return frame[y1:y2, x1:x2]

    def align_bgr(self, frame: np.ndarray, out_size: int = 112) -> np.ndarray:
        """
        Affine-align face to out_size×out_size using eye keypoints.
        Similarity transform (rotation + scale + translation) maps eye positions
        to fixed target coordinates, producing pose-normalized crops for MobileFaceNet.
        Falls back to padded crop when keypoints are unavailable or unreliable.
        """
        if len(self.keypoints) < 2:
            crop = self.crop_bgr(frame, padding=0.3)
            return cv2.resize(crop, (out_size, out_size))

        re = np.float32(self.keypoints[0])  # right_eye (x, y)
        le = np.float32(self.keypoints[1])  # left_eye  (x, y)

        # Ensure re is the left-of-image eye (smaller x) for consistent orientation
        if re[0] > le[0]:
            re, le = le, re

        # Validate: eye distance must be 15–100% of face bbox width
        eye_dist = math.hypot(float(le[0] - re[0]), float(le[1] - re[1]))
        if not (self.width * 0.15 <= eye_dist <= self.width * 1.0):
            logger.debug(
                "Unreliable eye keypoints (eye_dist=%.1f, face_w=%d) — using padded crop",
                eye_dist, self.width,
            )
            crop = self.crop_bgr(frame, padding=0.3)
            return cv2.resize(crop, (out_size, out_size))

        # Target eye positions in out_size × out_size output
        dst_re = np.float32([out_size * 0.35, out_size * 0.40])
        dst_le = np.float32([out_size * 0.65, out_size * 0.40])

        # Solve similarity transform from 2 point correspondences
        # (complex division: w = (dst_le - dst_re) / (le - re))
        dx, dy = float(le[0] - re[0]), float(le[1] - re[1])
        ex, ey = float(dst_le[0] - dst_re[0]), float(dst_le[1] - dst_re[1])
        norm_sq = dx * dx + dy * dy + 1e-8

        cos_s = (dx * ex + dy * ey) / norm_sq
        sin_s = (dx * ey - dy * ex) / norm_sq

        tx = dst_re[0] - cos_s * re[0] + sin_s * re[1]
        ty = dst_re[1] - sin_s * re[0] - cos_s * re[1]

        M = np.float32([[cos_s, -sin_s, tx],
                        [sin_s,  cos_s, ty]])

        return cv2.warpAffine(frame, M, (out_size, out_size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)


class FaceDetector:
    """
    BlazeFace short-range TFLite face detector.

    Input:  BGR frame (any resolution)
    Output: list[Detection] — pixel coordinates in original frame space

    Post-processing:
      1. decode anchor offsets → normalized [y1,x1,y2,x2] boxes
      2. sigmoid on classification scores
      3. NMS to remove duplicate detections
    """

    def __init__(
        self,
        model_path: str = "detection/models/face_detection.tflite",
        confidence_threshold: float = 0.6,
        nms_iou_threshold: float = 0.3,
        min_face_size_px: int = 60,
    ):
        self._confidence_threshold = confidence_threshold
        self._nms_iou = nms_iou_threshold
        self._min_face_size_px = min_face_size_px

        self._interpreter = tflite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()

        input_detail = self._interpreter.get_input_details()[0]
        self._input_idx = input_detail["index"]
        self._output_details = self._interpreter.get_output_details()

        self._anchors = self._build_anchors()
        logger.info("FaceDetector loaded: %s", model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect faces in a BGR frame.
        Returns detections sorted by score (descending), filtered by
        confidence_threshold and min_face_size_px.
        """
        h, w = frame.shape[:2]

        input_tensor = self._preprocess(frame)
        self._interpreter.set_tensor(self._input_idx, input_tensor)
        self._interpreter.invoke()

        # BlazeFace output: [0] regressors [1,896,16], [1] scores [1,896,1]
        raw_boxes  = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        raw_scores = self._interpreter.get_tensor(self._output_details[1]["index"])[0]

        clipped = np.clip(raw_scores[:, 0], -88.0, 88.0)
        scores = 1.0 / (1.0 + np.exp(-clipped))            # sigmoid

        boxes    = self._decode_boxes(raw_boxes)             # normalized [y1,x1,y2,x2]
        kps_all  = self._decode_keypoints(raw_boxes)         # [896, 6, 2] (y, x) normalized
        kept_indices = self._nms(boxes, scores)

        detections = []
        for i in kept_indices:
            y1n, x1n, y2n, x2n = boxes[i]
            x1 = max(0, int(x1n * w))
            y1 = max(0, int(y1n * h))
            x2 = min(w, int(x2n * w))
            y2 = min(h, int(y2n * h))

            if (x2 - x1) < self._min_face_size_px or (y2 - y1) < self._min_face_size_px:
                continue

            kps = []
            for k in range(6):
                ky, kx = kps_all[i, k]
                kps.append((min(w - 1, max(0, int(kx * w))),
                            min(h - 1, max(0, int(ky * h)))))

            detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2,
                                        score=float(scores[i]), keypoints=kps))

        detections.sort(key=lambda d: d.score, reverse=True)
        logger.debug("Detected %d face(s)", len(detections))
        return detections

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE))
        arr = resized.astype(np.float32)
        arr = (arr - 127.5) / 127.5     # normalize to [-1, 1]
        return np.expand_dims(arr, axis=0)

    def _build_anchors(self) -> np.ndarray:
        """
        BlazeFace short-range anchor grid.
        strides=[8,16], anchors_per_cell=[2,6], input=128×128
        """
        anchors = []
        for stride, num_a in zip([8, 16], [2, 6]):
            grid = _INPUT_SIZE // stride
            for y in range(grid):
                for x in range(grid):
                    cx = (x + 0.5) / grid
                    cy = (y + 0.5) / grid
                    for _ in range(num_a):
                        anchors.append([cy, cx])   # [y, x] order
        return np.array(anchors, dtype=np.float32)  # [896, 2]

    def _decode_boxes(self, raw_boxes: np.ndarray) -> np.ndarray:
        """Decode anchor-relative offsets into normalized [y1,x1,y2,x2]."""
        cy = raw_boxes[:, 0] / _INPUT_SIZE + self._anchors[:, 0]
        cx = raw_boxes[:, 1] / _INPUT_SIZE + self._anchors[:, 1]
        h  = raw_boxes[:, 2] / _INPUT_SIZE
        w  = raw_boxes[:, 3] / _INPUT_SIZE
        return np.stack([cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2], axis=1)

    def _decode_keypoints(self, raw_boxes: np.ndarray) -> np.ndarray:
        """
        Decode 6 facial keypoints from raw regressors.
        BlazeFace boxes use [dy, dx, dh, dw] order, but keypoints use [kx, ky] (x first).
        Returns [896, 6, 2] in (y, x) normalized coords.
        """
        kps = []
        for k in range(6):
            kx = raw_boxes[:, 4 + k * 2] / _INPUT_SIZE + self._anchors[:, 1]  # x uses cx anchor
            ky = raw_boxes[:, 5 + k * 2] / _INPUT_SIZE + self._anchors[:, 0]  # y uses cy anchor
            kps.append(np.stack([ky, kx], axis=1))
        return np.stack(kps, axis=1)  # [896, 6, 2]

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        """Non-maximum suppression — returns kept anchor indices."""
        candidates = np.where(scores >= self._confidence_threshold)[0].tolist()
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
        iy1 = max(a[0], b[0])
        ix1 = max(a[1], b[1])
        iy2 = min(a[2], b[2])
        ix2 = min(a[3], b[3])
        inter = max(0.0, iy2 - iy1) * max(0.0, ix2 - ix1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
