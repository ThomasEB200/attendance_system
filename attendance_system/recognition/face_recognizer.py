import logging

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

_EMBED_INPUT_SIZE = (112, 112)


class FaceRecognizer:
    """
    Generates a 192-dim L2-normalized face embedding from a BGR face crop.

    Model: MobileFaceNet TFLite (float32, batch=2, input 112×112, output 192-dim)
    Input:  BGR numpy crop (any resolution) — will be resized + normalized internally
    Output: 1-D numpy array, shape (192,), L2-normalized
    """

    def __init__(self, model_path: str = "recognition/models/face_embedding.tflite"):
        self._interpreter = tflite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()

        input_detail = self._interpreter.get_input_details()[0]
        self._input_idx  = input_detail["index"]
        self._output_idx = self._interpreter.get_output_details()[0]["index"]
        # Some model variants require batch > 1 — store and repeat accordingly
        self._batch_size = int(input_detail["shape"][0])

        logger.info("FaceRecognizer loaded: %s (batch_size=%d)", model_path, self._batch_size)

    def embed(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        """
        Compute a 192-dim L2-normalized embedding for a face crop.
        face_crop_bgr: BGR numpy array (H, W, 3), any resolution.
        Returns: np.ndarray shape (192,)
        """
        input_tensor = self._preprocess(face_crop_bgr)
        self._interpreter.set_tensor(self._input_idx, input_tensor)
        self._interpreter.invoke()

        raw = self._interpreter.get_tensor(self._output_idx)[0].astype(np.float32)
        norm = np.linalg.norm(raw) + 1e-10
        return raw / norm   # L2 normalize

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, _EMBED_INPUT_SIZE)
        arr = resized.astype(np.float32)
        arr = (arr - 127.5) / 128.0     # MobileNet standard normalization
        single = np.expand_dims(arr, axis=0)                      # [1, 112, 112, 3]
        return np.repeat(single, self._batch_size, axis=0)        # [batch, 112, 112, 3]
