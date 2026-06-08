import logging
from pathlib import Path

import numpy as np
from PIL import Image

from embedding_store import EmbeddingStore

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_EMBEDDING_INPUT_SIZE = (112, 112)   # MediaPipe face embedder input size
_UNKNOWN_LABEL = "Unknown"


class FaceRecognizer:
    """
    Identifies faces using a MobileNetV2 TFLite embedding model.

    If model_path does not exist, the recognizer still runs but returns
    Unknown for every face — useful for testing detection before the
    embedding model is available.

    Known faces directory layout:
      known_faces/
        Nguyen Van A/
          photo1.jpg
        Tran Thi B/
          photo1.jpg
    """

    def __init__(
        self,
        model_path: str,
        known_faces_dir: str,
        similarity_threshold: float = 0.45,
        cache_path: str = "data/embeddings.cache",
    ):
        self._threshold = similarity_threshold
        self._interpreter = None
        self._input_idx = None
        self._output_idx = None

        if Path(model_path).exists():
            self._interpreter = tflite.Interpreter(model_path=model_path)
            self._interpreter.allocate_tensors()
            input_detail  = self._interpreter.get_input_details()[0]
            self._input_idx  = input_detail["index"]
            self._output_idx = self._interpreter.get_output_details()[0]["index"]
            self._batch_size = input_detail["shape"][0]   # some models require batch > 1
        else:
            logger.warning(
                "Embedding model not found at '%s' — faces will show as Unknown", model_path
            )

        self._known_faces_dir = known_faces_dir
        self._store = EmbeddingStore(cache_path)
        self._known_embeddings = (
            self._store.load_or_build(known_faces_dir, self._get_embedding)
            if self._interpreter else {}
        )

    def identify(self, face_image: Image.Image) -> dict:
        """
        Returns {"name": str, "confidence": float}.
        name is Unknown when model is missing, no known faces, or similarity < threshold.
        """
        if self._interpreter is None or not self._known_embeddings:
            return {"name": _UNKNOWN_LABEL, "confidence": 0.0}

        embedding = self._get_embedding(face_image)
        best_name = _UNKNOWN_LABEL
        best_score = 0.0

        for name, embeddings in self._known_embeddings.items():
            score = max(self._cosine_similarity(embedding, e) for e in embeddings)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self._threshold:
            return {"name": _UNKNOWN_LABEL, "confidence": round(float(best_score), 3)}
        return {"name": best_name, "confidence": round(float(best_score), 3)}

    def reload(self) -> None:
        """Re-scan known_faces and rebuild cache if anything changed."""
        if self._interpreter:
            self._known_embeddings = self._store.load_or_build(
                self._known_faces_dir, self._get_embedding
            )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_embedding(self, image: Image.Image) -> np.ndarray:
        input_tensor = self._preprocess(image)
        self._interpreter.set_tensor(self._input_idx, input_tensor)
        self._interpreter.invoke()
        embedding = self._interpreter.get_tensor(self._output_idx)[0]  # take first item
        return embedding / (np.linalg.norm(embedding) + 1e-10)  # L2 normalize

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.resize(_EMBEDDING_INPUT_SIZE)
        arr = np.array(resized, dtype=np.float32)
        arr = (arr - 127.5) / 128.0  # MobileNetV2 standard normalization
        single = np.expand_dims(arr, axis=0)                    # [1, 112, 112, 3]
        return np.repeat(single, self._batch_size, axis=0)      # [batch, 112, 112, 3]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # both are already L2-normalized
