import hashlib
import logging
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_IMAGE_GLOB = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

# Saved as: {"fingerprint": str, "embeddings": dict[str, list[np.ndarray]]}
_CACHE_VERSION = 1


class EmbeddingStore:
    """
    Manages a disk cache of face embeddings.

    On load_or_build():
      - Computes a fingerprint of known_faces_dir (file names + mtimes + sizes).
      - If cache exists and fingerprint matches  → load from cache (fast).
      - If cache missing or fingerprint changed  → rebuild from images + save.

    This means the cache is automatically invalidated when:
      - A new person folder is added
      - An existing photo is replaced / removed
      - Any image file is modified
    """

    def __init__(self, cache_path: str = "data/embeddings.cache"):
        self._cache_path = Path(cache_path)

    def load_or_build(
        self,
        known_faces_dir: str,
        get_embedding_fn: Callable[[Image.Image], np.ndarray],
    ) -> dict[str, list[np.ndarray]]:
        """
        Returns embeddings dict: {person_name: [embedding, ...]}
        get_embedding_fn: callable that takes a PIL Image and returns a 1-D np.ndarray.
        """
        faces_dir = Path(known_faces_dir)
        current_fp = self._fingerprint(faces_dir)

        cached = self._try_load(current_fp)
        if cached is not None:
            return cached

        embeddings = self._build(faces_dir, get_embedding_fn)
        self._save(embeddings, current_fp)
        return embeddings

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fingerprint(self, directory: Path) -> str:
        """MD5 of sorted (relative path, mtime, size) for every image in the tree."""
        if not directory.exists():
            return ""
        entries = []
        for pattern in _IMAGE_GLOB:
            for f in sorted(directory.rglob(pattern)):
                stat = f.stat()
                entries.append(f"{f.relative_to(directory)}|{stat.st_mtime}|{stat.st_size}")
        return hashlib.md5("\n".join(entries).encode()).hexdigest()

    def _try_load(self, expected_fp: str) -> dict[str, list[np.ndarray]] | None:
        if not self._cache_path.exists():
            logger.info("No embedding cache found — will build from scratch")
            return None
        try:
            with open(self._cache_path, "rb") as f:
                data = pickle.load(f)

            if data.get("version") != _CACHE_VERSION:
                logger.info("Cache format outdated — rebuilding")
                return None

            if data.get("fingerprint") != expected_fp:
                logger.info("Known faces changed — rebuilding embeddings")
                return None

            embeddings = data["embeddings"]
            total_imgs = sum(len(v) for v in embeddings.values())
            logger.info(
                "Loaded embedding cache: %d people, %d images",
                len(embeddings), total_imgs,
            )
            return embeddings

        except Exception as e:
            logger.warning("Failed to load cache (%s) — rebuilding", e)
            return None

    def _build(
        self,
        directory: Path,
        get_embedding_fn: Callable[[Image.Image], np.ndarray],
    ) -> dict[str, list[np.ndarray]]:
        embeddings: dict[str, list[np.ndarray]] = {}

        if not directory.exists():
            logger.warning("known_faces directory not found: %s", directory)
            return embeddings

        person_dirs = [d for d in sorted(directory.iterdir()) if d.is_dir()]
        if not person_dirs:
            logger.warning("No person sub-folders found in %s", directory)
            return embeddings

        logger.info("Building embeddings for %d people…", len(person_dirs))
        for person_dir in person_dirs:
            person_embeddings = []
            image_files = [
                f for pattern in _IMAGE_GLOB for f in person_dir.glob(pattern)
            ]
            for img_path in sorted(image_files):
                try:
                    img = Image.open(img_path).convert("RGB")
                    person_embeddings.append(get_embedding_fn(img))
                except Exception as e:
                    logger.error("Skipping %s: %s", img_path.name, e)

            if person_embeddings:
                embeddings[person_dir.name] = person_embeddings
                logger.info("  %-20s — %d image(s)", person_dir.name, len(person_embeddings))
            else:
                logger.warning("  No valid images for '%s' — skipped", person_dir.name)

        return embeddings

    def _save(self, embeddings: dict, fingerprint: str) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": _CACHE_VERSION,
            "fingerprint": fingerprint,
            "embeddings": embeddings,
        }
        with open(self._cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        total_imgs = sum(len(v) for v in embeddings.values())
        logger.info(
            "Embedding cache saved: %d people, %d images → %s",
            len(embeddings), total_imgs, self._cache_path,
        )
