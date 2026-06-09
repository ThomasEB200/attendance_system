import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    name: str
    confidence: float
    matched: bool        # False when confidence < threshold or DB is empty
    employee_id: str = ""  # empty string when not matched


class FaceDB:
    """
    Stores one embedding per person and finds the closest match.

    DB format (faces.pkl):
      {
        "NV001": {"name": "Nguyen Van A", "embedding": np.ndarray(128,)},
        ...
      }

    Phase 1 constraint: exactly 1 reference embedding per person.
    Matching uses cosine similarity (both vectors are L2-normalized,
    so dot-product == cosine similarity).
    """

    UNKNOWN = "Unknown"

    def __init__(
        self,
        embeddings_path: str = "data/embeddings.pkl",
        similarity_threshold: float = 0.75,
        low_confidence_threshold: float = 0.60,
    ):
        self._path = Path(embeddings_path)
        self._threshold = similarity_threshold
        self._low_threshold = low_confidence_threshold
        self._db: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_match(self, embedding: np.ndarray) -> MatchResult:
        """
        Compare embedding against all stored entries.
        Returns MatchResult with matched=True only when similarity >= threshold.
        Similarity in range [low_threshold, threshold) → matched=False (retry).
        """
        if not self._db:
            logger.warning("FaceDB is empty — enroll at least one person first")
            return MatchResult(name=self.UNKNOWN, confidence=0.0, matched=False)

        best_id   = ""
        best_name = self.UNKNOWN
        best_score = 0.0

        for employee_id, entry in self._db.items():
            score = float(np.dot(embedding, entry["embedding"]))   # cosine similarity
            if score > best_score:
                best_score = score
                best_name  = entry["name"]
                best_id    = employee_id

        matched = best_score >= self._threshold
        return MatchResult(
            name=best_name if matched else self.UNKNOWN,
            confidence=round(best_score, 3),
            matched=matched,
            employee_id=best_id if matched else "",
        )

    def add_person(self, employee_id: str, name: str, embedding: np.ndarray) -> None:
        """Add or overwrite a person's reference embedding."""
        self._db[employee_id] = {"name": name, "embedding": embedding.copy()}
        self._save()
        logger.info("Enrolled: %s (%s)", name, employee_id)

    def remove_person(self, employee_id: str) -> bool:
        if employee_id not in self._db:
            return False
        name = self._db.pop(employee_id)["name"]
        self._save()
        logger.info("Removed: %s (%s)", name, employee_id)
        return True

    def list_people(self) -> list[tuple[str, str]]:
        """Returns list of (employee_id, name) sorted by id."""
        return sorted((eid, entry["name"]) for eid, entry in self._db.items())

    def is_empty(self) -> bool:
        return len(self._db) == 0

    def clear(self) -> None:
        """Remove all entries from the in-memory DB. Does not write to disk."""
        self._db.clear()
        logger.debug("FaceDB cleared (in-memory only)")

    def reload(self) -> None:
        """Re-read faces.pkl from disk (e.g., after a new enrollment)."""
        self._load()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            logger.info("No embeddings file at %s — starting with empty DB", self._path)
            self._db = {}
            return
        try:
            with open(self._path, "rb") as f:
                self._db = pickle.load(f)
            logger.info("FaceDB loaded: %d person(s) from %s", len(self._db), self._path)
        except Exception as e:
            logger.error("Failed to load embeddings: %s — starting empty", e)
            self._db = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(self._db, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("FaceDB saved: %d person(s) → %s", len(self._db), self._path)
