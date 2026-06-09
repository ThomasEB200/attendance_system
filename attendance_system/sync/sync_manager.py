"""
SyncManager — keeps local face embeddings in sync with an authoritative data source.

Boot sequence:
  1. fetch_employees() from provider
  2. compare photo hashes with local manifest.json
  3. if different (or no manifest) → clear employee_photos/, re-download, re-enroll all
  4. update manifest with successfully enrolled employees

If the provider is unreachable, logs a warning and continues with existing embeddings.
"""

import json
import logging
import math
import shutil
from pathlib import Path

from .base_provider import BaseProvider
from .employee_record import EmployeeRecord

logger = logging.getLogger(__name__)


class SyncManager:

    def __init__(self, provider: BaseProvider, data_dir: Path):
        self._provider   = provider
        self._photos_dir = data_dir / "employee_photos"
        self._manifest   = data_dir / "manifest.json"
        self._photos_dir.mkdir(parents=True, exist_ok=True)

    def sync_if_needed(self, detector, recognizer, db) -> bool:
        """
        Fetch remote employee list and rebuild embeddings if data changed.
        Returns True when a sync was performed, False when already up-to-date.
        Falls back to existing embeddings silently when provider is unreachable.
        """
        try:
            remote = self._provider.fetch_employees()
        except Exception as exc:
            logger.warning(
                "Provider unreachable (%s) — continuing with existing local embeddings", exc
            )
            return False

        if not remote:
            logger.warning("Provider returned empty employee list — skipping sync")
            return False

        if self._is_in_sync(remote):
            logger.info(
                "Embeddings up-to-date (%d employee(s)) — skipping sync", len(remote)
            )
            return False

        logger.info(
            "Data change detected — rebuilding embeddings for %d employee(s)", len(remote)
        )
        self._full_sync(remote, detector, recognizer, db)
        return True

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _is_in_sync(self, remote: list[EmployeeRecord]) -> bool:
        if not self._manifest.exists():
            return False
        try:
            local = json.loads(self._manifest.read_text())
        except Exception:
            return False
        remote_map = {r.employee_id: r.photo_hash for r in remote}
        return remote_map == local.get("employees", {})

    def _full_sync(
        self,
        remote: list[EmployeeRecord],
        detector,
        recognizer,
        db,
    ) -> None:
        # Wipe stale managed photos
        if self._photos_dir.exists():
            shutil.rmtree(self._photos_dir)
        self._photos_dir.mkdir(parents=True)

        db.clear()

        enrolled: list[str] = []
        failed:   list[str] = []

        for record in remote:
            dest = self._photos_dir / f"{record.employee_id}.jpg"

            try:
                self._provider.download_photo(record.employee_id, dest)
            except Exception as exc:
                logger.error(
                    "Photo download failed for %s (%s): %s",
                    record.name, record.employee_id, exc,
                )
                failed.append(record.employee_id)
                continue

            if self._enroll(dest, record, detector, recognizer, db):
                enrolled.append(record.employee_id)
            else:
                failed.append(record.employee_id)

        logger.info(
            "Sync complete: %d enrolled, %d failed", len(enrolled), len(failed)
        )
        if failed:
            logger.warning("Failed to enroll: %s", failed)

        enrolled_set = set(enrolled)
        self._manifest.write_text(json.dumps({
            "employees": {
                r.employee_id: r.photo_hash
                for r in remote
                if r.employee_id in enrolled_set
            }
        }, indent=2))

    def _enroll(
        self,
        photo_path: Path,
        record: EmployeeRecord,
        detector,
        recognizer,
        db,
    ) -> bool:
        from utils.image_utils import load_image_bgr

        try:
            frame = load_image_bgr(str(photo_path))
        except Exception as exc:
            logger.error(
                "Cannot load photo for %s (%s): %s",
                record.name, record.employee_id, exc,
            )
            return False

        detections = detector.detect(frame)
        if not detections:
            logger.error(
                "No face detected in reference photo for %s (%s) — "
                "check photo quality or face visibility",
                record.name, record.employee_id,
            )
            return False

        det = max(detections, key=lambda d: d.width * d.height)
        face_crop = det.align_bgr(frame)
        embedding = recognizer.embed(face_crop)
        db.add_person(record.employee_id, record.name, embedding)

        eye_d = 0.0
        if len(det.keypoints) >= 2:
            re, le = det.keypoints[0], det.keypoints[1]
            eye_d = math.hypot(le[0] - re[0], le[1] - re[1])

        logger.info(
            "Enrolled %-20s  id=%-8s  face=%dx%d  eye_dist=%.0f",
            record.name, record.employee_id,
            det.width, det.height, eye_d,
        )
        return True
