"""
FirebaseProvider — reads employee data from Firestore.

Firestore schema (collection: employees, document ID = employeeId):
    employeeId : string       e.g. "NV001"
    fullName   : string       e.g. "Võ Minh Thái"
    photo      : string       data:image/jpeg;base64,<base64>
    status     : string       "present" | "absent"
    createdAt  : timestamp
    updatedAt  : timestamp

Requirements:
    pip install firebase-admin

Credentials:
    Firebase Console → Project Settings → Service accounts → Generate new private key
    Save as attendance_system/firebase_credentials.json  (or configure via sync.firebase.credentials)
"""

import base64
import hashlib
import logging
from pathlib import Path

from .base_provider import BaseProvider
from .employee_record import EmployeeRecord

logger = logging.getLogger(__name__)


class FirebaseProvider(BaseProvider):

    def __init__(self, credentials_path: str):
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized with credentials: %s", credentials_path)

        self._fs = firestore.client()
        # Cache decoded photo bytes from fetch_employees() to avoid re-fetching on download
        self._photo_cache: dict[str, bytes] = {}

    def fetch_employees(self) -> list[EmployeeRecord]:
        self._photo_cache.clear()
        docs = list(self._fs.collection("employees").stream())
        records = []

        for doc in docs:
            data = doc.to_dict()
            employee_id = data.get("employeeId", doc.id)
            name = data.get("fullName", "")
            photo_b64 = data.get("photo", "")

            if not photo_b64:
                logger.warning("Employee %s has no photo field — skipping", employee_id)
                continue

            photo_bytes = _decode_data_uri(photo_b64)
            if photo_bytes is None:
                logger.warning("Employee %s has invalid photo data — skipping", employee_id)
                continue

            self._photo_cache[employee_id] = photo_bytes
            photo_hash = hashlib.md5(photo_bytes).hexdigest()

            records.append(EmployeeRecord(
                employee_id=employee_id,
                name=name,
                photo_hash=photo_hash,
            ))

        logger.info("Fetched %d employee(s) from Firestore", len(records))
        return records

    def download_photo(self, employee_id: str, dest: Path) -> None:
        photo_bytes = self._photo_cache.get(employee_id)

        if photo_bytes is None:
            # Cache miss — re-fetch individual document (e.g. if called without fetch_employees first)
            doc = self._fs.collection("employees").document(employee_id).get()
            if not doc.exists:
                raise FileNotFoundError(
                    f"No Firestore document found for employeeId={employee_id}"
                )
            photo_b64 = doc.to_dict().get("photo", "")
            photo_bytes = _decode_data_uri(photo_b64)
            if photo_bytes is None:
                raise ValueError(f"Invalid photo data for employeeId={employee_id}")

        dest.write_bytes(photo_bytes)
        logger.debug("Photo saved: %s (%d bytes)", dest.name, len(photo_bytes))


def _decode_data_uri(data_uri: str) -> bytes | None:
    """Strip 'data:image/...;base64,' prefix and decode to raw image bytes."""
    try:
        if "," in data_uri:
            data_uri = data_uri.split(",", 1)[1]
        return base64.b64decode(data_uri)
    except Exception:
        return None
