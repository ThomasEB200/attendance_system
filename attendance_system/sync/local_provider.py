"""
LocalProvider — development stub that reads from local files.

Directory layout expected under data_dir:
    employees.yaml          — employee list (id, name, photo filename)
    employee_source/        — source reference photos (never auto-deleted)
    employee_photos/        — managed by SyncManager (overwritten on sync)

Swap for FirebaseProvider once Firestore credentials are available.
"""

import hashlib
import shutil
from pathlib import Path

import yaml

from .base_provider import BaseProvider
from .employee_record import EmployeeRecord


class LocalProvider(BaseProvider):

    def __init__(
        self,
        data_dir: Path,
        config_filename: str = "employees.yaml",
        source_subdir: str = "employee_source",
    ):
        self._source_dir = data_dir / source_subdir
        config_path = data_dir / config_filename
        with open(config_path) as f:
            self._config = yaml.safe_load(f) or {}

    def fetch_employees(self) -> list[EmployeeRecord]:
        records = []
        for entry in self._config.get("employees", []):
            photo_path = self._source_dir / entry["photo"]
            if not photo_path.exists():
                continue
            photo_hash = hashlib.md5(photo_path.read_bytes()).hexdigest()
            records.append(EmployeeRecord(
                employee_id=entry["id"],
                name=entry["name"],
                photo_hash=photo_hash,
            ))
        return records

    def download_photo(self, employee_id: str, dest: Path) -> None:
        for entry in self._config.get("employees", []):
            if entry["id"] == employee_id:
                src = self._source_dir / entry["photo"]
                shutil.copy2(str(src), str(dest))
                return
        raise FileNotFoundError(f"No photo configured for employee_id={employee_id}")
