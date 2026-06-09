from abc import ABC, abstractmethod
from pathlib import Path

from .employee_record import EmployeeRecord


class BaseProvider(ABC):
    """Abstract data source for employee records and reference photos."""

    @abstractmethod
    def fetch_employees(self) -> list[EmployeeRecord]:
        """Return the authoritative employee list with current photo hashes."""

    @abstractmethod
    def download_photo(self, employee_id: str, dest: Path) -> None:
        """Write the reference photo for employee_id to dest (overwrites if exists)."""
