from dataclasses import dataclass


@dataclass(frozen=True)
class EmployeeRecord:
    employee_id: str
    name: str
    photo_hash: str  # MD5 hex of photo bytes — used to detect remote changes
