"""
AttendanceLogger — records a successful face recognition to Firestore once per day,
and resets all employees to "absent" every day at a configurable hour.

Rules:
  - One Firestore write per employee per calendar day (device local time).
  - Daily reset runs at reset_hour:00 via a background scheduler thread.
  - On first startup, reset also fires immediately so the day starts clean.
"""

import logging
import threading
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)


class AttendanceLogger:

    def __init__(self, credentials_path: str | None = None):
        """
        Args:
            credentials_path: Path to Firebase service account JSON.
                               None → console-only mode (no Firestore writes).
        """
        self._recorded_on: dict[str, str] = {}
        self._fs = None
        self._SERVER_TIMESTAMP = None

        if credentials_path:
            try:
                import firebase_admin
                from firebase_admin import credentials, firestore as fs_module
                if not firebase_admin._apps:
                    cred = credentials.Certificate(credentials_path)
                    firebase_admin.initialize_app(cred)
                self._fs = fs_module.client()
                self._SERVER_TIMESTAMP = fs_module.SERVER_TIMESTAMP
                logger.info("AttendanceLogger: Firebase connected")
            except Exception as exc:
                logger.warning(
                    "AttendanceLogger: Firebase init failed (%s) — console-only mode", exc
                )

    # ------------------------------------------------------------------
    # Daily scheduler
    # ------------------------------------------------------------------

    def start_daily_reset(self, reset_hour: int, stop_event: threading.Event, reset_minute: int = 0) -> None:
        """
        Start a background thread that resets all employees to absent
        every day at reset_hour:reset_minute (local time).
        Does NOT reset on startup — avoids wiping today's attendance after a power cut.
        """
        thread = threading.Thread(
            target=self._scheduler_loop,
            args=(reset_hour, reset_minute, stop_event),
            daemon=True,
            name="attendance-scheduler",
        )
        thread.start()
        logger.info("Daily reset scheduler started — next reset at %02d:%02d", reset_hour, reset_minute)

    def _scheduler_loop(self, reset_hour: int, reset_minute: int, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            now = datetime.now()
            target = now.replace(hour=reset_hour, minute=reset_minute, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)

            wait_seconds = (target - now).total_seconds()
            logger.debug(
                "Next daily reset in %.0fs  (%s)", wait_seconds, target.strftime("%Y-%m-%d %H:%M")
            )

            stop_event.wait(timeout=wait_seconds)
            if stop_event.is_set():
                break

            logger.info("Daily reset triggered (%02d:%02d)", reset_hour, reset_minute)
            self.reset_all_to_absent()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reset_all_to_absent(self) -> None:
        """Set every employee's status to 'absent' and clear in-memory records."""
        self._recorded_on.clear()

        if self._fs is None:
            logger.info("AttendanceLogger: console-only mode — Firestore reset skipped")
            return

        try:
            docs = list(self._fs.collection("employees").stream())
            batch = self._fs.batch()
            for doc in docs:
                batch.update(doc.reference, {
                    "status": "absent",
                    "updatedAt": self._SERVER_TIMESTAMP,
                })
            batch.commit()
            logger.info("Reset %d employee(s) → absent", len(docs))
        except Exception as exc:
            logger.error("reset_all_to_absent failed: %s", exc)

    def record_present(self, employee_id: str, name: str) -> bool:
        """
        Mark employee as present if not already recorded today.
        Returns True when a write was performed, False when skipped.
        """
        today = date.today().isoformat()

        if self._recorded_on.get(employee_id) == today:
            return False

        self._recorded_on[employee_id] = today

        if self._fs is not None:
            try:
                self._fs.collection("employees").document(employee_id).update({
                    "status": "present",
                    "updatedAt": self._SERVER_TIMESTAMP,
                })
                logger.info(
                    "Attendance → present: %s (%s)  date=%s", name, employee_id, today
                )
            except Exception as exc:
                logger.error("Firestore write failed for %s: %s", employee_id, exc)
        else:
            logger.info(
                "Attendance (console): %s (%s)  date=%s", name, employee_id, today
            )

        return True
