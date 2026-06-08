import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


class ServerClient:
    """Posts recognition results to the web server for display."""

    def __init__(self, server_url: str, timeout: int = 5):
        self._base_url = server_url.rstrip("/")
        self._timeout = timeout

    def post_result(self, faces: list[dict]) -> bool:
        payload = {
            "faces": faces,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            res = requests.post(
                f"{self._base_url}/result",
                json=payload,
                timeout=self._timeout,
            )
            res.raise_for_status()
            return True
        except Exception as e:
            logger.error("Failed to post result: %s", e)
            return False
