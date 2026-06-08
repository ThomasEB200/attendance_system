import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Camera:
    """
    Wraps picamera2 for capturing frames from the Pi camera module.
    picamera2 is pre-installed on Raspberry Pi OS (Bullseye+).
    Install manually: pip install picamera2
    """

    def __init__(self, width: int = 640, height: int = 480):
        from picamera2 import Picamera2

        self._cam = Picamera2()
        config = self._cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.start()
        logger.info("Camera started at %dx%d", width, height)

    def capture(self) -> Image.Image:
        """Capture a single frame and return as PIL Image (RGB)."""
        frame = self._cam.capture_array()
        return Image.fromarray(frame)

    def close(self) -> None:
        self._cam.stop()
        self._cam.close()
        logger.info("Camera closed")
