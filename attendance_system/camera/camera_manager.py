import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IS_PI = False
try:
    from picamera2 import Picamera2
    _IS_PI = True
except ImportError:
    pass


class CameraManager:
    """
    Captures BGR numpy frames from either:
      - picamera2 (OV5647 on Pi Zero 2W)   when picamera2 is importable
      - cv2.VideoCapture (USB/built-in cam) as fallback for PC development
    """

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        cv2_device_index: int = 0,
        isp_controls: dict | None = None,
    ):
        self._width = width
        self._height = height

        if _IS_PI:
            self._cam = Picamera2()
            cfg = self._cam.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self._cam.configure(cfg)
            self._cam.start()

            if isp_controls:
                self._cam.set_controls(isp_controls)
                logger.info("picamera2 ISP controls applied: %s", isp_controls)

            logger.info("picamera2 started at %dx%d", width, height)
        else:
            self._cam = cv2.VideoCapture(cv2_device_index)
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self._cam.isOpened():
                raise RuntimeError(f"Cannot open camera device {cv2_device_index}")
            logger.info("cv2.VideoCapture started (device %d) at %dx%d", cv2_device_index, width, height)

    def capture(self) -> np.ndarray | None:
        """
        Returns a BGR numpy array (H, W, 3), or None on failure.
        Pi: picamera2 RGB888 → needs cvtColor to BGR for cv2 consistency.
        """
        if _IS_PI:
            # picamera2 "RGB888" format stores pixels as BGR in memory (libcamera/V4L2 convention)
            return self._cam.capture_array()

        ok, frame = self._cam.read()
        if not ok:
            logger.warning("VideoCapture.read() failed")
            return None
        return frame

    def close(self) -> None:
        if _IS_PI:
            self._cam.stop()
            self._cam.close()
        else:
            self._cam.release()
        logger.info("Camera closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
