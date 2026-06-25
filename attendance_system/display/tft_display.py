"""
TFT display output for ILI9225 on Raspberry Pi Zero 2W.

Reads BGR annotated frames from a shared frame buffer and streams them to the
176×220 TFT display via SPI.  Designed to run in a background daemon thread
alongside the face-recognition pipeline.

Hardware: KMRTM2018-SPI V.1, ILI9225 controller.
Dependencies: spidev, RPi.GPIO  (Raspberry Pi Linux only).
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Add TFT_LIB/ to sys.path so that `from tft_lib import ...` works.
# This resolves to attendance_system/TFT_LIB/tft_lib/__init__.py
_TFT_LIB_ROOT = Path(__file__).resolve().parent.parent / "TFT_LIB"
if str(_TFT_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_TFT_LIB_ROOT))

_TFT_AVAILABLE = False
try:
    from tft_lib import ST7735
    from tft_lib.stream import StreamDisplay
    _TFT_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Return True when tft_lib and its hardware dependencies are importable."""
    return _TFT_AVAILABLE


class TFTDisplay:
    """
    Thin wrapper around ST7735 + StreamDisplay for live BGR frame streaming.

    Raises RuntimeError on construction when the required hardware libraries
    (spidev, RPi.GPIO) are not present — the caller can catch this to disable
    TFT output gracefully and fall back to web streaming.
    """

    def __init__(
        self,
        dc_pin:     int,
        rst_pin:    int,
        bl_pin:     int,
        scale_mode: str = "fill",
        spi_speed:  int = 16_000_000,
    ):
        if not _TFT_AVAILABLE:
            raise RuntimeError(
                "tft_lib requires spidev + RPi.GPIO — only available on Raspberry Pi"
            )
        self._tft = ST7735(
            dc_pin=dc_pin, rst_pin=rst_pin, bl_pin=bl_pin, spi_speed=spi_speed
        )
        self._sd = StreamDisplay(self._tft, scale_mode=scale_mode)
        logger.info(
            "TFT ready: %dx%d px  scale=%s  SPI=%d MHz",
            self._tft.width, self._tft.height, scale_mode, spi_speed // 1_000_000,
        )

    @property
    def width(self) -> int:
        return self._tft.width

    @property
    def height(self) -> int:
        return self._tft.height

    def push_rgb(self, frame: np.ndarray) -> None:
        """Push one RGB numpy frame to the display (blocks during SPI transfer)."""
        self._sd.push_rgb(frame)

    def push_bgr(self, frame: np.ndarray) -> None:
        """Push one BGR numpy frame (OpenCV convention) to the display."""
        self._sd.push_bgr(frame)

    def show_splash(self, line1: str, line2: str = "") -> None:
        """Clear the screen and render a centred two-line splash message."""
        from tft_lib import colors, text
        self._tft.fill_screen(colors.BLACK)
        cy = (self._tft.height - 28) // 2
        text.draw_centered(self._tft, cy, line1, fg=colors.WHITE, bg=colors.BLACK, scale=2)
        if line2:
            text.draw_centered(
                self._tft, cy + 18, line2, fg=colors.YELLOW, bg=colors.BLACK, scale=1
            )

    def cleanup(self) -> None:
        """Release SPI and GPIO resources."""
        self._tft.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()


def tft_stream_loop(
    get_frame:  Callable[[], "np.ndarray | None"],
    frame_lock: threading.Lock,
    tft:        TFTDisplay,
    stop_event: threading.Event,
    target_fps: float = 15.0,
) -> None:
    """
    Blocking loop that reads BGR frames and pushes them to the TFT.

    Call this from a daemon thread; set stop_event to terminate cleanly.

    At 16 MHz SPI, a full 176×220 frame (77 440 bytes) takes ~38 ms, so the
    practical ceiling is ~20 fps.  The default 15 fps leaves headroom for the
    face-recognition pipeline thread running concurrently.

    Args:
        get_frame:   Called under frame_lock; returns the latest BGR frame or None.
        frame_lock:  Lock protecting the shared frame buffer.
        tft:         Initialised TFTDisplay instance.
        stop_event:  Threading event — set it to stop the loop.
        target_fps:  Maximum push rate in frames per second.
    """
    interval  = 1.0 / max(target_fps, 1.0)
    next_push = time.perf_counter()

    while not stop_event.is_set():
        now = time.perf_counter()
        if now < next_push:
            time.sleep(next_push - now)
            continue

        with frame_lock:
            frame = get_frame()

        if frame is None:
            time.sleep(0.02)
            continue

        try:
            tft.push_bgr(frame)  # BGR pipeline → RGB for display
        except Exception:
            logger.exception("TFT push error")

        next_push = time.perf_counter() + interval
