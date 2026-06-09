"""
Live camera stream to TFT using picamera2 (Raspberry Pi camera module).

Run from the TFT_LIB directory:
    python examples/camera_stream.py

Requirements:
    sudo apt install python3-picamera2   (or: pip install picamera2)

The script captures frames at full sensor resolution, scales them to
176×220 (fit mode), converts to RGB565, and pushes to the display.
Typical throughput on Pi Zero 2W: 5-10 fps at 16 MHz SPI.
"""

import sys
import time
import signal

sys.path.insert(0, "..")

from tft_lib import ILI9225
from tft_lib import text, colors
from tft_lib.stream import StreamDisplay

try:
    from picamera2 import Picamera2
except ImportError:
    print("picamera2 not found. Install: sudo apt install python3-picamera2")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DC_PIN     = 25
RST_PIN    = 27
BL_PIN     = 18
SCALE_MODE = 'fit'      # 'fit', 'fill', 'stretch', 'none'
SHOW_FPS   = True       # overlay FPS counter on the display
CAM_W      = 320        # capture resolution – wider = more detail, slower
CAM_H      = 240


def main():
    tft = ILI9225(dc_pin=DC_PIN, rst_pin=RST_PIN, bl_pin=BL_PIN)
    sd  = StreamDisplay(tft, scale_mode=SCALE_MODE)

    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"format": "RGB888", "size": (CAM_W, CAM_H)}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(0.5)   # let camera auto-expose

    # Graceful shutdown on Ctrl+C
    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)

    frames  = 0
    t_start = time.time()

    print(f"Streaming {CAM_W}×{CAM_H} → 176×220 TFT  (Ctrl+C to stop)")

    while running:
        frame = cam.capture_array()   # (H, W, 3) uint8 RGB
        sd.push_rgb(frame)

        frames += 1
        elapsed = time.time() - t_start
        if SHOW_FPS and frames % 30 == 0:
            fps = frames / elapsed
            text.draw_string(
                tft, 2, 2, f"FPS:{fps:4.1f}",
                fg=colors.YELLOW, bg=colors.BLACK, scale=1,
            )

    cam.stop()
    tft.cleanup()
    print(f"\nStopped. Average FPS: {frames / (time.time() - t_start):.1f}")


if __name__ == "__main__":
    main()
