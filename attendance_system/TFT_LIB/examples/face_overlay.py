"""
Face detection stream with bounding-box overlay.

Uses face_recognition (dlib-based) for detection and draws annotated boxes
on the TFT display in real time.

Run from the TFT_LIB directory:
    python examples/face_overlay.py

Requirements:
    pip install face_recognition
    sudo apt install python3-picamera2

Performance note:
  face_recognition is CPU-intensive. On Pi Zero 2W, process every N-th frame
  (DETECT_EVERY = 3) so that detection latency does not block the stream.
"""

import sys
import time
import signal

sys.path.insert(0, "..")

import numpy as np
from PIL import Image

from tft_lib import ILI9225
from tft_lib import text, graphics, colors
from tft_lib.stream import StreamDisplay, rgb888_to_rgb565_bytes

try:
    from picamera2 import Picamera2
except ImportError:
    print("picamera2 not found: sudo apt install python3-picamera2")
    sys.exit(1)

try:
    import face_recognition
except ImportError:
    print("face_recognition not found: pip install face_recognition")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DC_PIN       = 25
RST_PIN      = 27
BL_PIN       = 18
CAM_W        = 320
CAM_H        = 240
DETECT_EVERY = 3    # run face detection on every N-th frame (reduce CPU load)
SCALE_MODE   = 'fit'

TFT_W, TFT_H = 176, 220


def _cam_to_tft_coords(cam_box, cam_w, cam_h, tft_w, tft_h, scale_mode='fit'):
    """
    Map a (top, right, bottom, left) face box from camera resolution
    to TFT pixel coordinates, matching the StreamDisplay scale transform.

    Returns (x, y, w, h) in TFT pixels, or None if fully outside.
    """
    top, right, bottom, left = cam_box
    ratio = min(tft_w / cam_w, tft_h / cam_h)
    ow    = int(cam_w * ratio)
    oh    = int(cam_h * ratio)
    ox    = (tft_w - ow) // 2
    oy    = (tft_h - oh) // 2

    x0 = int(left   * ratio) + ox
    y0 = int(top    * ratio) + oy
    x1 = int(right  * ratio) + ox
    y1 = int(bottom * ratio) + oy

    x0 = max(x0, 0);  y0 = max(y0, 0)
    x1 = min(x1, tft_w - 1); y1 = min(y1, tft_h - 1)
    w  = x1 - x0
    h  = y1 - y0
    if w <= 0 or h <= 0:
        return None
    return x0, y0, w, h


def main():
    tft = ILI9225(dc_pin=DC_PIN, rst_pin=RST_PIN, bl_pin=BL_PIN)
    sd  = StreamDisplay(tft, scale_mode=SCALE_MODE)

    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"format": "RGB888", "size": (CAM_W, CAM_H)}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(0.5)

    running = True
    def _stop(sig, frame): nonlocal running; running = False
    signal.signal(signal.SIGINT, _stop)

    face_boxes = []   # last detected boxes (reused between detection frames)
    frame_idx  = 0
    t0 = time.time()

    print("Face detection stream started (Ctrl+C to stop)")

    while running:
        frame = cam.capture_array()          # (H, W, 3) uint8 RGB

        # ---- Run face detection every DETECT_EVERY frames ---------------
        if frame_idx % DETECT_EVERY == 0:
            # face_recognition expects RGB (same as picamera2 output)
            small = np.array(Image.fromarray(frame).resize(
                (CAM_W // 2, CAM_H // 2), Image.BILINEAR
            ))
            locs = face_recognition.face_locations(small, model="hog")
            # Scale back up to original cam resolution
            face_boxes = [(t*2, r*2, b*2, l*2) for (t, r, b, l) in locs]

        # ---- Push the raw video frame -----------------------------------
        sd.push_rgb(frame)

        # ---- Draw detection boxes on top --------------------------------
        for box in face_boxes:
            coords = _cam_to_tft_coords(box, CAM_W, CAM_H, TFT_W, TFT_H)
            if coords:
                gx, gy, gw, gh = coords
                graphics.draw_face_box(
                    tft, gx, gy, gw, gh,
                    label="Face",
                    box_color=colors.GREEN,
                    thickness=2,
                )

        # ---- FPS overlay ------------------------------------------------
        if frame_idx % 30 == 0:
            fps = frame_idx / max(time.time() - t0, 1e-6)
            text.draw_string(
                tft, 2, 2, f"FPS:{fps:4.1f}",
                fg=colors.YELLOW, bg=colors.BLACK, scale=1,
            )

        frame_idx += 1

    cam.stop()
    tft.cleanup()
    print(f"Stopped. Processed {frame_idx} frames.")


if __name__ == "__main__":
    main()
