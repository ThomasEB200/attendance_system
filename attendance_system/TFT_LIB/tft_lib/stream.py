"""
Camera-to-TFT streaming helpers for ILI9225 on Raspberry Pi Zero 2W.

Supports:
  - numpy  arrays  (H × W × 3, uint8, RGB or BGR)
  - PIL    Image   objects
  - JPEG   bytes   (raw bytes from a camera or network socket)

Pixel format conversion uses numpy when available for speed.
Falls back to pure-PIL path if numpy is not installed.

Typical face-recognition streaming loop:
    from picamera2 import Picamera2
    from tft_lib import ILI9225
    from tft_lib.stream import StreamDisplay

    tft = ILI9225(dc_pin=25, rst_pin=27, bl_pin=18)
    sd  = StreamDisplay(tft)
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "RGB888", "size": (320, 240)}
    ))
    cam.start()

    while True:
        frame = cam.capture_array()   # numpy (H, W, 3) RGB
        sd.push_rgb(frame)
"""

import io
from typing import Optional

from PIL import Image

from .ili9225 import ILI9225

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

# Display native resolution
_TFT_W = 176
_TFT_H = 220


# ---------------------------------------------------------------------------
# Low-level frame converters
# ---------------------------------------------------------------------------

def rgb888_to_rgb565_bytes(frame) -> bytes:
    """
    Convert a (H, W, 3) uint8 RGB numpy array to raw big-endian RGB565 bytes.

    This is the fastest path – a single numpy vectorised operation with no
    Python-level pixel loops.

    Args:
        frame: numpy array, shape (H, W, 3), dtype uint8, channels = RGB.

    Returns:
        bytes of length H*W*2 ready for push_image().
    """
    if not _NUMPY:
        raise RuntimeError("numpy is required for rgb888_to_rgb565_bytes(). "
                           "Install with: pip install numpy")
    f = frame.astype(">u2")          # promote to uint16, native-endian
    r = (f[:, :, 0] & 0xF8) << 8
    g = (f[:, :, 1] & 0xFC) << 3
    b =  f[:, :, 2] >> 3
    rgb565 = (r | g | b).astype(">u2")   # big-endian uint16
    return rgb565.tobytes()


def bgr888_to_rgb565_bytes(frame) -> bytes:
    """
    Convert a (H, W, 3) uint8 BGR numpy array (OpenCV format) to RGB565 bytes.

    Swaps the R and B channels before packing so the display shows correct
    colours from cv2 / face_recognition outputs.

    Args:
        frame: numpy array, shape (H, W, 3), dtype uint8, channels = BGR.
    """
    if not _NUMPY:
        raise RuntimeError("numpy is required. Install with: pip install numpy")
    f = frame.astype(">u2")
    r = (f[:, :, 2] & 0xF8) << 8   # channel 2 = R in BGR
    g = (f[:, :, 1] & 0xFC) << 3
    b =  f[:, :, 0] >> 3            # channel 0 = B in BGR
    rgb565 = (r | g | b).astype(">u2")
    return rgb565.tobytes()


def pil_to_rgb565_bytes(image: Image.Image) -> bytes:
    """
    Convert a PIL Image to raw big-endian RGB565 bytes.

    The image must already be the exact target size (no scaling is applied).
    Use StreamDisplay.push_pil() for automatic scaling.
    """
    img_rgb = image.convert("RGB")
    if _NUMPY:
        return rgb888_to_rgb565_bytes(np.array(img_rgb, dtype=">u2"))
    # Pure-PIL fallback (slow for large images)
    pixels = img_rgb.getdata()
    buf = bytearray(len(pixels) * 2)
    for i, (r, g, b) in enumerate(pixels):
        v = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        buf[i * 2]     = (v >> 8) & 0xFF
        buf[i * 2 + 1] =  v & 0xFF
    return bytes(buf)


# ---------------------------------------------------------------------------
# StreamDisplay – high-level helper
# ---------------------------------------------------------------------------

class StreamDisplay:
    """
    High-level wrapper that scales and pushes frames from a camera to a TFT.

    Scaling modes:
        'fit'     – letterbox / pillarbox to fit inside 176×220, preserving
                    aspect ratio. Black bars fill unused areas.
        'fill'    – crop-and-scale to fill the entire 176×220, no black bars
                    but edges may be cropped.
        'stretch' – stretch/squash to exactly 176×220, aspect ratio ignored.
        'none'    – no scaling; image is centred and cropped to 176×220.

    Example:
        sd = StreamDisplay(tft, scale_mode='fit')
        sd.push_rgb(frame)   # frame is (H, W, 3) numpy array, RGB
    """

    def __init__(
        self,
        tft:        ILI9225,
        scale_mode: str = 'fit',
        bg_color:   int = 0x0000,
    ):
        """
        Args:
            tft:        Initialised ILI9225 instance.
            scale_mode: 'fit', 'fill', 'stretch', or 'none'.
            bg_color:   RGB565 color used for letterbox/pillarbox bars.
        """
        self._tft        = tft
        self._scale_mode = scale_mode
        self._bg_color   = bg_color
        self._last_size  = None   # cache PIL resample target for speed

    # ------------------------------------------------------------------
    # Public push methods
    # ------------------------------------------------------------------

    def push_rgb(self, frame, x: int = 0, y: int = 0):
        """
        Push an RGB camera frame to the display.

        Args:
            frame: numpy array (H, W, 3) uint8 in RGB channel order.
                   Also accepts PIL Image in any mode (auto-converted).
            x, y:  Destination offset (used when scale_mode='none').
        """
        image = self._to_pil_rgb(frame)
        self._push_scaled(image)

    def push_bgr(self, frame):
        """
        Push an OpenCV / face_recognition BGR frame to the display.

        Args:
            frame: numpy array (H, W, 3) uint8 in BGR channel order.
        """
        if _NUMPY:
            import numpy as np
            frame_rgb = frame[:, :, ::-1]   # flip BGR → RGB in-place view
        else:
            frame_rgb = frame
        self.push_rgb(frame_rgb)

    def push_pil(self, image: Image.Image):
        """
        Push a PIL Image to the display (any mode, auto-converted to RGB).

        Args:
            image: PIL Image object (any size / mode).
        """
        self._push_scaled(image.convert("RGB"))

    def push_jpeg(self, jpeg_bytes: bytes):
        """
        Decode a JPEG byte string and push it to the display.

        This is useful when receiving MJPEG frames from an ESP32-CAM stream,
        a V4L2 camera, or an HTTP endpoint.

        Args:
            jpeg_bytes: Raw JPEG data as bytes.
        """
        image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        self._push_scaled(image)

    def push_full_rgb565(self, data: bytes):
        """
        Push a pre-converted RGB565 buffer to the full display (176×220).

        Use this when you manage colour conversion yourself for maximum speed.

        Args:
            data: bytes of length 176*220*2 = 77440, big-endian RGB565.
        """
        self._tft.push_image(0, 0, self._tft.width, self._tft.height, data)

    # ------------------------------------------------------------------
    # Scaling logic
    # ------------------------------------------------------------------

    def _to_pil_rgb(self, frame) -> Image.Image:
        """Convert numpy array or PIL Image to RGB PIL Image."""
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if _NUMPY:
            import numpy as np
            return Image.fromarray(frame.astype("uint8"), "RGB")
        raise TypeError("frame must be a PIL Image or numpy array. "
                        "Install numpy: pip install numpy")

    def _push_scaled(self, image: Image.Image):
        """Scale image to TFT resolution, convert to RGB565, and push."""
        tw = self._tft.width
        th = self._tft.height
        iw, ih = image.size
        mode = self._scale_mode

        if mode == 'stretch' or (iw == tw and ih == th):
            scaled = image.resize((tw, th), Image.BILINEAR)
            data   = pil_to_rgb565_bytes(scaled)
            self._tft.push_image(0, 0, tw, th, data)
            return

        if mode == 'fit':
            ratio  = min(tw / iw, th / ih)
            nw, nh = int(iw * ratio), int(ih * ratio)
            scaled = image.resize((nw, nh), Image.BILINEAR)
            # Black bars
            ox = (tw - nw) // 2
            oy = (th - nh) // 2
            if ox > 0:
                # Left and right bars
                bg_bytes = _solid_bar(ox, th, self._bg_color)
                self._tft.push_image(0,       0, ox, th, bg_bytes)
                self._tft.push_image(ox + nw, 0, tw - ox - nw, th, bg_bytes)
            if oy > 0:
                # Top and bottom bars
                bg_bytes = _solid_bar(tw, oy, self._bg_color)
                self._tft.push_image(0, 0,       tw, oy, bg_bytes)
                self._tft.push_image(0, oy + nh, tw, th - oy - nh, bg_bytes)
            data = pil_to_rgb565_bytes(scaled)
            self._tft.push_image(ox, oy, nw, nh, data)
            return

        if mode == 'fill':
            ratio  = max(tw / iw, th / ih)
            nw, nh = int(iw * ratio), int(ih * ratio)
            scaled = image.resize((nw, nh), Image.BILINEAR)
            # Centre-crop
            cx = (nw - tw) // 2
            cy = (nh - th) // 2
            cropped = scaled.crop((cx, cy, cx + tw, cy + th))
            data = pil_to_rgb565_bytes(cropped)
            self._tft.push_image(0, 0, tw, th, data)
            return

        if mode == 'none':
            # Centre-align, crop if larger, pad with background if smaller
            ox = max((tw - iw) // 2, 0)
            oy = max((th - ih) // 2, 0)
            crop_l = max((iw - tw) // 2, 0)
            crop_t = max((ih - th) // 2, 0)
            visible_w = min(iw - crop_l, tw - ox)
            visible_h = min(ih - crop_t, th - oy)
            if ox > 0 or oy > 0:
                self._tft.fill_screen(self._bg_color)
            region = image.crop((crop_l, crop_t,
                                  crop_l + visible_w, crop_t + visible_h))
            data = pil_to_rgb565_bytes(region)
            self._tft.push_image(ox, oy, visible_w, visible_h, data)
            return

        raise ValueError(f"Unknown scale_mode '{mode}'. "
                         "Use 'fit', 'fill', 'stretch', or 'none'.")


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _solid_bar(w: int, h: int, color: int) -> bytes:
    """Return a raw RGB565 buffer filled with a solid color."""
    hi = (color >> 8) & 0xFF
    lo =  color & 0xFF
    return bytes([hi, lo] * (w * h))
