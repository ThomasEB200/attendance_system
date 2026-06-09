"""
Color constants and conversion utilities for ILI9225 TFT (RGB565 format).

RGB565 bit layout:
  bits 15-11 = Red   (5 bits, 0-31)
  bits 10-5  = Green (6 bits, 0-63)
  bits  4-0  = Blue  (5 bits, 0-31)
"""

# ---------------------------------------------------------------------------
# Standard RGB565 color constants
# ---------------------------------------------------------------------------
BLACK       = 0x0000
WHITE       = 0xFFFF
RED         = 0xF800
GREEN       = 0x07E0
BLUE        = 0x001F
YELLOW      = 0xFFE0
CYAN        = 0x07FF
MAGENTA     = 0xF81F
ORANGE      = 0xFD20
PURPLE      = 0x8010
PINK        = 0xFE19
GRAY        = 0x8410
DARK_GRAY   = 0x4208
LIGHT_GRAY  = 0xC618
NAVY        = 0x000F
DARK_GREEN  = 0x03E0
DARK_CYAN   = 0x03EF
MAROON      = 0x7800
OLIVE       = 0x7BE0

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def rgb_to_565(r: int, g: int, b: int) -> int:
    """Convert 8-bit (r, g, b) values to a 16-bit RGB565 integer."""
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)


def bgr_to_565(b: int, g: int, r: int) -> int:
    """Convert 8-bit (b, g, r) OpenCV-style tuple to a 16-bit RGB565 integer."""
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)


def color_to_bytes(color: int) -> bytes:
    """Pack an RGB565 integer into 2 big-endian bytes ready for SPI."""
    return bytes([(color >> 8) & 0xFF, color & 0xFF])


def from_hex(hex_str: str) -> int:
    """Convert '#RRGGBB' or 'RRGGBB' hex string to RGB565.

    Example:
        red = from_hex('#FF0000')
    """
    h = hex_str.lstrip('#')
    return rgb_to_565(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
