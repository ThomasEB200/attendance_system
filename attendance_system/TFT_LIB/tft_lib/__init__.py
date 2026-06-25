"""
TFT_LIB – MicroPython-style TFT library for ILI9225 on Raspberry Pi Zero 2W.

Quick start:
    from tft_lib import ILI9225, colors, text, graphics, stream
    from tft_lib.colors import RED, WHITE, BLACK

    tft = ILI9225(dc_pin=25, rst_pin=27, bl_pin=18)
    tft.fill_screen(BLACK)
    text.draw_string(tft, 10, 10, "Hello!", fg=WHITE, bg=BLACK)
"""

from .ili9225 import ILI9225, PORTRAIT_0, LANDSCAPE_90, PORTRAIT_180, LANDSCAPE_270
from .st7735 import ST7735
from . import colors, fonts, graphics, text, stream

__all__ = [
    "ILI9225",
    "ST7735",
    "PORTRAIT_0", "LANDSCAPE_90", "PORTRAIT_180", "LANDSCAPE_270",
    "colors",
    "fonts",
    "graphics",
    "text",
    "stream",
]
