"""
Basic display example – fills the screen with colors and draws shapes.

Run from the TFT_LIB directory:
    python examples/basic_display.py
"""

import sys
import time

sys.path.insert(0, "..")          # allow "from tft_lib import ..." without install

from tft_lib import ILI9225, PORTRAIT_0, LANDSCAPE_90
from tft_lib import colors, graphics, text

# ---------------------------------------------------------------------------
# Pin configuration – adjust to match your wiring
# ---------------------------------------------------------------------------
DC_PIN  = 25
RST_PIN = 27
BL_PIN  = 18


def main():
    with ILI9225(dc_pin=DC_PIN, rst_pin=RST_PIN, bl_pin=BL_PIN) as tft:
        # ----- Basic fills --------------------------------------------------
        print("Filling screen colors …")
        for color in (colors.RED, colors.GREEN, colors.BLUE,
                      colors.YELLOW, colors.CYAN, colors.MAGENTA):
            tft.fill_screen(color)
            time.sleep(0.3)

        tft.fill_screen(colors.BLACK)
        time.sleep(0.2)

        # ----- Rectangles ---------------------------------------------------
        graphics.fill_rect(tft,  5,  5, 80, 60, colors.RED)
        graphics.draw_rect(tft,  5,  5, 80, 60, colors.WHITE)
        graphics.fill_rect(tft, 90,  5, 80, 60, colors.BLUE)
        graphics.draw_rect(tft, 90,  5, 80, 60, colors.WHITE)

        # ----- Circles ------------------------------------------------------
        graphics.fill_circle(tft,  44, 130, 30, colors.GREEN)
        graphics.draw_circle(tft, 132, 130, 30, colors.YELLOW)

        # ----- Lines --------------------------------------------------------
        for i in range(0, 176, 20):
            graphics.draw_line(tft, i, 170, 176 - i, 220, colors.CYAN)

        # ----- Text ---------------------------------------------------------
        text.draw_string(tft,  4,  4, "TFT_LIB", fg=colors.WHITE, bg=colors.RED,   scale=1)
        text.draw_string(tft, 90,  4, "ILI9225", fg=colors.WHITE, bg=colors.BLUE,  scale=1)
        text.draw_centered(tft, 195, "Pi Zero 2W", fg=colors.YELLOW, bg=colors.BLACK, scale=1)
        text.draw_number(tft, 4, 210, 3.14159, fg=colors.CYAN, bg=colors.BLACK,
                         scale=1, fmt="pi={:.4f}")

        print("Done. Display showing demo. Press Ctrl+C to exit.")
        time.sleep(10)


if __name__ == "__main__":
    main()
