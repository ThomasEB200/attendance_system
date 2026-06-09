"""
Text and number rendering demo for the ILI9225 TFT.

Demonstrates all text functions: draw_string, draw_number, draw_centered,
and scale multipliers.

Run from the TFT_LIB directory:
    python examples/text_demo.py
"""

import sys
import time

sys.path.insert(0, "..")

from tft_lib import ILI9225
from tft_lib import colors, text

DC_PIN  = 25
RST_PIN = 27
BL_PIN  = 18


def main():
    with ILI9225(dc_pin=DC_PIN, rst_pin=RST_PIN, bl_pin=BL_PIN) as tft:
        tft.fill_screen(colors.BLACK)

        # ---- Scale sizes -----------------------------------------------
        y = 2
        text.draw_string(tft, 2, y, "Scale 1", fg=colors.WHITE, bg=colors.BLACK, scale=1)
        y += text.char_height(1) + 2
        text.draw_string(tft, 2, y, "Scale2",  fg=colors.CYAN,  bg=colors.BLACK, scale=2)
        y += text.char_height(2) + 2
        text.draw_string(tft, 2, y, "Sc3",     fg=colors.YELLOW,bg=colors.BLACK, scale=3)
        y += text.char_height(3) + 4

        # ---- Color combinations ----------------------------------------
        palette = [
            (colors.RED,     colors.BLACK, "RED"),
            (colors.GREEN,   colors.BLACK, "GREEN"),
            (colors.BLUE,    colors.WHITE, "BLUE"),
            (colors.MAGENTA, colors.BLACK, "MAGENTA"),
            (colors.ORANGE,  colors.BLACK, "ORANGE"),
        ]
        for fg, bg, label in palette:
            text.draw_string(tft, 2, y, label, fg=fg, bg=bg, scale=1)
            y += text.char_height(1) + 1

        # ---- Numbers -------------------------------------------------------
        y += 2
        text.draw_number(tft, 2, y, 42,       fg=colors.WHITE, bg=colors.BLACK, fmt="Int: {}")
        y += text.char_height(1) + 2
        text.draw_number(tft, 2, y, 3.14159,  fg=colors.CYAN,  bg=colors.BLACK, fmt="Pi:{:.4f}")
        y += text.char_height(1) + 2
        text.draw_number(tft, 2, y, 7,        fg=colors.YELLOW,bg=colors.BLACK, fmt="Hex:0x{:02X}")

        # ---- Centered text ------------------------------------------------
        text.draw_centered(tft, 210, "CENTERED", fg=colors.YELLOW, bg=colors.BLACK, scale=1)

        print("Text demo complete. Waiting 10 s …")
        time.sleep(10)


if __name__ == "__main__":
    main()
