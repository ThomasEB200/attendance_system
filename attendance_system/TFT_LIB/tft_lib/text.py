"""
Text and number rendering for the ILI9225 TFT.

Uses the built-in 5×7 bitmap font (fonts.py).  No external dependencies.

Character grid:
  - Base size:  5 px wide × 7 px tall per character.
  - Spacing:    1 px gap between characters (configurable).
  - scale=2  → 10 × 14 px per character.

Performance tip:
  Always supply a bg color when drawing text on a live stream overlay.
  When bg is given, the entire character cell is pushed as one rectangular
  SPI burst (fast).  When bg is None, only foreground pixels are written
  one-by-one (slow, but composites onto the existing image).
"""

from .ili9225 import ILI9225
from .fonts   import FONT_5X7, FONT_WIDTH, FONT_HEIGHT, FONT_FIRST_CHAR
from .colors  import WHITE, BLACK


# Horizontal gap between characters (in base pixels, scaled with scale param)
_CHAR_GAP = 1


def _char_buf(char_data: bytes, fg: int, bg: int, scale: int) -> bytes:
    """
    Build a raw RGB565 pixel buffer for one character cell.

    Returns a bytes object of size (FONT_WIDTH+gap)*scale × FONT_HEIGHT*scale × 2.
    The gap column is always filled with bg.
    """
    cw   = (FONT_WIDTH + _CHAR_GAP) * scale
    ch   = FONT_HEIGHT * scale
    buf  = bytearray(cw * ch * 2)
    fg_h = (fg >> 8) & 0xFF
    fg_l =  fg & 0xFF
    bg_h = (bg >> 8) & 0xFF
    bg_l =  bg & 0xFF

    for row in range(FONT_HEIGHT):
        for col in range(FONT_WIDTH + _CHAR_GAP):
            # The gap column (index FONT_WIDTH) is always background
            bit = 0 if col >= FONT_WIDTH else (char_data[col] >> row) & 1
            h, l = (fg_h, fg_l) if bit else (bg_h, bg_l)
            for sy in range(scale):
                for sx in range(scale):
                    i = ((row * scale + sy) * cw + col * scale + sx) * 2
                    buf[i]     = h
                    buf[i + 1] = l

    return bytes(buf)


def char_width(scale: int = 1) -> int:
    """Return the pixel width of one character cell at the given scale."""
    return (FONT_WIDTH + _CHAR_GAP) * scale


def char_height(scale: int = 1) -> int:
    """Return the pixel height of one character cell at the given scale."""
    return FONT_HEIGHT * scale


def text_width(text: str, scale: int = 1) -> int:
    """Return the total pixel width of a string at the given scale."""
    return len(text) * char_width(scale)


def draw_char(
    tft:   ILI9225,
    x:     int,
    y:     int,
    char:  str,
    fg:    int  = WHITE,
    bg:    int  = BLACK,
    scale: int  = 1,
):
    """
    Render a single ASCII character at pixel position (x, y).

    Args:
        tft:   ILI9225 instance.
        x, y:  Top-left corner for this character.
        char:  A single character string.
        fg:    Foreground RGB565 color.
        bg:    Background RGB565 color.  Pass None to skip background pixels
               (transparent overlay, slower).
        scale: Pixel multiplier (1 = 5×7 px, 2 = 10×14 px, …).
    """
    code = ord(char)
    if code < FONT_FIRST_CHAR or code > 127:
        code = FONT_FIRST_CHAR   # render unknown chars as space

    offset    = (code - FONT_FIRST_CHAR) * FONT_WIDTH
    char_data = FONT_5X7[offset : offset + FONT_WIDTH]

    if bg is not None:
        # Fast path: push full character cell as a single SPI burst
        buf = _char_buf(char_data, fg, bg, scale)
        cw  = (FONT_WIDTH + _CHAR_GAP) * scale
        ch  = FONT_HEIGHT * scale
        tft.push_image(x, y, cw, ch, buf)
    else:
        # Slow path: draw only lit pixels (transparent background)
        for row in range(FONT_HEIGHT):
            for col in range(FONT_WIDTH):
                if (char_data[col] >> row) & 1:
                    for sy in range(scale):
                        for sx in range(scale):
                            tft.draw_pixel(
                                x + col * scale + sx,
                                y + row * scale + sy,
                                fg,
                            )


def draw_string(
    tft:   ILI9225,
    x:     int,
    y:     int,
    text:  str,
    fg:    int  = WHITE,
    bg:    int  = BLACK,
    scale: int  = 1,
):
    """
    Render a string of characters left-to-right starting at (x, y).

    Characters that would overflow the right edge of the display are clipped.
    Newline characters (\n) advance to the next line.

    Args:
        tft:   ILI9225 instance.
        x, y:  Top-left origin for the first character.
        text:  String to render.
        fg:    Foreground color (RGB565).
        bg:    Background color (RGB565), or None for transparent.
        scale: Pixel multiplier.

    Returns:
        (x_end, y_end) – pixel position just after the last character,
        useful for chaining multiple draw_string() calls.
    """
    cx = x
    cy = y
    cw = char_width(scale)
    ch = char_height(scale)

    for ch_char in text:
        if ch_char == '\n':
            cx  = x
            cy += ch
            continue
        if cx + cw > tft.width:
            cx  = x
            cy += ch
        draw_char(tft, cx, cy, ch_char, fg=fg, bg=bg, scale=scale)
        cx += cw

    return cx, cy + ch


def draw_number(
    tft:    ILI9225,
    x:      int,
    y:      int,
    value:  float | int,
    fg:     int  = WHITE,
    bg:     int  = BLACK,
    scale:  int  = 1,
    fmt:    str  = "{}",
):
    """
    Render a number (int or float) at (x, y).

    Args:
        tft:   ILI9225 instance.
        x, y:  Top-left origin.
        value: The numeric value to display.
        fg:    Foreground color.
        bg:    Background color.
        scale: Pixel multiplier.
        fmt:   Python format string, e.g. "{:.1f}" for one decimal place,
               "{:05d}" for zero-padded integer.
    """
    draw_string(tft, x, y, fmt.format(value), fg=fg, bg=bg, scale=scale)


def draw_centered(
    tft:   ILI9225,
    y:     int,
    text:  str,
    fg:    int  = WHITE,
    bg:    int  = BLACK,
    scale: int  = 1,
):
    """
    Render a string horizontally centred on the display at row y.

    Args:
        tft:   ILI9225 instance.
        y:     Top pixel row for the text.
        text:  String to centre.
        fg, bg, scale: Same as draw_string().
    """
    total_w = text_width(text, scale)
    x = max((tft.width - total_w) // 2, 0)
    draw_string(tft, x, y, text, fg=fg, bg=bg, scale=scale)
