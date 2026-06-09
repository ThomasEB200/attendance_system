"""
Drawing primitives for the ILI9225 TFT (lines, rectangles, circles, face boxes).

All functions accept an ILI9225 instance as the first argument and operate
in the display's current rotation / coordinate space.

Performance notes:
  - fill_rect() on the ILI9225 object is hardware-optimised (single SPI burst).
  - draw_line() and draw_circle() call draw_pixel() per point; use sparingly
    during live streaming – draw overlays on a captured frame instead when fps
    is critical.
"""

from .ili9225 import ILI9225
from .colors import WHITE, RED, GREEN, BLACK


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

def draw_line(tft: ILI9225, x0: int, y0: int, x1: int, y1: int, color: int):
    """
    Draw an anti-alias-free line using Bresenham's algorithm.

    Args:
        tft:          ILI9225 instance.
        x0, y0:       Start point.
        x1, y1:       End point.
        color:        RGB565 color.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        tft.draw_pixel(x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0  += sx
        if e2 <  dx:
            err += dx
            y0  += sy


def draw_hline(tft: ILI9225, x: int, y: int, w: int, color: int):
    """Draw a horizontal line of width w starting at (x, y)."""
    tft.fill_rect(x, y, w, 1, color)


def draw_vline(tft: ILI9225, x: int, y: int, h: int, color: int):
    """Draw a vertical line of height h starting at (x, y)."""
    tft.fill_rect(x, y, 1, h, color)


# ---------------------------------------------------------------------------
# Rectangles
# ---------------------------------------------------------------------------

def draw_rect(tft: ILI9225, x: int, y: int, w: int, h: int, color: int):
    """
    Draw an unfilled rectangle outline.

    Args:
        x, y:  Top-left corner.
        w, h:  Width and height.
        color: RGB565 stroke color.
    """
    draw_hline(tft, x,         y,         w, color)
    draw_hline(tft, x,         y + h - 1, w, color)
    draw_vline(tft, x,         y,         h, color)
    draw_vline(tft, x + w - 1, y,         h, color)


def fill_rect(tft: ILI9225, x: int, y: int, w: int, h: int, color: int):
    """Filled rectangle – delegates directly to the hardware-optimised method."""
    tft.fill_rect(x, y, w, h, color)


def draw_rect_thick(
    tft: ILI9225,
    x: int, y: int,
    w: int, h: int,
    color: int,
    thickness: int = 2,
):
    """
    Draw a rectangle outline with multiple-pixel thickness.

    Useful for face-detection bounding boxes where a 1-pixel border is
    too thin to read clearly on a small display.
    """
    for t in range(thickness):
        draw_rect(tft, x + t, y + t, w - 2 * t, h - 2 * t, color)


# ---------------------------------------------------------------------------
# Circles
# ---------------------------------------------------------------------------

def draw_circle(tft: ILI9225, cx: int, cy: int, r: int, color: int):
    """
    Draw an unfilled circle using Bresenham's midpoint algorithm.

    Args:
        cx, cy: Centre coordinates.
        r:      Radius in pixels.
        color:  RGB565 color.
    """
    x, y, err = r, 0, 1 - r
    while x >= y:
        tft.draw_pixel(cx + x, cy + y, color)
        tft.draw_pixel(cx - x, cy + y, color)
        tft.draw_pixel(cx + x, cy - y, color)
        tft.draw_pixel(cx - x, cy - y, color)
        tft.draw_pixel(cx + y, cy + x, color)
        tft.draw_pixel(cx - y, cy + x, color)
        tft.draw_pixel(cx + y, cy - x, color)
        tft.draw_pixel(cx - y, cy - x, color)
        y += 1
        if err < 0:
            err += 2 * y + 1
        else:
            x  -= 1
            err += 2 * (y - x) + 1


def fill_circle(tft: ILI9225, cx: int, cy: int, r: int, color: int):
    """
    Draw a filled circle.

    Args:
        cx, cy: Centre coordinates.
        r:      Radius in pixels.
        color:  RGB565 fill color.
    """
    x, y, err = r, 0, 1 - r
    while x >= y:
        draw_hline(tft, cx - x, cy + y, 2 * x + 1, color)
        draw_hline(tft, cx - x, cy - y, 2 * x + 1, color)
        draw_hline(tft, cx - y, cy + x, 2 * y + 1, color)
        draw_hline(tft, cx - y, cy - x, 2 * y + 1, color)
        y += 1
        if err < 0:
            err += 2 * y + 1
        else:
            x  -= 1
            err += 2 * (y - x) + 1


# ---------------------------------------------------------------------------
# Face-detection overlay helpers
# ---------------------------------------------------------------------------

def draw_face_box(
    tft:       ILI9225,
    x:         int,
    y:         int,
    w:         int,
    h:         int,
    label:     str  = "",
    box_color: int  = GREEN,
    thickness: int  = 2,
):
    """
    Draw a face-detection bounding box with an optional label.

    The label is rendered using the built-in text module so there is no
    external font dependency.

    Args:
        tft:        ILI9225 display instance.
        x, y:       Top-left corner of the bounding box.
        w, h:       Box width and height in pixels.
        label:      String to display above the box (e.g. "Face 0.97").
        box_color:  RGB565 border color (default: green).
        thickness:  Border thickness in pixels (default: 2).
    """
    draw_rect_thick(tft, x, y, w, h, box_color, thickness)

    if label:
        # Import here to avoid circular dependency
        from . import text
        lx = max(x, 0)
        ly = max(y - 9, 0)   # one text line above the box
        text.draw_string(tft, lx, ly, label, fg=box_color, bg=BLACK, scale=1)


def draw_crosshair(tft: ILI9225, cx: int, cy: int, size: int = 8, color: int = WHITE):
    """Draw a simple crosshair at (cx, cy) – useful for targeting / calibration."""
    draw_hline(tft, cx - size, cy,        size * 2 + 1, color)
    draw_vline(tft, cx,        cy - size, size * 2 + 1, color)
