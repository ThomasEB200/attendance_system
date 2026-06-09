# API Reference – TFT_LIB

## Module overview

| Module          | Purpose                                      |
|-----------------|----------------------------------------------|
| `ili9225`       | Hardware driver (SPI, init, pixel push)       |
| `colors`        | RGB565 constants and conversion helpers       |
| `fonts`         | 5×7 bitmap font data                         |
| `graphics`      | Lines, rectangles, circles, face boxes       |
| `text`          | Character and string rendering               |
| `stream`        | Camera frame scaling and display             |

---

## `tft_lib.ILI9225`

```python
tft = ILI9225(dc_pin, rst_pin=None, bl_pin=None,
              spi_bus=0, spi_dev=0, spi_speed=16_000_000)
```

| Parameter   | Type  | Default      | Description                          |
|-------------|-------|-------------|--------------------------------------|
| `dc_pin`    | int   | –           | BCM GPIO for Data/Command line        |
| `rst_pin`   | int   | `None`      | BCM GPIO for hardware reset           |
| `bl_pin`    | int   | `None`      | BCM GPIO for backlight                |
| `spi_bus`   | int   | `0`         | Linux SPI bus (`/dev/spidev{bus}.x`)  |
| `spi_dev`   | int   | `0`         | SPI CE/CS device number               |
| `spi_speed` | int   | `16_000_000`| SPI clock in Hz (max ~20 MHz)         |

Supports context manager (`with ILI9225(...) as tft:`).

### Methods

| Method | Description |
|--------|-------------|
| `set_rotation(r)` | Set orientation: `PORTRAIT_0`, `LANDSCAPE_90`, `PORTRAIT_180`, `LANDSCAPE_270` |
| `fill_screen(color)` | Fill entire display with an RGB565 color |
| `fill_rect(x, y, w, h, color)` | Fill a rectangle |
| `draw_pixel(x, y, color)` | Draw a single pixel |
| `push_image(x, y, w, h, data)` | Blit raw RGB565 bytes to a rectangular region |
| `set_window(x0, y0, x1, y1)` | Set active GRAM window (low-level) |
| `backlight(on)` | Turn backlight on/off |
| `cleanup()` | Release SPI and GPIO |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `width`  | int  | Logical display width  (depends on rotation) |
| `height` | int  | Logical display height (depends on rotation) |

---

## `tft_lib.colors`

### Constants (RGB565)

`BLACK`, `WHITE`, `RED`, `GREEN`, `BLUE`, `YELLOW`, `CYAN`, `MAGENTA`,
`ORANGE`, `PURPLE`, `PINK`, `GRAY`, `DARK_GRAY`, `LIGHT_GRAY`,
`NAVY`, `DARK_GREEN`, `DARK_CYAN`, `MAROON`, `OLIVE`

### Functions

```python
rgb_to_565(r, g, b)       → int   # (0-255, 0-255, 0-255) → RGB565
bgr_to_565(b, g, r)       → int   # OpenCV BGR tuple → RGB565
color_to_bytes(color)     → bytes # RGB565 int → 2 big-endian bytes
from_hex("#RRGGBB")       → int   # HTML hex string → RGB565
```

---

## `tft_lib.graphics`

All functions take `tft` (an `ILI9225` instance) as the first argument.

```python
draw_line(tft, x0, y0, x1, y1, color)
draw_hline(tft, x, y, w, color)
draw_vline(tft, x, y, h, color)
draw_rect(tft, x, y, w, h, color)           # outline only
fill_rect(tft, x, y, w, h, color)           # filled
draw_rect_thick(tft, x, y, w, h, color, thickness=2)
draw_circle(tft, cx, cy, r, color)          # outline
fill_circle(tft, cx, cy, r, color)          # filled
draw_face_box(tft, x, y, w, h, label="", box_color=GREEN, thickness=2)
draw_crosshair(tft, cx, cy, size=8, color=WHITE)
```

---

## `tft_lib.text`

All functions take `tft` as the first argument.

```python
draw_char(tft, x, y, char, fg=WHITE, bg=BLACK, scale=1)
draw_string(tft, x, y, text, fg=WHITE, bg=BLACK, scale=1) → (x_end, y_end)
draw_number(tft, x, y, value, fg=WHITE, bg=BLACK, scale=1, fmt="{}")
draw_centered(tft, y, text, fg=WHITE, bg=BLACK, scale=1)
```

**Helper functions** (no `tft` argument):
```python
char_width(scale=1)       → int   # pixels wide per character cell
char_height(scale=1)      → int   # pixels tall per character cell
text_width(text, scale=1) → int   # total pixel width of a string
```

**`bg=None`** draws only foreground pixels (transparent overlay, slower).
**`scale`** is an integer multiplier: 1 = 5×7 px, 2 = 10×14 px, 3 = 15×21 px.

---

## `tft_lib.stream`

### `StreamDisplay`

```python
sd = StreamDisplay(tft, scale_mode='fit', bg_color=0x0000)
```

| scale_mode  | Behaviour |
|-------------|-----------|
| `'fit'`     | Letterbox / pillarbox – preserves aspect ratio, black bars |
| `'fill'`    | Centre-crop to fill screen – no black bars, edges may be cut |
| `'stretch'` | Stretch to exact 176×220 – aspect ratio not preserved |
| `'none'`    | Centre-align, no scaling – crops / pads as needed |

**Methods:**

```python
sd.push_rgb(frame)          # numpy (H,W,3) uint8 RGB  or PIL Image
sd.push_bgr(frame)          # numpy (H,W,3) uint8 BGR  (OpenCV / cv2)
sd.push_pil(image)          # PIL Image (any mode)
sd.push_jpeg(jpeg_bytes)    # raw JPEG bytes
sd.push_full_rgb565(data)   # pre-converted RGB565 bytes, 176*220*2
```

### Standalone converters

```python
from tft_lib.stream import rgb888_to_rgb565_bytes, bgr888_to_rgb565_bytes, pil_to_rgb565_bytes

rgb888_to_rgb565_bytes(frame)  # numpy (H,W,3) RGB → bytes (requires numpy)
bgr888_to_rgb565_bytes(frame)  # numpy (H,W,3) BGR → bytes (requires numpy)
pil_to_rgb565_bytes(image)     # PIL Image → bytes
```
