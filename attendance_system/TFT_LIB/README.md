# TFT_LIB

A clean Python library for driving the **ILI9225** 2-inch TFT display
(KMRTM2018-SPI V.1, 176 × 220 pixels) from a **Raspberry Pi Zero 2W** running Linux.

Designed for real-time camera streaming and face-recognition overlays.

---

## Features

| Feature | Detail |
|---------|--------|
| Hardware driver | ILI9225, 4-wire SPI, up to 20 MHz |
| Color format | RGB565 (16-bit) – correct colors from picamera2 / OpenCV |
| Streaming | Push numpy / PIL / JPEG frames with auto-scaling |
| Scale modes | `fit`, `fill`, `stretch`, `none` |
| Text | Built-in 5×7 bitmap font, scale multiplier, transparent bg |
| Graphics | Lines, filled/outline rectangles, circles, face-box overlay |
| Rotation | 0 °, 90 °, 180 °, 270 ° |
| Dependencies | `spidev`, `RPi.GPIO`, `Pillow`, `numpy` (optional but recommended) |

---

## Directory Structure

```
TFT_LIB/
├── README.md                ← this file
├── requirements.txt         ← pip dependencies
├── tft_lib/
│   ├── __init__.py          ← public API imports
│   ├── ili9225.py           ← hardware driver (SPI + GPIO)
│   ├── colors.py            ← RGB565 constants & conversion
│   ├── fonts.py             ← 5×7 bitmap font data (ASCII 32-127)
│   ├── graphics.py          ← lines, rectangles, circles, face boxes
│   ├── text.py              ← character & string rendering
│   └── stream.py            ← camera frame → TFT display
├── examples/
│   ├── basic_display.py     ← colors, shapes, text demo
│   ├── camera_stream.py     ← live picamera2 stream
│   ├── face_overlay.py      ← face_recognition bounding boxes
│   └── text_demo.py         ← all text functions demo
└── docs/
    ├── wiring.md            ← pin connections, SPI setup
    ├── api_reference.md     ← full function reference
    └── troubleshooting.md   ← common issues and fixes
```

---

## Quick Start

### 1. Enable SPI

```bash
sudo raspi-config
# Interface Options → SPI → Enable → Finish → Reboot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Wire the display

See [docs/wiring.md](docs/wiring.md) for the full pin table.  
Default wiring (BCM numbers):

| TFT | Pi GPIO |
|-----|---------|
| CLK | 11 |
| SDI | 10 |
| CS  | 8  |
| DC  | 25 |
| RST | 27 |
| LED | 18 |

### 4. Run the basic demo

```bash
cd TFT_LIB
python examples/basic_display.py
```

---

## Usage Examples

### Clear screen and draw text

```python
from tft_lib import ILI9225
from tft_lib import colors, text

tft = ILI9225(dc_pin=25, rst_pin=27, bl_pin=18)
tft.fill_screen(colors.BLACK)
text.draw_string(tft, 10, 10, "Hello!", fg=colors.WHITE, bg=colors.BLACK)
text.draw_number(tft, 10, 25, 98.6, fg=colors.CYAN, bg=colors.BLACK, fmt="{:.1f} °C")
```

### Stream a picamera2 frame

```python
from picamera2 import Picamera2
from tft_lib import ILI9225
from tft_lib.stream import StreamDisplay

tft = ILI9225(dc_pin=25, rst_pin=27, bl_pin=18)
sd  = StreamDisplay(tft, scale_mode='fit')

cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={"format": "RGB888", "size": (320, 240)}
))
cam.start()

while True:
    frame = cam.capture_array()   # numpy (H, W, 3) RGB
    sd.push_rgb(frame)
```

### Stream from an OpenCV capture (BGR)

```python
import cv2
from tft_lib import ILI9225
from tft_lib.stream import StreamDisplay

tft = ILI9225(dc_pin=25, rst_pin=27, bl_pin=18)
sd  = StreamDisplay(tft, scale_mode='fill')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()       # numpy (H, W, 3) BGR
    if ret:
        sd.push_bgr(frame)
```

### Draw a face-detection bounding box

```python
from tft_lib import graphics, colors

# box = (x, y, width, height) in TFT pixel space
graphics.draw_face_box(tft, x=30, y=40, w=80, h=90,
                        label="Face 0.97",
                        box_color=colors.GREEN,
                        thickness=2)
```

### Draw shapes

```python
from tft_lib import graphics, colors

graphics.fill_rect(tft,  0,  0, 176, 10, colors.NAVY)
graphics.draw_rect(tft, 10, 20,  80, 60, colors.WHITE)
graphics.fill_circle(tft, 88, 130, 30, colors.RED)
graphics.draw_line(tft, 0, 0, 175, 219, colors.YELLOW)
```

---

## Color Format

All color values are **16-bit RGB565** integers.

```python
from tft_lib.colors import rgb_to_565, bgr_to_565, from_hex

orange = rgb_to_565(255, 165, 0)
blue   = bgr_to_565(255, 0, 0)    # OpenCV BGR tuple
teal   = from_hex('#008080')
```

---

## Performance Tips

1. **Use numpy** – the RGB888→RGB565 conversion is vectorised; without numpy
   it falls back to a slow Python loop.

2. **Capture at display resolution** to skip the software resize step:
   ```python
   cam.create_preview_configuration(main={"size": (176, 220)})
   ```

3. **Increase the SPI DMA buffer** to reduce kernel overhead:
   ```bash
   echo 65536 | sudo tee /sys/module/spidev/parameters/bufsiz
   ```

4. **Run face detection every N frames** (see `examples/face_overlay.py`)
   so detection latency does not block the display loop.

---

## Documentation

| Document | Content |
|----------|---------|
| [docs/wiring.md](docs/wiring.md) | Full pin table, SPI setup, buffer tuning |
| [docs/api_reference.md](docs/api_reference.md) | All functions and parameters |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common errors and fixes |
