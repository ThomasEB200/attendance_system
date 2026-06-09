"""
ILI9225 TFT display driver for Raspberry Pi (Linux / SPI).

Supported hardware: KMRTM2018-SPI V.1 2-inch TFT, 176x220, ILI9225 controller.
Interface:          4-wire SPI via spidev + 3 GPIO lines (DC, RST, BL).

Default Pi Zero 2W wiring (BCM numbering):
  TFT VCC  → 3.3 V  (or 5 V – check your board's regulator)
  TFT GND  → GND
  TFT CLK  → GPIO 11  (SPI0 SCLK)
  TFT SDI  → GPIO 10  (SPI0 MOSI)
  TFT CS   → GPIO  8  (SPI0 CE0 – managed by spidev)
  TFT DC   → GPIO 25  (configurable via dc_pin)
  TFT RST  → GPIO 27  (configurable via rst_pin, optional)
  TFT LED  → GPIO 18  (configurable via bl_pin, or tie to 3.3 V)

Enable SPI before use:
  sudo raspi-config  →  Interface Options  →  SPI  →  Enable
"""

import time
import spidev
import RPi.GPIO as GPIO

from .colors import BLACK

# ---------------------------------------------------------------------------
# Rotation constants (use these instead of raw integers)
# ---------------------------------------------------------------------------
PORTRAIT_0    = 0   # 176 wide × 220 tall  (normal)
LANDSCAPE_90  = 1   # 220 wide × 176 tall  (rotated 90° CW)
PORTRAIT_180  = 2   # 176 wide × 220 tall  (upside-down)
LANDSCAPE_270 = 3   # 220 wide × 176 tall  (rotated 270° CW)

# ---------------------------------------------------------------------------
# ILI9225 register map
# ---------------------------------------------------------------------------
_R = {
    'DRIVER_OUTPUT':   0x01,
    'AC_DRIVING':      0x02,
    'ENTRY_MODE':      0x03,
    'DISP_CTRL1':      0x07,
    'BLANK_PERIOD':    0x08,
    'FRAME_CYCLE':     0x0B,
    'INTERFACE':       0x0C,
    'OSC_CTRL':        0x0F,
    'PWR1':            0x10,
    'PWR2':            0x11,
    'PWR3':            0x12,
    'PWR4':            0x13,
    'PWR5':            0x14,
    'VCI_RECYCLING':   0x15,
    'RAM_ADDR_X':      0x20,  # Horizontal GRAM address
    'RAM_ADDR_Y':      0x21,  # Vertical GRAM address
    'GRAM_DATA':       0x22,  # Write pixel data to GRAM
    'GATE_SCAN':       0x30,
    'VSCROLL1':        0x31,
    'VSCROLL2':        0x32,
    'VSCROLL3':        0x33,
    'PARTIAL1':        0x34,
    'PARTIAL2':        0x35,
    'H_WIN_END':       0x36,  # Horizontal window end   address
    'H_WIN_START':     0x37,  # Horizontal window start address
    'V_WIN_END':       0x38,  # Vertical   window end   address
    'V_WIN_START':     0x39,  # Vertical   window start address
    'GAMMA1':  0x50, 'GAMMA2':  0x51, 'GAMMA3':  0x52, 'GAMMA4':  0x53,
    'GAMMA5':  0x54, 'GAMMA6':  0x55, 'GAMMA7':  0x56, 'GAMMA8':  0x57,
    'GAMMA9':  0x58, 'GAMMA10': 0x59,
}

# Entry-mode values: bit 12 = 0 → RGB pixel order (matches camera output)
_ENTRY_PORTRAIT   = 0x0030   # AM=0, ID=11
_ENTRY_LANDSCAPE  = 0x0038   # AM=1, ID=11

# Rotation → (DRIVER_OUTPUT value, entry mode, swap w/h)
_ROTATION_CFG = {
    PORTRAIT_0:    (0x011C, _ENTRY_PORTRAIT,  False),
    LANDSCAPE_90:  (0x001C, _ENTRY_LANDSCAPE, True),
    PORTRAIT_180:  (0x021C, _ENTRY_PORTRAIT,  False),
    LANDSCAPE_270: (0x031C, _ENTRY_LANDSCAPE, True),
}

_SPI_CHUNK = 4096   # safe per-transfer byte limit for Linux spidev


class ILI9225:
    """
    Driver for the ILI9225 TFT controller (KMRTM2018-SPI V.1, 2 inch, 176×220).

    Typical usage:
        tft = ILI9225(dc_pin=25, rst_pin=27, bl_pin=18)
        tft.fill_screen(0x0000)          # clear to black
        tft.push_image(0, 0, 176, 220, raw_rgb565_bytes)
    """

    # Physical pixel dimensions (portrait orientation)
    NATIVE_WIDTH  = 176
    NATIVE_HEIGHT = 220

    def __init__(
        self,
        dc_pin:    int,
        rst_pin:   int = None,
        bl_pin:    int = None,
        spi_bus:   int = 0,
        spi_dev:   int = 0,
        spi_speed: int = 16_000_000,
    ):
        """
        Args:
            dc_pin:    BCM GPIO number for Data/Command (RS) line. Required.
            rst_pin:   BCM GPIO number for hardware reset. Optional – if None,
                       only a software delay is used on startup.
            bl_pin:    BCM GPIO number for backlight. Optional – if None, wire
                       the LED pin directly to 3.3 V for always-on backlight.
            spi_bus:   Linux SPI bus index  (default 0 → /dev/spidev0.x).
            spi_dev:   Linux SPI device/CE# (default 0 → CE0 / GPIO 8).
            spi_speed: SPI clock in Hz. ILI9225 supports up to ~20 MHz.
                       16 MHz is a safe default for Pi Zero 2W wiring.
        """
        self._dc  = dc_pin
        self._rst = rst_pin
        self._bl  = bl_pin

        # Logical dimensions change with rotation
        self._width  = self.NATIVE_WIDTH
        self._height = self.NATIVE_HEIGHT

        self._init_gpio()
        self._init_spi(spi_bus, spi_dev, spi_speed)
        self._reset()
        self._init_sequence()
        self.set_rotation(PORTRAIT_0)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _init_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self._dc, GPIO.OUT)
        if self._rst is not None:
            GPIO.setup(self._rst, GPIO.OUT)
        if self._bl is not None:
            GPIO.setup(self._bl, GPIO.OUT)
            GPIO.output(self._bl, GPIO.HIGH)   # backlight on by default

    def _init_spi(self, bus: int, dev: int, speed: int):
        self._spi = spidev.SpiDev()
        self._spi.open(bus, dev)
        self._spi.max_speed_hz = speed
        self._spi.mode = 0          # ILI9225: CPOL=0, CPHA=0

    def _reset(self):
        """Toggle the RST line to hardware-reset the display."""
        if self._rst is not None:
            GPIO.output(self._rst, GPIO.HIGH)
            time.sleep(0.010)
            GPIO.output(self._rst, GPIO.LOW)
            time.sleep(0.050)
            GPIO.output(self._rst, GPIO.HIGH)
            time.sleep(0.150)
        else:
            time.sleep(0.150)   # allow internal power-on reset

    # ------------------------------------------------------------------
    # Low-level SPI primitives
    # ------------------------------------------------------------------

    def _cmd(self, reg: int):
        """Send one register-address byte (DC=0)."""
        GPIO.output(self._dc, GPIO.LOW)
        self._spi.writebytes([reg])

    def _dat(self, hi: int, lo: int):
        """Send two data bytes (DC=1)."""
        GPIO.output(self._dc, GPIO.HIGH)
        self._spi.writebytes([hi, lo])

    def _reg(self, reg: int, value: int):
        """Write a 16-bit value to an ILI9225 register (addr + data)."""
        self._cmd(reg)
        self._dat((value >> 8) & 0xFF, value & 0xFF)

    # ------------------------------------------------------------------
    # Power-on initialization  (matches ILI9225_Init.h from TFT_eSPI)
    # ------------------------------------------------------------------

    def _init_sequence(self):
        """
        Full ILI9225 initialization sequence.
        Timing and values are taken directly from the TFT_eSPI C++ driver.
        """
        r = _R

        # ---- Step 1: reset all power registers -------------------------
        for reg in ('PWR1', 'PWR2', 'PWR3', 'PWR4', 'PWR5'):
            self._reg(r[reg], 0x0000)
        time.sleep(0.040)

        # ---- Step 2: power-on sequence ---------------------------------
        self._reg(r['PWR2'], 0x0018)
        self._reg(r['PWR3'], 0x6121)
        self._reg(r['PWR4'], 0x006F)
        self._reg(r['PWR5'], 0x495F)
        self._reg(r['PWR1'], 0x0800)
        time.sleep(0.010)
        self._reg(r['PWR2'], 0x103B)
        time.sleep(0.050)

        # ---- Step 3: display / panel setup -----------------------------
        self._reg(r['AC_DRIVING'],   0x0100)
        self._reg(r['DISP_CTRL1'],   0x0000)
        self._reg(r['BLANK_PERIOD'], 0x0808)
        self._reg(r['FRAME_CYCLE'],  0x1100)
        self._reg(r['INTERFACE'],    0x0000)
        self._reg(r['OSC_CTRL'],     0x0D01)
        self._reg(r['VCI_RECYCLING'],0x0020)
        self._reg(r['RAM_ADDR_X'],   0x0000)
        self._reg(r['RAM_ADDR_Y'],   0x0000)

        # ---- Step 4: scroll / window defaults --------------------------
        self._reg(r['GATE_SCAN'],    0x0000)
        self._reg(r['VSCROLL1'],     0x00DB)   # 219
        self._reg(r['VSCROLL2'],     0x0000)
        self._reg(r['VSCROLL3'],     0x0000)
        self._reg(r['PARTIAL1'],     0x00DB)
        self._reg(r['PARTIAL2'],     0x0000)
        self._reg(r['H_WIN_END'],    0x00AF)   # 175
        self._reg(r['H_WIN_START'],  0x0000)
        self._reg(r['V_WIN_END'],    0x00DB)   # 219
        self._reg(r['V_WIN_START'],  0x0000)

        # ---- Step 5: gamma curve ---------------------------------------
        self._reg(r['GAMMA1'],  0x0000)
        self._reg(r['GAMMA2'],  0x0808)
        self._reg(r['GAMMA3'],  0x080A)
        self._reg(r['GAMMA4'],  0x000A)
        self._reg(r['GAMMA5'],  0x0A08)
        self._reg(r['GAMMA6'],  0x0808)
        self._reg(r['GAMMA7'],  0x0000)
        self._reg(r['GAMMA8'],  0x0A00)
        self._reg(r['GAMMA9'],  0x0710)
        self._reg(r['GAMMA10'], 0x0710)

        # ---- Step 6: turn on display -----------------------------------
        self._reg(r['DISP_CTRL1'], 0x0012)
        time.sleep(0.050)
        self._reg(r['DISP_CTRL1'], 0x1017)

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def set_rotation(self, rotation: int):
        """
        Set the display orientation.

        Args:
            rotation: One of PORTRAIT_0, LANDSCAPE_90, PORTRAIT_180,
                      LANDSCAPE_270 (from this module or tft_lib).
        """
        rotation = rotation % 4
        drv_out, entry, swap = _ROTATION_CFG[rotation]
        self._reg(_R['DRIVER_OUTPUT'], drv_out)
        self._reg(_R['ENTRY_MODE'],    entry)
        if swap:
            self._width, self._height = self.NATIVE_HEIGHT, self.NATIVE_WIDTH
        else:
            self._width, self._height = self.NATIVE_WIDTH, self.NATIVE_HEIGHT

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Current logical width in pixels (depends on rotation)."""
        return self._width

    @property
    def height(self) -> int:
        """Current logical height in pixels (depends on rotation)."""
        return self._height

    # ------------------------------------------------------------------
    # Window / GRAM addressing
    # ------------------------------------------------------------------

    def set_window(self, x0: int, y0: int, x1: int, y1: int):
        """
        Define the active pixel window for subsequent push_pixels() calls.

        The ILI9225 auto-increments the RAM pointer within this window
        after each pixel write, so you only need to call set_window() once
        before pushing a rectangle of data.

        Args:
            x0, y0: Top-left corner (inclusive).
            x1, y1: Bottom-right corner (inclusive).
        """
        r = _R
        self._reg(r['H_WIN_END'],   x1)
        self._reg(r['H_WIN_START'], x0)
        self._reg(r['V_WIN_END'],   y1)
        self._reg(r['V_WIN_START'], y0)
        self._reg(r['RAM_ADDR_X'],  x0)
        self._reg(r['RAM_ADDR_Y'],  y0)

    def _begin_gram(self):
        """Send the GRAM-write command; follow immediately with pixel bytes (DC=1)."""
        self._cmd(_R['GRAM_DATA'])

    def _push_bytes(self, data):
        """
        Stream raw bytes to GRAM (DC must already be set high by _begin_gram).
        Automatically chunks to _SPI_CHUNK bytes to stay within kernel limits.
        """
        GPIO.output(self._dc, GPIO.HIGH)
        view = memoryview(bytes(data))
        for i in range(0, len(view), _SPI_CHUNK):
            self._spi.writebytes2(view[i : i + _SPI_CHUNK])

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def draw_pixel(self, x: int, y: int, color: int):
        """Draw a single pixel at (x, y) with the given RGB565 color."""
        if not (0 <= x < self._width and 0 <= y < self._height):
            return
        self.set_window(x, y, x, y)
        self._begin_gram()
        self._push_bytes(bytes([(color >> 8) & 0xFF, color & 0xFF]))

    def fill_rect(self, x: int, y: int, w: int, h: int, color: int):
        """
        Fill an axis-aligned rectangle with a solid RGB565 color.

        Args:
            x, y: Top-left corner.
            w, h: Width and height in pixels.
            color: RGB565 color value.
        """
        if w <= 0 or h <= 0:
            return
        x  = max(x, 0)
        y  = max(y, 0)
        x1 = min(x + w - 1, self._width  - 1)
        y1 = min(y + h - 1, self._height - 1)
        pixel_count = (x1 - x + 1) * (y1 - y + 1)
        hi = (color >> 8) & 0xFF
        lo =  color & 0xFF
        self.set_window(x, y, x1, y1)
        self._begin_gram()
        self._push_bytes(bytes([hi, lo] * pixel_count))

    def fill_screen(self, color: int = BLACK):
        """Fill the entire display with a solid color (default: black)."""
        self.fill_rect(0, 0, self._width, self._height, color)

    def push_image(self, x: int, y: int, w: int, h: int, data):
        """
        Blit a raw RGB565 image buffer onto the display.

        Args:
            x, y:  Top-left destination corner.
            w, h:  Image dimensions in pixels.
            data:  bytes / bytearray / memoryview of length w*h*2.
                   Pixels are stored row-first, big-endian RGB565.
        """
        self.set_window(x, y, x + w - 1, y + h - 1)
        self._begin_gram()
        self._push_bytes(data)

    # ------------------------------------------------------------------
    # Backlight control
    # ------------------------------------------------------------------

    def backlight(self, on: bool = True):
        """Turn the backlight on (True) or off (False), if bl_pin was given."""
        if self._bl is not None:
            GPIO.output(self._bl, GPIO.HIGH if on else GPIO.LOW)

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """
        Release SPI bus and GPIO resources.
        Call this at program exit (or use a try/finally block).
        """
        self.backlight(False)
        self._spi.close()
        GPIO.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()
