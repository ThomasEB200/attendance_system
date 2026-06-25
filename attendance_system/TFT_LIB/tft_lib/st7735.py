"""
ST7735 TFT display driver for Raspberry Pi (Linux / SPI).

Supported hardware: 1.8-inch TFT, 128x160, ST7735 controller (Red PCB).
Interface:          4-wire SPI via spidev + 3 GPIO lines (DC, RST, BL).
"""

import time
import spidev
import RPi.GPIO as GPIO

from .colors import BLACK

# Rotation constants
PORTRAIT_0    = 0   # 128 wide × 160 tall  (normal)
LANDSCAPE_90  = 1   # 160 wide × 128 tall  (rotated 90° CW)
PORTRAIT_180  = 2   # 128 wide × 160 tall  (upside-down)
LANDSCAPE_270 = 3   # 160 wide × 128 tall  (rotated 270° CW)

# ST7735 commands
_SWRESET = 0x01
_SLPOUT  = 0x11
_INVOFF  = 0x20
_INVON   = 0x21
_DISPON  = 0x29
_CASET   = 0x2A
_RASET   = 0x2B
_RAMWR   = 0x2C
_MADCTL  = 0x36
_COLMOD  = 0x3A
_FRMCTR1 = 0xB1
_FRMCTR2 = 0xB2
_FRMCTR3 = 0xB3
_INVCTR  = 0xB4
_PWCTR1  = 0xC0
_PWCTR2  = 0xC1
_PWCTR3  = 0xC2
_PWCTR4  = 0xC3
_PWCTR5  = 0xC4
_VMCTR1  = 0xC5
_GMCTRP1 = 0xE0
_GMCTRN1 = 0xE1

_SPI_CHUNK = 4096   # safe per-transfer byte limit for Linux spidev


class ST7735:
    """
    Driver for the ST7735 TFT controller (1.8 inch, 128x160).
    """

    NATIVE_WIDTH  = 128
    NATIVE_HEIGHT = 160

    def __init__(
        self,
        dc_pin:    int,
        rst_pin:   int = None,
        bl_pin:    int = None,
        spi_bus:   int = 0,
        spi_dev:   int = 0,
        spi_speed: int = 16_000_000,
    ):
        self._dc  = dc_pin
        self._rst = rst_pin
        self._bl  = bl_pin

        self._width  = self.NATIVE_WIDTH
        self._height = self.NATIVE_HEIGHT

        self._init_gpio()
        self._init_spi(spi_bus, spi_dev, spi_speed)
        self._reset()
        self._init_sequence()
        self.set_rotation(PORTRAIT_0)

    def _init_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self._dc, GPIO.OUT)
        if self._rst is not None:
            GPIO.setup(self._rst, GPIO.OUT)
        if self._bl is not None:
            GPIO.setup(self._bl, GPIO.OUT)
            GPIO.output(self._bl, GPIO.HIGH)

    def _init_spi(self, bus: int, dev: int, speed: int):
        self._spi = spidev.SpiDev()
        self._spi.open(bus, dev)
        self._spi.max_speed_hz = speed
        self._spi.mode = 0

    def _reset(self):
        if self._rst is not None:
            GPIO.output(self._rst, GPIO.HIGH)
            time.sleep(0.010)
            GPIO.output(self._rst, GPIO.LOW)
            time.sleep(0.050)
            GPIO.output(self._rst, GPIO.HIGH)
            time.sleep(0.120)

    def _cmd(self, cmd: int, data: list = None):
        """Send a command byte, optionally followed by data bytes."""
        GPIO.output(self._dc, GPIO.LOW)
        self._spi.writebytes([cmd])
        if data:
            GPIO.output(self._dc, GPIO.HIGH)
            self._spi.writebytes(data)

    def _init_sequence(self):
        """ST7735 initialisation sequence."""
        self._cmd(_SWRESET)
        time.sleep(0.150)

        self._cmd(_SLPOUT)
        time.sleep(0.150)

        # Framerate
        self._cmd(_FRMCTR1, [0x01, 0x2C, 0x2D])
        self._cmd(_FRMCTR2, [0x01, 0x2C, 0x2D])
        self._cmd(_FRMCTR3, [0x01, 0x2C, 0x2D, 0x01, 0x2C, 0x2D])

        # Inversion control
        self._cmd(_INVCTR, [0x07])

        # Power control
        self._cmd(_PWCTR1, [0xA2, 0x02, 0x84])
        self._cmd(_PWCTR2, [0xC5])
        self._cmd(_PWCTR3, [0x0A, 0x00])
        self._cmd(_PWCTR4, [0x8A, 0x2A])
        self._cmd(_PWCTR5, [0x8A, 0xEE])

        # VCOM control
        self._cmd(_VMCTR1, [0x0E])

        # Inversion OFF (standard RGB)
        self._cmd(_INVOFF)

        # Color mode: 16-bit / pixel
        self._cmd(_COLMOD, [0x05])

        # Gamma
        self._cmd(_GMCTRP1, [0x02, 0x1c, 0x07, 0x12, 0x37, 0x32, 0x29, 0x2d, 0x29, 0x25, 0x2b, 0x39, 0x00, 0x01, 0x03, 0x10])
        self._cmd(_GMCTRN1, [0x03, 0x1d, 0x07, 0x06, 0x2e, 0x2c, 0x29, 0x2d, 0x2e, 0x2e, 0x37, 0x3f, 0x00, 0x00, 0x02, 0x10])

        self._cmd(_DISPON)
        time.sleep(0.100)

    def set_rotation(self, rotation: int):
        """Set the display orientation."""
        # MADCTL values for RGB color order
        # MY=0x80, MX=0x40, MV=0x20, ML=0x10, RGB=0x00
        if rotation == PORTRAIT_0:
            madctl = 0xC0
            self._width, self._height = self.NATIVE_WIDTH, self.NATIVE_HEIGHT
        elif rotation == LANDSCAPE_90:
            madctl = 0xA0
            self._width, self._height = self.NATIVE_HEIGHT, self.NATIVE_WIDTH
        elif rotation == PORTRAIT_180:
            madctl = 0x00
            self._width, self._height = self.NATIVE_WIDTH, self.NATIVE_HEIGHT
        elif rotation == LANDSCAPE_270:
            madctl = 0x60
            self._width, self._height = self.NATIVE_HEIGHT, self.NATIVE_WIDTH
        else:
            madctl = 0xC0
            self._width, self._height = self.NATIVE_WIDTH, self.NATIVE_HEIGHT

        self._cmd(_MADCTL, [madctl])

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def set_window(self, x0: int, y0: int, x1: int, y1: int):
        """Define the active pixel window."""
        self._cmd(_CASET, [0x00, x0, 0x00, x1])
        self._cmd(_RASET, [0x00, y0, 0x00, y1])

    def _begin_gram(self):
        """Send the GRAM-write command."""
        self._cmd(_RAMWR)

    def _push_bytes(self, data):
        """Stream raw bytes to GRAM."""
        GPIO.output(self._dc, GPIO.HIGH)
        view = memoryview(bytes(data))
        for i in range(0, len(view), _SPI_CHUNK):
            self._spi.writebytes2(view[i : i + _SPI_CHUNK])

    def draw_pixel(self, x: int, y: int, color: int):
        if not (0 <= x < self._width and 0 <= y < self._height):
            return
        self.set_window(x, y, x, y)
        self._begin_gram()
        self._push_bytes(bytes([(color >> 8) & 0xFF, color & 0xFF]))

    def fill_rect(self, x: int, y: int, w: int, h: int, color: int):
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
        self.fill_rect(0, 0, self._width, self._height, color)

    def push_image(self, x: int, y: int, w: int, h: int, data):
        self.set_window(x, y, x + w - 1, y + h - 1)
        self._begin_gram()
        self._push_bytes(data)

    def backlight(self, on: bool = True):
        if self._bl is not None:
            GPIO.output(self._bl, GPIO.HIGH if on else GPIO.LOW)

    def cleanup(self):
        self.backlight(False)
        self._spi.close()
        GPIO.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()
