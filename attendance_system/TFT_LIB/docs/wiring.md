# Wiring Guide – ILI9225 TFT on Raspberry Pi Zero 2W

## Display: KMRTM2018-SPI V.1 · 2-inch · ILI9225 · 176 × 220

---

## Pin Mapping

| TFT Pin | Function          | Pi Zero 2W BCM | Physical Pin |
|---------|-------------------|---------------|--------------|
| VCC     | Power             | 3.3 V         | Pin 1        |
| GND     | Ground            | GND           | Pin 6        |
| CLK     | SPI clock         | GPIO 11       | Pin 23       |
| SDI     | SPI MOSI (data)   | GPIO 10       | Pin 19       |
| CS      | Chip select (CE0) | GPIO 8        | Pin 24       |
| RS/DC   | Data / Command    | GPIO 25       | Pin 22       |
| RST     | Hardware reset    | GPIO 27       | Pin 13       |
| LED/BL  | Backlight         | GPIO 18       | Pin 12       |

> **VCC voltage:** Most KMRTM2018 boards have an onboard 3.3 V regulator and
> accept 5 V on VCC.  Check the silkscreen on your board.  If in doubt, use
> 3.3 V (Pi Zero 2W Pin 1).

> **LED/BL:** The backlight can also be tied directly to 3.3 V if you don't
> need software control. When connected to GPIO 18 (PWM-capable), brightness
> control via `lgpio` PWM is possible in future extensions.

---

## Wiring Diagram (text form)

```
Pi Zero 2W                     KMRTM2018 TFT
──────────────────────────────────────────────
3.3V  (pin 1)  ────────────── VCC
GND   (pin 6)  ────────────── GND
GPIO11 (pin23) ────────────── CLK
GPIO10 (pin19) ────────────── SDI (MOSI)
GPIO8  (pin24) ────────────── CS
GPIO25 (pin22) ────────────── RS  (DC)
GPIO27 (pin13) ────────────── RST
GPIO18 (pin12) ────────────── LED
```

---

## Enable SPI on Raspberry Pi

```bash
sudo raspi-config
# Interface Options → SPI → Yes → Finish → Reboot
```

Verify the SPI device node exists:
```bash
ls /dev/spidev0.*
# Should show: /dev/spidev0.0  /dev/spidev0.1
```

---

## Increase SPI Buffer Size (optional, for higher fps)

The default Linux SPI DMA buffer is 4096 bytes.  The library already handles
chunking automatically, but you can raise the limit for slightly lower overhead:

```bash
# Add to /boot/firmware/config.txt  (Bookworm) or /boot/config.txt (Bullseye)
dtparam=spi=on
dtoverlay=spi0-1cs,cs0_pin=8

# Increase buffer
echo 65536 | sudo tee /sys/module/spidev/parameters/bufsiz
```

Make it permanent by adding to `/etc/rc.local` before `exit 0`:
```bash
echo 65536 > /sys/module/spidev/parameters/bufsiz
```

---

## Alternative GPIO Pin Assignments

If pins 25 / 27 / 18 conflict with other peripherals, change them in your
script and pass the new BCM numbers to `ILI9225()`:

```python
tft = ILI9225(dc_pin=24, rst_pin=22, bl_pin=13)
```

Any free output-capable BCM GPIO can be used for DC, RST, and BL.
The SPI pins (GPIO 8, 10, 11) are fixed to the hardware SPI0 controller.
