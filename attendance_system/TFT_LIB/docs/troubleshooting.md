# Troubleshooting

## Display stays white / blank after init

**Cause 1 – SPI not enabled.**
```bash
ls /dev/spidev0.0   # must exist
# If not: sudo raspi-config → Interface Options → SPI → Enable → Reboot
```

**Cause 2 – Wrong wiring.**  
Double-check DC and CS pins.  A swapped DC/RST line is the most common mistake.

**Cause 3 – Insufficient power.**  
The ILI9225 + backlight draws ~60–80 mA.  Connect VCC directly to the Pi's 3.3 V
rail (Pin 1), not through a breadboard with long jumper wires.

---

## `RuntimeError: No access to /dev/mem` or GPIO permission denied

Run as root, or add your user to the `gpio` and `spi` groups:
```bash
sudo usermod -aG gpio,spi $USER
# Log out and back in, then retry
```

---

## `OSError: [Errno 22] Invalid argument` from spidev

The requested SPI speed may exceed hardware limits.  Reduce it:
```python
tft = ILI9225(dc_pin=25, spi_speed=8_000_000)  # 8 MHz safe baseline
```

---

## Colors look wrong (red/blue swapped)

The ILI9225 is initialised in **RGB** mode by default.  If colours still look
inverted (e.g. `colors.RED` looks blue), your display may be wired in BGR mode.
Change the entry-mode constants in `ili9225.py`:

```python
# In ili9225.py, line ~50:
_ENTRY_PORTRAIT  = 0x1030   # bit12=1 → BGR
_ENTRY_LANDSCAPE = 0x1038
```

---

## Very low frame rate (< 2 fps)

1. **Lower capture resolution.** In `camera_stream.py`, reduce `CAM_W`/`CAM_H`
   (e.g. 160×120 instead of 320×240).

2. **Increase SPI speed** (carefully – long wires limit max speed):
   ```python
   tft = ILI9225(dc_pin=25, spi_speed=20_000_000)  # 20 MHz
   ```

3. **Increase SPI DMA buffer** so fewer kernel round-trips are needed per frame:
   ```bash
   echo 65536 | sudo tee /sys/module/spidev/parameters/bufsiz
   ```

4. **Use scale_mode='stretch'** in `StreamDisplay` – skips the black-bar fill
   pass and is slightly faster than `'fit'`.

5. **Pre-scale the camera** to `176×220` in the `Picamera2` config so no
   software resize is needed:
   ```python
   cfg = cam.create_preview_configuration(
       main={"format": "RGB888", "size": (176, 220)}
   )
   ```

---

## `ModuleNotFoundError: No module named 'spidev'`

```bash
pip install spidev
# Or on Raspbian:
sudo apt install python3-spidev
```

---

## `ModuleNotFoundError: No module named 'RPi'`

```bash
pip install RPi.GPIO
# Or:
sudo apt install python3-rpi.gpio
```

---

## `picamera2` not found

```bash
sudo apt install python3-picamera2
```

---

## Display freezes mid-stream / SPI errors

Reduce SPI speed and shorten / improve wiring.  Use short (< 20 cm) jumper
wires and add a 10–22 Ω series resistor on CLK and MOSI if signal integrity is
an issue.

---

## Using `lgpio` instead of `RPi.GPIO`

`lgpio` is the recommended GPIO library for Pi 5 / Bookworm.  To use it, replace
the GPIO calls in `ili9225.py`:

```python
import lgpio

# In __init__:
self._gpio = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(self._gpio, self._dc)
lgpio.gpio_claim_output(self._gpio, self._rst)
lgpio.gpio_claim_output(self._gpio, self._bl)

# Replace GPIO.output(pin, HIGH/LOW) with:
lgpio.gpio_write(self._gpio, pin, 1)  # or 0

# In cleanup():
lgpio.gpiochip_close(self._gpio)
```
