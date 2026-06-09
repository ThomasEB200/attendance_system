import cv2
import numpy as np
from PIL import Image, ExifTags


def load_image_bgr(path: str) -> np.ndarray:
    """
    Load any image as a BGR numpy array, applying EXIF rotation.
    Handles portrait phone photos that are physically stored rotated.
    """
    pil_img = Image.open(path).convert("RGB")
    try:
        exif = pil_img._getexif()
        if exif:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    if value == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif value == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif value == 8:
                        pil_img = pil_img.rotate(90, expand=True)
                    break
    except Exception:
        pass
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
