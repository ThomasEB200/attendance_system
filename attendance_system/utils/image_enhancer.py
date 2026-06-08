import cv2
import numpy as np


class ImageEnhancer:
    """
    Lightweight frame enhancement to close the gap between webcam and phone camera quality.

    Pipeline:
      1. [optional] Bilateral denoise  — removes sensor noise while preserving edges
      2. CLAHE on L channel (LAB)      — local contrast boost, handles uneven lighting
      3. [optional] Unsharp mask       — sharpening, phone cameras apply this by default
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: int = 8,
        sharpen_strength: float = 0.6,
        denoise: bool = False,
    ):
        self._sharpen = sharpen_strength
        self._denoise = denoise
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid, clahe_tile_grid),
        )

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Enhance a BGR frame. Returns a new array."""
        if self._denoise:
            # Edge-preserving denoise — use when camera has heavy visible noise
            frame = cv2.bilateralFilter(frame, 5, 75, 75)

        # CLAHE on L channel — equalize contrast locally without blowing out colors
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        if self._sharpen > 0:
            # Unsharp mask: result = original*(1+s) - blurred*s
            blurred = cv2.GaussianBlur(frame, (0, 0), 1.5)
            frame = cv2.addWeighted(frame, 1.0 + self._sharpen, blurred, -self._sharpen, 0)

        return frame
