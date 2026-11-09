"""R3 tách lớp mực theo HSV (KeHoach.md mục 6.1; C4 · S37–41)."""
import numpy as np


def ink_masks(img_bgr: np.ndarray, red_h=((0, 10), (170, 180)), blue_h=(100, 130), s_min: int = 60) -> dict:
    """Trả về {"red", "blue", "black"}: mask uint8, mực = 255."""
    raise NotImplementedError("ink_masks")
