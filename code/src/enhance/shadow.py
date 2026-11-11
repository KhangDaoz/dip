"""A3 khử bóng: ước lượng nền bằng phép đóng (ch. 7) rồi chia nền (toán tử điểm, ch. 3).

Bọc `legacy/cleaning/background.py`. Làm trên kênh V (HSV) để giữ màu cho ảnh BGR.
"""
from __future__ import annotations

import cv2
import numpy as np

from legacy.cleaning.background import background_closing, remove_shadow


def _flatten(gray: np.ndarray, kernel: int) -> np.ndarray:
    return remove_shadow(gray, background_closing(gray, kernel=kernel, median_ksize=21))


def remove_shadows(img: np.ndarray, kernel: int = 81) -> np.ndarray:
    """SE vuông `kernel` (≈ 4× chiều cao chữ VAIPE 15–25 px). Ảnh xám hoặc BGR."""
    if img.ndim == 2:
        return _flatten(img, kernel)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = _flatten(hsv[:, :, 2], kernel)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
