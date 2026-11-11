"""A3 khử bóng: ước lượng nền bằng phép đóng (ch. 7) rồi chia nền (toán tử điểm, ch. 3).

Ảnh ≈ chiếu sáng × phản xạ: nét chữ tối và hẹp hơn SE nên bị phép đóng lấp đi, phần còn lại xấp xỉ ánh sáng
chiếu lên giấy; chia cho nền thì bóng biến mất. Ảnh BGR làm trên kênh V (HSV) để giữ màu.
"""
from __future__ import annotations

import cv2
import numpy as np


def background_closing(gray: np.ndarray, kernel: int = 81, median_ksize: int = 21) -> np.ndarray:
    """Closing ảnh xám = lọc Max rồi Min (C3 · S74; C7 · S35, S38), sau đó trung vị để làm trơn."""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    if median_ksize and median_ksize > 1:
        bg = cv2.medianBlur(bg, median_ksize | 1)
    return bg


def divide_background(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Toán tử điểm I_clean = min(255, I / B̂ · 255) (C3 · S10–11)."""
    out = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _flatten(gray: np.ndarray, kernel: int) -> np.ndarray:
    return divide_background(gray, background_closing(gray, kernel=kernel))


def remove_shadows(img: np.ndarray, kernel: int = 81) -> np.ndarray:
    """SE vuông `kernel` (≈ 4× chiều cao chữ VAIPE 15–25 px). Ảnh xám hoặc BGR."""
    if img.ndim == 2:
        return _flatten(img, kernel)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = _flatten(hsv[:, :, 2], kernel)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
