"""Cắt vùng chữ cho OCR: đệm `pad` px, phóng `scale` lần nội suy bậc ba (ch. 2), tùy chọn nhị phân (ch. 5)."""
from __future__ import annotations

import cv2
import numpy as np

from common.io import to_gray


def crop_box(img: np.ndarray, box, pad: int = 2, scale: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
    x1, y1 = min(w, int(box[2]) + pad), min(h, int(box[3]) + pad)
    crop = img[y0:y1, x0:x1]
    if crop.size and scale != 1.0:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                          interpolation=cv2.INTER_CUBIC)
    return crop


def prepare(crop: np.ndarray, mode: str = "gray") -> np.ndarray:
    """mode: color (giữ nguyên) · gray · otsu · sauvola · clahe — dùng cho C1 và E-C2. Nhị phân trả chữ đen nền trắng."""
    if mode == "color":
        return crop
    g = to_gray(crop)
    if mode == "gray":
        return g
    if mode == "otsu":
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return b
    if mode == "sauvola":
        from legacy.cleaning.binarize import sauvola
        return 255 - sauvola(g, window=31, k=0.2)
    if mode == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
    raise ValueError(f"mode không hợp lệ: {mode}")
