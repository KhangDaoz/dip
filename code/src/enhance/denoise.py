"""A4 khử nhiễu (ch. 3): lọc trung vị (muối tiêu, JPEG) hoặc song phương (giữ biên)."""
from __future__ import annotations

import cv2
import numpy as np


def denoise(img: np.ndarray, method: str = "median", ksize: int = 3) -> np.ndarray:
    if method == "median":
        return cv2.medianBlur(img, ksize | 1)
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=max(ksize, 5), sigmaColor=40, sigmaSpace=5)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize | 1, ksize | 1), 0)
    raise ValueError(f"method phải thuộc median/bilateral/gaussian, nhận {method}")
