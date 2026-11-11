from __future__ import annotations

import cv2
import numpy as np


def _local_stats(gray: np.ndarray, window: int):
    g = gray.astype(np.float64)
    w = (int(window) | 1, int(window) | 1)
    m = cv2.boxFilter(g, -1, w, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(g * g, -1, w, borderType=cv2.BORDER_REFLECT)
    return m, np.sqrt(np.maximum(m2 - m * m, 0))


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2, r: float = 128) -> np.ndarray:
    m, s = _local_stats(gray, window)
    return np.where(gray.astype(np.float64) <= m * (1 + k * (s / r - 1)), 255, 0).astype(np.uint8)


def otsu(gray: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b
