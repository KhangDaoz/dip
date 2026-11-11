from __future__ import annotations

import cv2
import numpy as np


def background_closing(gray: np.ndarray, kernel: int = 81, median_ksize: int = 21) -> np.ndarray:
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    if median_ksize and median_ksize > 1:
        bg = cv2.medianBlur(bg, median_ksize | 1)
    return bg


def divide_background(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    out = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _flatten(gray: np.ndarray, kernel: int) -> np.ndarray:
    return divide_background(gray, background_closing(gray, kernel=kernel))


def remove_shadows(img: np.ndarray, kernel: int = 81) -> np.ndarray:
    if img.ndim == 2:
        return _flatten(img, kernel)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = _flatten(hsv[:, :, 2], kernel)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
