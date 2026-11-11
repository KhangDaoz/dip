from __future__ import annotations

import cv2
import numpy as np

from common.contracts import TextBox
from common.io import to_gray
from enhance.binarize import sauvola

A, V = 2.5, 0.2
LINES = False
WINDOW, K = 31, 0.2


def remove_lines(ink: np.ndarray, min_len_ratio: float = 0.08) -> np.ndarray:
    h, w = ink.shape
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * min_len_ratio)), 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * min_len_ratio)))))
    return cv2.subtract(ink, cv2.bitwise_or(horiz, vert))


def char_height(ink: np.ndarray, min_h: int = 5, max_ratio: float = 0.1) -> float:
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    hs = hs[(hs >= min_h) & (hs <= max_ratio * ink.shape[0])]
    return float(np.median(hs)) if len(hs) else 15.0


def detect_classical(img: np.ndarray, a: float = A, v: float = V, window: int = WINDOW, k: float = K,
                     lines: bool = LINES, min_h_ratio: float = 0.5, max_h_ratio: float = 3.0) -> list:
    ink = sauvola(to_gray(img), window=window, k=k)
    if lines:
        ink = remove_lines(ink)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    h = char_height(ink)
    se = np.ones((max(1, int(round(v * h))), max(1, int(round(a * h)))), np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, se), connectivity=8)
    out = []
    for x, y, w, hh, _ in stats[1:]:
        if not (min_h_ratio * h <= hh <= max_h_ratio * h) or w < h:
            continue
        out.append(TextBox(box=[int(x), int(y), int(x + w), int(y + hh)], label=None, score=1.0))
    out.sort(key=lambda t: (t.box[1], t.box[0]))
    return out
