"""Khối B — đối chứng cổ điển: Sauvola → chiều cao chữ → giãn → thành phần liên thông (plan §5.2).

Không phân lớp → TextBox.label = None (khối D phân lớp bằng luật trên chữ OCR).
Chương: 5 (ngưỡng cục bộ), 7 (giãn, thành phần liên thông).
"""
from __future__ import annotations

import cv2
import numpy as np

from common.contracts import TextBox
from common.io import to_gray
from legacy.cleaning.binarize import sauvola
from legacy.table_extraction.geometry import char_height, words_from_binary

DEFAULT_PARAMS = {"window": 31, "k": 0.2, "a": 1.0, "v": 0.3, "open_ksize": 2, "max_h_ratio": 4.0}


def detect_classical(img: np.ndarray, window: int = 31, k: float = 0.2, a: float = 1.0, v: float = 0.3,
                     open_ksize: int = 2, max_h_ratio: float = 4.0) -> list:
    """Trả về list[TextBox] (box xyxy, label None, score 1).

    a: bề ngang SE = a·h — đủ nối khe giữa ký tự/từ trong một cụm, nhỏ hơn khoảng trống giữa các cột.
    v: bề cao SE = v·h — nối dấu tiếng Việt với thân chữ, nhỏ hơn khe giữa hai dòng.
    max_h_ratio: bỏ thành phần cao > max_h_ratio·h (khung, logo, dấu).
    Chọn a, v trên valid (E-B1).
    """
    gray = to_gray(img)
    ink = sauvola(gray, window=window, k=k)
    if open_ksize and open_ksize > 1:   # bỏ chấm nhiễu nhỏ trước khi giãn
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))
    h = char_height(ink)
    boxes = []
    for x, y, w, hh in words_from_binary(ink, h, a=a, v=v):
        if hh > max_h_ratio * h:
            continue
        boxes.append(TextBox(box=[x, y, x + w, y + hh], label=None, score=1.0))
    return boxes
