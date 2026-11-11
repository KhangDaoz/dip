"""Khối B — đối chứng cổ điển: tìm vùng chữ bằng nhị phân hóa + hình thái học (plan §5.2). Không phân lớp.

    ảnh ─► xám ─► Sauvola (ch. 5) ─► xóa đường kẻ dài (mở với SE đường thẳng, ch. 7)
        ─► h = trung vị chiều cao thành phần liên thông ─► giãn SE (v·h) × (a·h) nối ký tự thành cụm (ch. 7)
        ─► thành phần liên thông → box, lọc theo kích thước

`a` đủ lớn để nối khe giữa các chữ trong một cụm, đủ nhỏ để không nối qua khoảng trống giữa hai cột
("1) RENAPRIL 5MG" và "SL: 28 Viên" cùng hàng nhưng là 2 vùng). Chọn a, v trên VALID (experiments/e_b_detection.py).

    from detect.classical import detect_classical
    boxes = detect_classical(img_bgr)     # list[TextBox], label=None, score=1
"""
from __future__ import annotations

import cv2
import numpy as np

from common.contracts import TextBox
from common.io import to_gray
from enhance.binarize import sauvola

# Chọn trên valid (results/e_b/b0_classical_grid_valid.csv) — xem experiments/e_b_detection.py
A, V = 2.5, 0.2       # F1 0,810 trên 60 ảnh valid (lưới a 0,5–4,0 × v 0–0,5 × có/không xóa kẻ)
LINES = False         # bật xóa đường kẻ cho F1 thấp hơn trên valid (chưa phân tích nguyên nhân)
WINDOW, K = 31, 0.2


def remove_lines(ink: np.ndarray, min_len_ratio: float = 0.08) -> np.ndarray:
    """Xóa đường kẻ ngang / dọc dài: mở với SE đường thẳng dài ≥ min_len_ratio × cạnh ảnh rồi trừ đi (ch. 7)."""
    h, w = ink.shape
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * min_len_ratio)), 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * min_len_ratio)))))
    return cv2.subtract(ink, cv2.bitwise_or(horiz, vert))


def char_height(ink: np.ndarray, min_h: int = 5, max_ratio: float = 0.1) -> float:
    """h = trung vị chiều cao thành phần liên thông, bỏ nhiễu nhỏ và khối quá lớn."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    hs = hs[(hs >= min_h) & (hs <= max_ratio * ink.shape[0])]
    return float(np.median(hs)) if len(hs) else 15.0


def detect_classical(img: np.ndarray, a: float = A, v: float = V, window: int = WINDOW, k: float = K,
                     lines: bool = LINES, min_h_ratio: float = 0.5, max_h_ratio: float = 3.0) -> list:
    """Ảnh BGR/xám → list[TextBox] (xyxy, label None, score 1), sắp trên → dưới, trái → phải."""
    ink = sauvola(to_gray(img), window=window, k=k)
    if lines:
        ink = remove_lines(ink)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # bỏ chấm nhiễu 1 px
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
