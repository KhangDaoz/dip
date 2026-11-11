"""Mở rộng K8 (bảng có kẻ, bộ 4) và K9 (cờ chữ đỏ) — làm nếu kịp (KeHoach.md mục 7.1)."""
import cv2
import numpy as np


def ruled_lines(binary: np.ndarray, frac: int = 20) -> tuple:
    """K8: Opening SE 1 × (W/frac) và (H/frac) × 1 để GIỮ đường kẻ (C7 · S6, S28); Closing nối chỗ đứt (C7 · S35, S38).
    Trả về (mask_ngang, mask_dọc)."""
    H, W = binary.shape
    b = (binary > 0).astype(np.uint8) * 255
    horiz = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((1, max(W // frac, 3)), np.uint8))
    vert = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((max(H // frac, 3), 1), np.uint8))
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8))
    return horiz, vert


def red_result_cells(red_text_mask: np.ndarray, cells: list, min_ratio: float = 0.05) -> list:
    """K9: ô kết quả giao với mask chữ đỏ từ ② (C4 · S37–41). Trả về chỉ số các ô có chữ đỏ."""
    out = []
    for i, c in enumerate(cells):
        x, y, w, h = c["bbox"]
        region = red_text_mask[y:y + h, x:x + w]
        if region.size and (region > 0).mean() >= min_ratio:
            out.append(i)
    return out
