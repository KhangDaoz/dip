"""K6 hậu xử lý (KeHoach.md mục 7.1; C7 · S28, S34; QĐ · S7)."""
import cv2
import numpy as np


def opening_before_projection(binary: np.ndarray, ksize: int = 2) -> np.ndarray:
    """Opening (C7 · S28, S34) bỏ chấm nhiễu trước khi chiếu, tránh tạo dòng / cột giả."""
    if ksize <= 1:
        return binary.copy()
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((ksize, ksize), np.uint8))


def rule_lines(binary: np.ndarray, frac: int = 20) -> np.ndarray:
    """Mask đường kẻ bảng: Opening với SE đường thẳng 1×(W/frac) và (H/frac)×1 (C7 · S6, S28).

    Chỉ đoạn thẳng dài hơn SE mới "fit" SE nên còn lại sau Opening; nét chữ (ngắn hơn nhiều) bị xóa.
    """
    H, W = binary.shape[:2]
    b = (binary > 0).astype(np.uint8) * 255
    hor = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((1, max(3, W // frac)), np.uint8))
    ver = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((max(3, H // frac), 1), np.uint8))
    return cv2.bitwise_or(hor, ver)


def remove_rule_lines(binary: np.ndarray, frac: int = 20) -> np.ndarray:
    """Xóa đường kẻ trước khi chiếu: kẻ dọc làm P(y) dòng nào cũng có mực (mất thung lũng giữa các dòng),
    kẻ ngang làm Q(x) cột nào cũng có mực. Mask kẻ được Dilation 3×3 để xóa cả viền răng cưa."""
    lines = cv2.dilate(rule_lines(binary, frac), np.ones((3, 3), np.uint8))
    out = binary.copy()
    out[lines > 0] = 0
    return out


def drop_thin_rows(rows: list, words: list, h: float, min_ratio: float = 0.5) -> list:
    """Bỏ dòng cao < min_ratio·h."""
    keep = []
    for r in rows:
        top = min(words[i][1] for i in r)
        bot = max(words[i][1] + words[i][3] for i in r)
        if bot - top >= min_ratio * h:
            keep.append(r)
    return keep


def merge_narrow_columns(columns: list, h: float) -> list:
    """Gộp cột hẹp hơn h vào cột kề gần hơn."""
    cols = [list(c) for c in sorted(columns)]
    changed = True
    while changed and len(cols) > 1:
        changed = False
        for i, (s, e) in enumerate(cols):
            if e - s < h:
                if i == 0:
                    j = 1
                elif i == len(cols) - 1:
                    j = i - 1
                else:
                    j = i - 1 if s - cols[i - 1][1] <= cols[i + 1][0] - e else i + 1
                cols[j] = [min(cols[j][0], s), max(cols[j][1], e)]
                del cols[i]
                changed = True
                break
    return cols
