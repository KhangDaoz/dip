"""R8 hậu xử lý (KeHoach.md mục 6.1; C7 · S28, S34; QĐ · S7)."""
import cv2
import numpy as np


def opening(binary: np.ndarray, ksize: int = 2) -> np.ndarray:
    """Opening (f ⊖ s) ⊕ s bỏ chấm nhiễu nhỏ hơn SE (C7 · S28, S34)."""
    if ksize <= 1:
        return binary.copy()
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((ksize, ksize), np.uint8))


def remove_small(binary: np.ndarray, min_area: int = 10) -> np.ndarray:
    """Bỏ thành phần liên thông có diện tích < min_area (lọc vùng theo diện tích, QĐ · S7; C5 · S84)."""
    if min_area <= 1:
        return binary.copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[lab], 255, 0).astype(np.uint8)
