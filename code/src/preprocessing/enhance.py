"""Tăng tương phản."""
import cv2
import numpy as np


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """CLAHE trên kênh L của Lab (ảnh màu) hoặc trực tiếp (ảnh xám). Giữ nguyên shape."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid_size))
    if img.ndim == 2:
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def equalize_global(img: np.ndarray) -> np.ndarray:
    """Cân bằng histogram toàn cục — phương án so sánh."""
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
