"""H9 tăng tương phản (KeHoach.md mục 5.1). Chỉ chạy trên bản sao dùng cho phân loại."""
import cv2
import numpy as np


def clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Phương án chính: CLAHE — cân bằng lược đồ theo từng ô, giới hạn độ dốc bằng clipLimit (📘 QĐ · S4; TH · C3)."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)).apply(gray)


def equalize_global(gray: np.ndarray) -> np.ndarray:
    """Đối chứng: cân bằng lược đồ toàn cục s_k = (L−1) Σ_{j≤k} n_j / n (C3 · S32–33)."""
    hist = np.bincount(gray.ravel(), minlength=256)
    cdf = np.cumsum(hist) / gray.size
    lut = np.round(255 * cdf).astype(np.uint8)
    return lut[gray]
