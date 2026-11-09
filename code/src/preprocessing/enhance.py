"""H9 tăng tương phản (KeHoach.md mục 5.1). Chỉ chạy trên bản sao dùng cho phân loại."""
import numpy as np


def clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Phương án chính: CLAHE (📘 QĐ · S4; TH · C3)."""
    raise NotImplementedError("clahe")


def equalize_global(gray: np.ndarray) -> np.ndarray:
    """Đối chứng: cân bằng lược đồ toàn cục s_k = (L−1) Σ n_j / n (C3 · S32–33)."""
    raise NotImplementedError("equalize_global")
