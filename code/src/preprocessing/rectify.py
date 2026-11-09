"""H7 nắn phối cảnh (KeHoach.md mục 5.1; 📘 QĐ · S6, nội suy C2 · S23–24)."""
import numpy as np


def homography_from_corners(corners: np.ndarray) -> tuple:
    """Tính H 3×3 từ 4 góc (TL, TR, BR, BL) sang hình chữ nhật đích. Trả về (H, (width, height))."""
    raise NotImplementedError("homography_from_corners")


def warp(img: np.ndarray, H: np.ndarray, size: tuple, interpolation: str = "bicubic") -> np.ndarray:
    """Ánh xạ ngược theo H. interpolation ∈ {"bicubic", "bilinear"} (đối chứng)."""
    raise NotImplementedError("warp")
