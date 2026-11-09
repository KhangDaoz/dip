"""H2 khử nhiễu trước khi dò biên (KeHoach.md mục 5.1)."""
import numpy as np


def median(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Phương án chính: lọc trung vị (C3 · S56, S73)."""
    raise NotImplementedError("median")


def mean(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Đối chứng: lọc trung bình (C3 · S50–51)."""
    raise NotImplementedError("mean")


def gaussian(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Đối chứng: trung bình có trọng số / Gaussian (C3 · S52–53)."""
    raise NotImplementedError("gaussian")


def midpoint(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Đối chứng: lọc điểm giữa ½(max + min) (C3 · S75)."""
    raise NotImplementedError("midpoint")
