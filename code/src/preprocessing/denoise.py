"""H2 khử nhiễu trước khi dò biên (KeHoach.md mục 5.1)."""
import cv2
import numpy as np


def median(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Phương án chính: lọc trung vị f̂ = median{g(s,t)} (C3 · S56, S73) — bỏ nhiễu muối tiêu mà ít làm mờ biên."""
    return cv2.medianBlur(gray, ksize)


def mean(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Đối chứng: lọc trung bình, nhân 1/(k²) (C3 · S50–51)."""
    return cv2.blur(gray, (ksize, ksize))


def gaussian(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Đối chứng: trung bình có trọng số / Gaussian (C3 · S52–53)."""
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def midpoint(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Đối chứng: lọc điểm giữa ½(max + min) (C3 · S74–75). Max = phép giãn, min = phép co trên ảnh xám."""
    se = np.ones((ksize, ksize), np.uint8)
    mx = cv2.dilate(gray, se).astype(np.uint16)
    mn = cv2.erode(gray, se).astype(np.uint16)
    return ((mx + mn) // 2).astype(np.uint8)
