"""H6 tìm 4 góc tờ giấy (KeHoach.md mục 5.1).

Mọi hàm trả về mảng 4×2 float theo thứ tự TL, TR, BR, BL trên ảnh đầu vào, hoặc None nếu thất bại.
"""
import numpy as np


def corners_contour(gray: np.ndarray, canny_low: int = 50, canny_high: int = 100,
                    approx_eps: float = 0.02):
    """Phương án chính: Canny (C5 · S36–49, T1 = 2·T2) → đường bao lớn nhất → xấp xỉ tứ giác."""
    raise NotImplementedError("corners_contour")


def corners_hough(gray: np.ndarray, canny_low: int = 50, canny_high: int = 100):
    """Dự phòng 1: đường thẳng Hough → giao điểm 4 cạnh."""
    raise NotImplementedError("corners_hough")


def corners_minrect(gray: np.ndarray):
    """Dự phòng 2: hình chữ nhật bao nhỏ nhất của vùng giấy."""
    raise NotImplementedError("corners_minrect")


def find_corners(gray: np.ndarray) -> tuple:
    """Chạy lần lượt contour → hough → minrect → cả ảnh. Trả về (corners, method), method ∈ CORNER_METHODS."""
    raise NotImplementedError("find_corners")
