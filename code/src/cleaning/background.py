"""Ước lượng nền và khử bóng đổ. Thí nghiệm T1."""
import numpy as np
import cv2

from .binarize import _check_gray


def estimate_background(gray: np.ndarray, method: str = "median", kernel: int = 41) -> np.ndarray:
    """Ước lượng nền giấy sáng, trả uint8 cùng shape, không sửa gray.

    median: Median trực tiếp; closing: Closing thang xám với SE vuông;
    closing_median: Closing rồi Median, cùng kích thước kernel lẻ >= 3.
    Cả hai phép dùng biên lặp (BORDER_REPLICATE, mặc định của medianBlur).
    Đây là các phương án để so sánh, chưa phải cấu hình đã chọn trên val.
    """
    _check_gray(gray)
    if not isinstance(kernel, int) or kernel < 3 or kernel % 2 == 0:
        raise ValueError("kernel must be an odd integer >= 3")
    if method == "median":
        return cv2.medianBlur(gray, kernel)
    if method not in ("closing", "closing_median"):
        raise ValueError(f"Unknown background method: {method!r}")
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se, borderType=cv2.BORDER_REPLICATE)
    return cv2.medianBlur(closed, kernel) if method == "closing_median" else closed


def remove_shadow(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Chia ảnh cho nền rồi chuẩn hóa về 0..255 → ảnh xám nền phẳng."""
    _check_gray(gray)
    _check_gray(background)
    if gray.shape != background.shape:
        raise ValueError("gray and background must have the same shape")
    # Scale paper to white without min/max stretching, which amplifies noise on
    # nearly uniform pages. A zero background stays finite; no lost ink recovery.
    ratio = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)
    return np.clip(np.rint(ratio * 255.0), 0, 255).astype(np.uint8)
