"""A1 nắn phối cảnh + A2 chỉnh nghiêng (ch. 5: Canny, chiếu ngang; ch. 2: nội suy).

Bọc `legacy/preprocessing/{corners,rectify,deskew}.py`. Mọi hàm: ảnh BGR/xám uint8 → cùng loại.
"""
from __future__ import annotations

import cv2
import numpy as np

from common.io import to_gray
from legacy.cleaning.binarize import sauvola
from legacy.preprocessing.corners import find_corners
from legacy.preprocessing.deskew import skew_projection
from legacy.preprocessing.rectify import homography_from_corners, warp


def rectify(img: np.ndarray, canny_low: int = 50, canny_high: int = 150) -> tuple:
    """A1: Canny → contour 4 góc → warpPerspective. Không tìm được tờ giấy thì trả ảnh gốc.

    Trả về (ảnh, method) với method ∈ {contour, hough, minrect, full_image}.
    """
    corners, method = find_corners(to_gray(img), canny_low, canny_high)
    if method == "full_image":
        return img, method
    H, size = homography_from_corners(corners)
    return warp(img, H, size), method


def estimate_skew(img: np.ndarray, max_angle: float = 10.0, step: float = 0.25) -> float:
    """A2: góc nghiêng (độ, dương = nội dung xoay ngược chiều kim đồng hồ) = argmax Var(chiếu ngang)."""
    ink = sauvola(to_gray(img), window=31, k=0.2)
    return skew_projection(ink, max_angle=max_angle, step=step)


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Xoay ảnh `angle_deg` độ (dương = ngược chiều kim đồng hồ), nội suy bậc ba, viền lặp lại."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def deskew(img: np.ndarray, max_angle: float = 10.0, step: float = 0.25, min_angle: float = 0.2) -> tuple:
    """A2: ước lượng góc rồi xoay ngược. Góc < min_angle thì bỏ qua (tránh nội suy vô ích). Trả về (ảnh, góc)."""
    angle = estimate_skew(img, max_angle, step)
    if abs(angle) < min_angle:
        return img, angle
    return rotate(img, -angle), angle
