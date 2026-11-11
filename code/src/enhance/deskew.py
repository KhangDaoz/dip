"""A2 chỉnh nghiêng bằng lược đồ chiếu ngang (ch. 5). Góc theo độ, dương = nội dung xoay ngược chiều kim đồng hồ.

θ* = argmax_θ Var(P_θ), P_θ(y) = Σ_x B_θ(x, y): khi dòng chữ nằm ngang, mỗi hàng hoặc đầy mực hoặc trống nên
lược đồ có đỉnh / thung lũng rõ nhất. Làm thẳng: xoay ảnh một góc −θ*.
"""
from __future__ import annotations

import cv2
import numpy as np

from common.io import to_gray

from .binarize import sauvola

WORK_SIDE = 800   # thu nhỏ ảnh nhị phân khi dò góc cho nhanh; góc không đổi khi co giãn đều


def _shrink(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    s = WORK_SIDE / max(h, w)
    if s >= 1:
        return binary
    return cv2.resize(binary, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _profile_score(binary: np.ndarray, theta: float) -> float:
    """Var(P_θ): xoay ảnh −θ rồi đếm pixel mực theo hàng."""
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -theta, 1.0)
    rot = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return float(rot.sum(axis=1, dtype=np.float64).var())


def skew_projection(binary: np.ndarray, max_angle: float = 10.0, step: float = 0.25) -> float:
    """Tìm thô bước 1°, rồi tinh bước `step` quanh góc tốt nhất. `binary`: mực = 255."""
    b = (_shrink(binary) > 0).astype(np.uint8)
    coarse = np.arange(-max_angle, max_angle + 1e-9, 1.0)
    best = max(coarse, key=lambda t: _profile_score(b, t))
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, step)
    return float(max(fine, key=lambda t: _profile_score(b, t)))


def estimate_skew(img: np.ndarray, max_angle: float = 10.0, step: float = 0.25) -> float:
    """Góc nghiêng của ảnh BGR/xám: Sauvola → skew_projection."""
    return skew_projection(sauvola(to_gray(img), window=31, k=0.2), max_angle=max_angle, step=step)


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Xoay ảnh `angle_deg` độ (dương = ngược chiều kim đồng hồ), nội suy bậc ba, viền lặp lại."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def deskew(img: np.ndarray, max_angle: float = 10.0, step: float = 0.25, min_angle: float = 0.2) -> tuple:
    """Ước lượng góc rồi xoay ngược. Góc < min_angle thì bỏ qua (tránh nội suy vô ích). Trả về (ảnh, góc)."""
    angle = estimate_skew(img, max_angle, step)
    if abs(angle) < min_angle:
        return img, angle
    return rotate(img, -angle), angle
