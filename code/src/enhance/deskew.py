from __future__ import annotations

import cv2
import numpy as np

from common.io import to_gray

from .binarize import sauvola

WORK_SIDE = 800


def _shrink(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    s = WORK_SIDE / max(h, w)
    if s >= 1:
        return binary
    return cv2.resize(binary, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _profile_score(binary: np.ndarray, theta: float) -> float:
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -theta, 1.0)
    rot = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return float(rot.sum(axis=1, dtype=np.float64).var())


def skew_projection(binary: np.ndarray, max_angle: float = 10.0, step: float = 0.25) -> float:
    b = (_shrink(binary) > 0).astype(np.uint8)
    coarse = np.arange(-max_angle, max_angle + 1e-9, 1.0)
    best = max(coarse, key=lambda t: _profile_score(b, t))
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, step)
    return float(max(fine, key=lambda t: _profile_score(b, t)))


def estimate_skew(img: np.ndarray, max_angle: float = 10.0, step: float = 0.25) -> float:
    return skew_projection(sauvola(to_gray(img), window=31, k=0.2), max_angle=max_angle, step=step)


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def deskew(img: np.ndarray, max_angle: float = 10.0, step: float = 0.25, min_angle: float = 0.2) -> tuple:
    angle = estimate_skew(img, max_angle, step)
    if abs(angle) < min_angle:
        return img, angle
    return rotate(img, -angle), angle
