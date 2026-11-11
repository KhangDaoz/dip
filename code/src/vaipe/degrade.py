"""Các phép làm hỏng có kiểm soát để sinh ảnh test suy giảm L0/L2/L4 (plan §2.4; tools/make_degraded.py).

Mỗi hàm nhận ảnh BGR, trả về (ảnh hỏng, tham số thật dạng dict).
"""
from __future__ import annotations

import cv2
import numpy as np


def _paper_color(img):
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return tuple(int(v) for v in np.median(edge.reshape(-1, img.shape[2] if img.ndim == 3 else 1), axis=0))


def rotate(img: np.ndarray, angle_deg: float) -> tuple:
    """Xoay quanh tâm, giữ kích thước, vùng trống tô màu giấy (dương = ngược chiều kim đồng hồ)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=_paper_color(img))
    return out, {"angle_deg": float(angle_deg)}


def perspective(img: np.ndarray, level: float, rng: np.random.Generator) -> tuple:
    """Dịch ngẫu nhiên 4 góc level × cạnh, đặt trên nền bàn tối (khung rộng hơn để giữ đủ 4 góc giấy)."""
    h, w = img.shape[:2]
    m = int(level * max(h, w)) + 20
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = (src + rng.uniform(-level, level, (4, 2)) * np.array([w, h]) + m).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    size = (w + 2 * m, h + 2 * m)
    bg = np.full((size[1], size[0], 3), 60, np.uint8)
    warped = cv2.warpPerspective(img, H, size, flags=cv2.INTER_CUBIC)
    page = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, size, flags=cv2.INTER_NEAREST)
    bg[page > 0] = warped[page > 0]
    return bg, {"corners": dst.tolist()}


def gaussian_blur(img: np.ndarray, level: float) -> tuple:
    """Tích chập nhân Gaussian (C3 · S47–53): σ = 0,8·level, cửa sổ lẻ ≈ 6σ."""
    sigma = 0.8 * level
    k = int(6 * sigma) | 1
    return cv2.GaussianBlur(img, (k, k), sigma), {"sigma": float(sigma)}


def motion_blur(img: np.ndarray, level: int, rng: np.random.Generator) -> tuple:
    """Tích chập nhân đường thẳng dài 4·level+1, hướng ngẫu nhiên (C3 · S47–48)."""
    length = 4 * level + 1
    angle = float(rng.uniform(0, 180))
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1), (length, length))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k), {"length": length, "angle": angle}


def shadow(img: np.ndarray, level: int, rng: np.random.Generator) -> tuple:
    """I' = I · M, M là mặt nạ chiếu sáng (gradient + vệt tối mờ), 3 mức (C3 · S10–11)."""
    h, w = img.shape[:2]
    strength = (0.25, 0.45, 0.65)[level - 1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = np.cos(ang) * xx / w + np.sin(ang) * yy / h
    grad = (grad - grad.min()) / max(np.ptp(grad), 1e-6)
    blob = np.zeros((h, w), np.float32)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    cv2.ellipse(blob, (int(cx), int(cy)), (int(w * rng.uniform(0.15, 0.3)), int(h * rng.uniform(0.2, 0.4))),
                float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    blob = cv2.GaussianBlur(blob, (0, 0), max(w, h) * 0.05)
    M = 1 - strength * np.clip(0.6 * grad + blob, 0, 1)
    return np.clip(img.astype(np.float32) * M[..., None], 0, 255).astype(np.uint8), {"shadow_level": level}


def jpeg(img: np.ndarray, quality: int) -> tuple:
    """Nén JPEG (C6): lượng tử hóa DCT khối 8×8 làm mất tần số cao."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), {"jpeg_quality": int(quality)}


def gaussian_noise(img: np.ndarray, level: int, rng: np.random.Generator) -> tuple:
    """g = f + η, η ~ N(0, σ²), σ = 4·level (C3 · S65–66)."""
    sigma = 4.0 * level
    out = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return out, {"noise_sigma": float(sigma)}
