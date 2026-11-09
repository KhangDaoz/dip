"""Các phép làm hỏng có kiểm soát của bộ 5 (KeHoach.md mục 8.2).

Mỗi hàm nhận ảnh sạch BGR + mức (level) + rng, trả về (ảnh hỏng, ground truth dạng dict).
"""
import numpy as np


def perspective(img: np.ndarray, level: float, rng: np.random.Generator) -> tuple:
    """Dịch ngẫu nhiên 4 góc level × cạnh; H 3×3; nội suy cubic cho ảnh, nearest cho mask. GT: H, 4 góc (📘 QĐ · S6)."""
    raise NotImplementedError("perspective")


def rotate(img: np.ndarray, angle_deg: float) -> tuple:
    """Ma trận xoay. GT: góc thật."""
    raise NotImplementedError("rotate")


def gaussian_blur(img: np.ndarray, level: int) -> tuple:
    """Tích chập nhân Gaussian, 4 mức (C3 · S47–53)."""
    raise NotImplementedError("gaussian_blur")


def motion_blur(img: np.ndarray, level: int, rng: np.random.Generator) -> tuple:
    """Tích chập nhân đường thẳng, 4 mức (C3 · S47–48)."""
    raise NotImplementedError("motion_blur")


def shadow(img: np.ndarray, level: int, rng: np.random.Generator) -> tuple:
    """I' = I · M, M là mặt nạ chiếu sáng (gradient, bàn tay làm mờ), 3 mức (C3 · S10–11). GT: mask bóng."""
    raise NotImplementedError("shadow")


def glare(img: np.ndarray, level: int, rng: np.random.Generator, exposure_range=(0.80, 0.92)) -> tuple:
    """Giảm phơi sáng trước, rồi tăng V / giảm S trong elip Gaussian, 3 mức (C4 · S39–41). GT: mask lóa, ảnh sạch đã giảm phơi sáng."""
    raise NotImplementedError("glare")


def stamp_pen(img: np.ndarray, level: int, rng: np.random.Generator, kind: str = "stamp") -> tuple:
    """Dấu đỏ / bút xanh / bút đen trộn trong suốt, 3 mức che (C4 · S29–31, S38). GT: mask dấu / bút."""
    raise NotImplementedError("stamp_pen")


def jpeg(img: np.ndarray, quality: int) -> tuple:
    """Nén JPEG chất lượng 30–90, 3 mức (C4 · S15, S18; C6 · S40–54)."""
    raise NotImplementedError("jpeg")


def gaussian_noise(img: np.ndarray, level: int, rng: np.random.Generator) -> tuple:
    """g = f + η, 3 mức (C3 · S65–66)."""
    raise NotImplementedError("gaussian_noise")


def clean_binary(img: np.ndarray) -> np.ndarray:
    """Ảnh nhị phân chuẩn = Otsu trên ảnh sạch (C5 · S65–67). Mực = 255."""
    raise NotImplementedError("clean_binary")
