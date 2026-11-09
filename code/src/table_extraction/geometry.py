"""K1 chiều cao chữ, K2 tách cụm từ (KeHoach.md mục 7.1)."""
import numpy as np


def char_height(binary: np.ndarray, min_h: int = 4, max_ratio: float = 0.2) -> float:
    """K1: h = trung vị chiều cao thành phần liên thông, bỏ quá nhỏ / quá lớn (C5 · S84; C3 · S73)."""
    raise NotImplementedError("char_height")


def words_from_binary(binary: np.ndarray, h: float, a: float = 0.5) -> list:
    """K2: Dilation ngang SE 1 × (a·h) rồi lấy vùng liên thông (C7 · S5, S19, S23–24). Trả về list bbox [x, y, w, h]."""
    raise NotImplementedError("words_from_binary")
