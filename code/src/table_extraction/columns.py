"""K4 tìm cột (KeHoach.md mục 7.1). Trả về list khoảng [x_start, x_end] của từng cột."""
import numpy as np


def columns_projection(binary: np.ndarray, rows: list, words: list, h: float,
                       gap_factor: float = 1.0, min_cross: float = 0.6) -> list:
    """Phương án chính: Q(x) = Σ_y B(x, y) trên thân bảng; khe cột = dải trắng rộng ≥ gap_factor·h
    xuyên qua ≥ min_cross số dòng (➕ C3 · S23; C5 · S63)."""
    raise NotImplementedError("columns_projection")


def columns_xycut(binary: np.ndarray, words: list, h: float, min_gap_factor: float = 1.0) -> list:
    """Đối chứng: XY-cut đệ quy — tách vùng top-down (➕ C5 · S89–90)."""
    raise NotImplementedError("columns_xycut")


def columns_fixed(words: list, gap_px: float = 20.0) -> list:
    """Đối chứng: gom theo khoảng cách x cố định (pixel)."""
    raise NotImplementedError("columns_fixed")
