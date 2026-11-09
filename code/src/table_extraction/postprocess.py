"""K6 hậu xử lý (KeHoach.md mục 7.1; C7 · S28, S34; QĐ · S7)."""
import numpy as np


def opening_before_projection(binary: np.ndarray, ksize: int = 2) -> np.ndarray:
    raise NotImplementedError("opening_before_projection")


def drop_thin_rows(rows: list, words: list, h: float, min_ratio: float = 0.5) -> list:
    """Bỏ dòng cao < min_ratio·h."""
    raise NotImplementedError("drop_thin_rows")


def merge_narrow_columns(columns: list, h: float) -> list:
    """Gộp cột hẹp hơn h vào cột kề."""
    raise NotImplementedError("merge_narrow_columns")
