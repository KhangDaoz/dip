"""Mở rộng K8 (bảng có kẻ, bộ 4) và K9 (cờ chữ đỏ) — làm nếu kịp (KeHoach.md mục 7.1)."""
import numpy as np


def ruled_lines(binary: np.ndarray, frac: int = 20) -> tuple:
    """K8: Opening SE 1 × (W/frac) và (H/frac) × 1 để GIỮ đường kẻ (C7 · S6, S28); Closing nối chỗ đứt (C7 · S35, S38).
    Trả về (mask_ngang, mask_dọc)."""
    raise NotImplementedError("ruled_lines")


def red_result_cells(red_text_mask: np.ndarray, cells: list) -> list:
    """K9: ô kết quả giao với mask chữ đỏ từ ② (C4 · S37–41)."""
    raise NotImplementedError("red_result_cells")
