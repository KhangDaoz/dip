"""S-B3 gom khoản thuốc — bước cốt lõi (KeHoach.md mục 8.1). Khoản = bbox [x, y, w, h]."""
import numpy as np

METHODS = ("dilation_rule", "each_line", "fixed_gap", "fixed_count", "dilation", "closing")


def items_dilation_rule(binary: np.ndarray, lines: list, h: float, s: float = 0.5, indent_tol: float = 1.0) -> list:
    """Phương án chính: Dilation dọc SE 1 × (s·h) (C7 · S5, S19, S24) + luật số thứ tự / thụt lề mở khoản mới (C5 · S87)."""
    raise NotImplementedError("items_dilation_rule")


def items_each_line(lines: list) -> list:
    """Đối chứng (a): mỗi dòng một khoản."""
    raise NotImplementedError("items_each_line")


def items_fixed_gap(lines: list, gap_px: float = 15.0) -> list:
    """Đối chứng (b): khe cố định."""
    raise NotImplementedError("items_fixed_gap")


def items_fixed_count(lines: list, n: int = 2) -> list:
    """Đối chứng (c): số dòng cố định mỗi khoản."""
    raise NotImplementedError("items_fixed_count")


def items_dilation(binary: np.ndarray, lines: list, h: float, s: float = 0.5) -> list:
    """Đối chứng (d): chỉ Dilation dọc, không luật."""
    raise NotImplementedError("items_dilation")


def items_closing(binary: np.ndarray, lines: list, h: float, s: float = 0.5) -> list:
    """Đối chứng (f): Closing dọc (C7 · S35, S38) — giữ kích thước khối."""
    raise NotImplementedError("items_closing")


def drop_fake_items(binary: np.ndarray, items: list, h: float, min_height_ratio: float = 0.5) -> list:
    """S-B4 hậu xử lý: Opening 3×3 trước projection; bỏ khoản cao < 0,5h hoặc ít mực (C7 · S28, S34; QĐ · S7)."""
    raise NotImplementedError("drop_fake_items")
