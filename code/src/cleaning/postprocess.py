"""R8 hậu xử lý (KeHoach.md mục 6.1; C7 · S28, S34; QĐ · S7)."""
import numpy as np


def opening(binary: np.ndarray, ksize: int = 2) -> np.ndarray:
    """Opening (f ⊖ s) ⊕ s bỏ chấm nhiễu."""
    raise NotImplementedError("opening")


def remove_small(binary: np.ndarray, min_area: int = 10) -> np.ndarray:
    """Bỏ thành phần liên thông có diện tích < min_area."""
    raise NotImplementedError("remove_small")
