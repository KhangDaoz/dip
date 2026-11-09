"""R6 xóa dấu / chữ ký — CHỈ chạy cho đơn thuốc (KeHoach.md mục 6.1)."""
import numpy as np


def remove_marks(img_bgr: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
    """method ∈ {"telea", "ns", "fill"}; "fill" = tô màu nền (toán tử điểm, C3 · S10). Không đổi pixel ngoài mask."""
    raise NotImplementedError("remove_marks")
