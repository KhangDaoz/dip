"""H5 khử lóa nhẹ (KeHoach.md mục 5.1)."""
import numpy as np


def remove_glare(img_bgr: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
    """Inpainting vùng lóa; chỉ gọi khi tỉ lệ lóa 1–8 %. method ∈ {"telea", "ns"}."""
    raise NotImplementedError("remove_glare")
