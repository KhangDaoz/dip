"""K3 gom dòng (KeHoach.md mục 7.1). Trả về list dòng, mỗi dòng là list chỉ số cụm từ."""
import numpy as np


def rows_projection(binary: np.ndarray, words: list, h: float, se_factor: float = 3.0,
                    smooth_window: int = 3, valley_ratio: float = 0.05) -> list:
    """Phương án chính: P(y) = Σ_x B(x, y) trên ảnh Dilation ngang SE ≈ se_factor·h; làm trơn W (C5 · S62);
    tách tại thung lũng P(y) ≤ T (C5 · S63–64)."""
    raise NotImplementedError("rows_projection")


def rows_fixed(words: list, gap_px: float = 10.0) -> list:
    """Đối chứng: gom theo tâm y với khoảng cách cố định (pixel)."""
    raise NotImplementedError("rows_fixed")
