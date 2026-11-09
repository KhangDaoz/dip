"""S-B0 chiều cao chữ, S-B2 tách dòng (KeHoach.md mục 8.1). Dòng = bbox [x, y, w, h]."""
import numpy as np


def char_height(binary: np.ndarray) -> float:
    """S-B0: trung vị chiều cao thành phần liên thông (C5 · S83–84; C3 · S73)."""
    raise NotImplementedError("char_height")


def lines_projection(binary: np.ndarray, h: float, smooth_window: int = 3, valley_ratio: float = 0.05,
                     min_height_ratio: float = 0.3) -> list:
    """Phương án chính: P(y) = Σ_x B(x, y); làm trơn W; khe dòng P(y) ≤ T; bỏ dòng cao < 0,3h (➕ C3 · S23; C5 · S62–64)."""
    raise NotImplementedError("lines_projection")


def lines_components(binary: np.ndarray, h: float) -> list:
    """Đối chứng: thành phần liên thông gom theo tâm y (C5 · S84–85)."""
    raise NotImplementedError("lines_components")
