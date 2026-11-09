"""R4 phân biệt dấu / chữ đỏ, R5 mask dấu đặc (KeHoach.md mục 6.1)."""
import numpy as np


def circularity(area: float, perimeter: float) -> float:
    """C = 4πA / P² (C8 · S16): tròn 1, vuông 0,785."""
    raise NotImplementedError("circularity")


def split_stamp_vs_text(red_mask: np.ndarray, min_area_ratio: float = 0.01, min_circularity: float = 0.6) -> tuple:
    """R4: theo từng thành phần liên thông của mask đỏ (C5 · S84): dấu = diện tích lớn, C cao, có vành kín.
    Trả về (stamp_mask, red_text_mask)."""
    raise NotImplementedError("split_stamp_vs_text")


def fill_stamp(ring_mask: np.ndarray) -> np.ndarray:
    """R5: tách vành β(A) = A − (A ⊖ B) (C7 · S40), làm đầy vùng X_k = (X_{k−1} ⊕ B) ∩ Aᶜ (C7 · S43–45)."""
    raise NotImplementedError("fill_stamp")
