"""R9 thành phần liên thông + nhãn bằng luật (KeHoach.md mục 6.1). Không dùng ML."""
import numpy as np


def extract_components(binary: np.ndarray, min_area: int = 10) -> list:
    """Gán nhãn liên thông (C5 · S84–85). Trả về list dict {"bbox": [x, y, w, h], "area", "mask_index"}."""
    raise NotImplementedError("extract_components")


def label_components(components: list, ink: dict, stamp_mask: np.ndarray) -> list:
    """Luật màu (R3) + hình dạng (R4) → list[common.contracts.Component], label ∈ COMPONENT_LABELS."""
    raise NotImplementedError("label_components")
