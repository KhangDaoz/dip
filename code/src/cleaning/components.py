"""Thành phần liên thông — dùng cho hợp đồng ② và ML Bài toán 2."""
import numpy as np


def extract_components(binary: np.ndarray, min_area: int = 4) -> list:
    """connectedComponentsWithStats → list dict {"bbox": [x, y, w, h], "area": int, "mask": ndarray}."""
    raise NotImplementedError("extract_components")


def component_features(component: dict, img: np.ndarray, page_stats: dict) -> np.ndarray:
    """Vector đặc trưng hình dạng / nét / màu / kết cấu / ngữ cảnh (KeHoach_Truong.md mục 4.3)."""
    raise NotImplementedError("component_features")
