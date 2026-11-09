"""S-B1 khoanh vùng thuốc (KeHoach.md mục 8.1)."""
import numpy as np


def find_drug_region(binary: np.ndarray, h: float, k: float = 2.0) -> list:
    """Dilation SE k·h gom khối văn bản (C7 · S19, S22–24); chọn khối có nhiều dòng bắt đầu bằng số thứ tự
    ở lề trái. Đối chứng: lấy cả trang. Trả về bbox [x, y, w, h]."""
    raise NotImplementedError("find_drug_region")
