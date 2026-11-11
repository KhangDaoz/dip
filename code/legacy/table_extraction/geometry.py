"""K1 chiều cao chữ, K2 tách cụm từ (KeHoach.md mục 7.1)."""
import cv2
import numpy as np


def char_height(binary: np.ndarray, min_h: int = 4, max_ratio: float = 0.2) -> float:
    """K1: h = trung vị chiều cao thành phần liên thông, bỏ quá nhỏ / quá lớn (C5 · S84; C3 · S73)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    hs = hs[(hs >= min_h) & (hs <= max_ratio * binary.shape[0])]
    return float(np.median(hs)) if len(hs) else 15.0


def words_from_binary(binary: np.ndarray, h: float, a: float = 0.5, min_area_ratio: float = 0.05,
                      v: float = 0.0) -> list:
    """K2: Dilation SE (v·h) × (a·h) rồi lấy vùng liên thông (C7 · S5, S19, S23–24). Trả về list bbox [x, y, w, h].

    v = 0: SE chỉ 1 hàng, chỉ nối khe giữa các ký tự cùng dòng. v > 0 (nhỏ hơn khe giữa hai dòng) nối thêm nét chữ bị
    đứt theo chiều dọc — gặp ở ảnh độ phân giải thấp (bộ 2, chữ cao 4–11 px) sau nhị phân hóa.
    """
    width = max(1, int(round(a * h)))
    height = max(1, int(round(v * h))) if v > 0 else 1
    smear = cv2.dilate((binary > 0).astype(np.uint8) * 255, np.ones((height, width), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(smear, connectivity=8)
    out = []
    for x, y, w, hh, area in stats[1:]:
        if hh >= 0.3 * h and w * hh >= min_area_ratio * h * h:
            out.append([int(x), int(y), int(w), int(hh)])
    return sorted(out, key=lambda b: (b[1], b[0]))
