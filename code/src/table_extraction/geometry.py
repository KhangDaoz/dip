"""Hình học cơ bản trên ảnh nhị phân (mực = 255)."""
import cv2
import numpy as np


def _check_binary(binary: np.ndarray) -> None:
    if (not isinstance(binary, np.ndarray) or binary.ndim != 2 or binary.size == 0
            or binary.dtype != np.uint8):
        raise ValueError("binary phải là ảnh xám uint8 hai chiều không rỗng")


def char_height(binary: np.ndarray) -> float:
    """Chiều cao chữ trung vị của trang (từ thành phần liên thông) — dùng để chọn kích thước SE thích nghi."""
    _check_binary(binary)
    ink = np.where(binary > 0, 255, 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    heights = []
    for _, _, width, height, area in stats[1:count]:
        # Loại chấm nhiễu, đường kẻ ngang/dọc và các cụm lớn hơn nhiều chữ thường.
        if area >= 4 and height >= 2 and width <= 8 * height and height <= binary.shape[0] // 8:
            heights.append(float(height))
    if not heights:
        return 1.0
    median = float(np.median(heights))
    central = [h for h in heights if 0.5 * median <= h <= 2.0 * median]
    return float(np.median(central)) if central else median


def words_from_binary(binary: np.ndarray, char_h: float = None) -> list:
    """Dilation ngang thích nghi gom ký tự thành các cụm từ (ô kết quả). Trả về list bbox [x, y, w, h]."""
    _check_binary(binary)
    if char_h is None:
        char_h = char_height(binary)
    if not np.isfinite(char_h) or char_h <= 0:
        raise ValueError("char_h phải là số hữu hạn dương")

    ink = np.where(binary > 0, 255, 0).astype(np.uint8)
    # Dilation chỉ nối ký tự gần nhau theo chiều ngang; chiều cao không nở,
    # tránh nối các dòng sát nhau thành một thành phần.
    gap = max(1, int(round(0.45 * char_h)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * gap + 1, 1))
    joined = cv2.dilate(ink, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    min_area = max(3, int(round(0.08 * char_h * char_h)))
    boxes = []
    for x, y, width, height, area in stats[1:count]:
        if area >= min_area and height >= max(2, int(0.35 * char_h)):
            boxes.append([int(x), int(y), int(width), int(height)])
    return sorted(boxes, key=lambda box: (box[1] + box[3] / 2, box[0]))
