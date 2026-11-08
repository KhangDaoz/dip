"""Gom cụm từ (ô kết quả) thành dòng. Thí nghiệm K1."""
import cv2
import numpy as np


def _validate_words(words: list) -> np.ndarray:
    if not isinstance(words, list):
        raise ValueError("words phải là list bbox [x, y, w, h]")
    if not words:
        return np.empty((0, 4), dtype=float)
    boxes = np.asarray(words, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("mỗi word phải là bbox hữu hạn [x, y, w, h]")
    if np.any(boxes[:, 2:] <= 0):
        raise ValueError("chiều rộng và chiều cao bbox phải dương")
    return boxes


def _clusters_by_centres(words: list, tolerance: float) -> list:
    boxes = _validate_words(words)
    if len(boxes) == 0:
        return []
    centres = boxes[:, 1] + boxes[:, 3] / 2
    order = np.argsort(centres, kind="stable")
    clusters = []
    for index in order:
        if not clusters or centres[index] - float(np.median(centres[clusters[-1]])) >= tolerance:
            clusters.append([int(index)])
        else:
            clusters[-1].append(int(index))
    return [sorted(cluster) for cluster in clusters]


def rows_projection(binary, words: list, char_h: float = None) -> list:
    """Projection ngang + Dilation thích nghi. Trả về list dòng, mỗi dòng là list chỉ số cụm từ trong `words`."""
    boxes = _validate_words(words)
    if len(boxes) == 0:
        return []
    if (not isinstance(binary, np.ndarray) or binary.ndim != 2 or binary.dtype != np.uint8
            or binary.size == 0):
        raise ValueError("binary phải là ảnh xám uint8 hai chiều không rỗng")
    if char_h is None:
        from .geometry import char_height
        char_h = char_height(binary)
    if not np.isfinite(char_h) or char_h <= 0:
        raise ValueError("char_h phải là số hữu hạn dương")

    ink = np.where(binary > 0, 255, 0).astype(np.uint8)
    height, width = ink.shape
    # Bỏ đường kẻ bảng dài khỏi phép chiếu ngang; chúng tạo một dải mực
    # xuyên trang và che mất các thung lũng giữa hai hàng.
    line_width = max(12, int(round(width * 0.12)))
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, 1)))
    text = cv2.subtract(ink, horizontal)
    # Chỉ chiếu vùng các cụm từ đưa vào; nền, dấu/logo và đường kẻ nằm ngoài
    # các box không được phép nối những hàng khác nhau.
    support = np.zeros_like(text)
    for x, y, box_w, box_h in boxes:
        x1, y1 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
        x2 = min(width, int(np.ceil(x + box_w)))
        y2 = min(height, int(np.ceil(y + box_h)))
        if x2 > x1 and y2 > y1:
            support[y1:y2, x1:x2] = 255
    text = cv2.bitwise_and(text, support)
    # Chiếu theo y riêng trong từng bbox. Dùng trọng tâm pixel mực làm tọa
    # độ đại diện cho cụm từ; điều này bớt nhạy với box cao/thấp khác nhau.
    projected_centres = []
    for x, y, box_w, box_h in boxes:
        x1, y1 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
        x2, y2 = min(width, int(np.ceil(x + box_w))), min(height, int(np.ceil(y + box_h)))
        crop = text[y1:y2, x1:x2]
        row_ink = np.count_nonzero(crop, axis=1).astype(float) if crop.size else np.array([])
        if row_ink.size and row_ink.max() > 0:
            # Bỏ các hàng có vài pixel rác/đường viền, giữ phần chiếu có mực rõ.
            row_ink[row_ink < max(1.0, 0.15 * row_ink.max())] = 0
            if row_ink.sum() > 0:
                local_centre = float(np.dot(np.arange(len(row_ink)), row_ink) / row_ink.sum())
                projected_centres.append(y1 + local_centre)
                continue
        projected_centres.append(float(y + box_h / 2))

    tolerance = max(2.0, 0.55 * float(char_h))
    order = np.argsort(projected_centres, kind="stable")
    clusters = []
    for index in order:
        if not clusters or (projected_centres[index]
                            - float(np.median([projected_centres[i] for i in clusters[-1]]))) >= tolerance:
            clusters.append([int(index)])
        else:
            clusters[-1].append(int(index))
    return [sorted(cluster) for cluster in clusters]


def rows_fixed(words: list, y_tol: float = 0.5) -> list:
    """Gom các cụm từ (bounding box) có tâm y lệch < y_tol × chiều cao box trung vị."""
    boxes = _validate_words(words)
    if isinstance(y_tol, bool) or not np.isfinite(y_tol) or y_tol <= 0:
        raise ValueError("y_tol phải là số hữu hạn dương")
    if len(boxes) == 0:
        return []
    return _clusters_by_centres(words, float(y_tol) * float(np.median(boxes[:, 3])))
