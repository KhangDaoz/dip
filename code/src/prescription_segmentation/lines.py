"""Tách dòng văn bản. Mọi hàm trả về list bbox [x, y, w, h] từ trên xuống."""
import cv2
import numpy as np


def _runs(mask_1d: np.ndarray) -> list:
    """Các đoạn liên tiếp True của mảng 1 chiều → list (start, end) với end không tính."""
    padded = np.concatenate([[False], mask_1d, [False]])
    diff = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def _band_bbox(binary: np.ndarray, y0: int, y1: int) -> list:
    cols = np.flatnonzero(binary[y0:y1].any(axis=0))
    return [int(cols[0]), int(y0), int(cols[-1] - cols[0] + 1), int(y1 - y0)]


def lines_projection(binary, min_gap: int = 3, min_frac: float = 0.0, min_height: int = 3) -> list:
    """Projection ngang: dòng = đoạn liên tiếp có tổng mực > min_frac × đỉnh, cách nhau ≥ min_gap px.

    Hai đoạn cách nhau < min_gap px được nối (dấu tiếng Việt, nét đứt); đoạn thấp hơn min_height bị bỏ.
    """
    profile = (np.asarray(binary) > 0).sum(axis=1)
    if profile.max() == 0:
        return []
    runs = _runs(profile > min_frac * profile.max())
    merged = []
    for y0, y1 in runs:
        if merged and y0 - merged[-1][1] < min_gap:
            merged[-1][1] = y1
        else:
            merged.append([y0, y1])
    return [_band_bbox(binary, y0, y1) for y0, y1 in merged if y1 - y0 >= min_height]


def lines_components(binary, min_area: int = 4, overlap: float = 0.5, max_height_ratio: float = 4.0) -> list:
    """Gom thành phần liên thông theo tâm y.

    Bỏ nhiễu (< min_area px) và thành phần cao bất thường (> max_height_ratio × chiều cao trung vị: đường kẻ
    dọc, khung bảng). Một thành phần vào dòng nếu phần chồng dọc ≥ overlap × chiều cao của bên thấp hơn.
    """
    binary = (np.asarray(binary) > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = [stats[i, :4].tolist() for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not comps:
        return []
    h_med = float(np.median([c[3] for c in comps]))
    comps = [c for c in comps if c[3] <= max_height_ratio * h_med]
    comps.sort(key=lambda c: c[1] + c[3] / 2)

    lines = []   # [x0, y0, x1, y1]
    for x, y, w, h in comps:
        best, best_ov = None, 0.0
        for ln in lines[-5:]:          # chỉ xét vài dòng gần nhất vì đã sắp theo tâm y
            ov = min(y + h, ln[3]) - max(y, ln[1])
            ratio = ov / max(1, min(h, ln[3] - ln[1]))
            if ratio >= overlap and ratio > best_ov:
                best, best_ov = ln, ratio
        if best is None:
            lines.append([x, y, x + w, y + h])
        else:
            best[0], best[1] = min(best[0], x), min(best[1], y)
            best[2], best[3] = max(best[2], x + w), max(best[3], y + h)
    lines = _absorb_small_lines(sorted(lines, key=lambda ln: ln[1]), overlap)
    return [[int(a), int(b), int(c - a), int(d - b)] for a, b, c, d in lines]


def _absorb_small_lines(lines: list, overlap: float, small: float = 0.6) -> list:
    """Dòng thấp (dấu thanh, dấu chấm tách khỏi chữ) hoặc chồng dọc nhiều với dòng kề → nhập vào dòng gần nhất."""
    changed = True
    while changed and len(lines) > 1:
        changed = False
        h_med = float(np.median([ln[3] - ln[1] for ln in lines]))
        for i, ln in enumerate(lines):
            h = ln[3] - ln[1]
            nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(lines)]

            def dist(j):
                o = lines[j]
                return max(0, max(o[1], ln[1]) - min(o[3], ln[3]))

            def ov(j):
                o = lines[j]
                return (min(o[3], ln[3]) - max(o[1], ln[1])) / max(1, min(h, o[3] - o[1]))

            j = min(nbrs, key=dist)
            if h < small * h_med or ov(j) >= overlap:
                o = lines[j]
                lines[j] = [min(o[0], ln[0]), min(o[1], ln[1]), max(o[2], ln[2]), max(o[3], ln[3])]
                del lines[i]
                changed = True
                break
    return lines
