"""K3 gom dòng (KeHoach.md mục 7.1). Trả về list dòng, mỗi dòng là list chỉ số cụm từ."""
import cv2
import numpy as np


def smooth(profile: np.ndarray, window: int) -> np.ndarray:
    """Làm trơn lược đồ h_smooth(b) = 1/W Σ h_raw(b − w), W lẻ (C5 · S62)."""
    if window <= 1:
        return profile.astype(np.float64)
    return np.convolve(profile.astype(np.float64), np.ones(window) / window, mode="same")


def segments(profile: np.ndarray, thr: float) -> list:
    """Các đoạn liên tiếp có profile > thr (thung lũng ≤ thr là khe, C5 · S63–64). Trả về [(start, end_excl)]."""
    on = profile > thr
    edges = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _assign(words: list, bands: list) -> list:
    rows = [[] for _ in bands]
    for i, (x, y, w, h) in enumerate(words):
        cy = y + h / 2
        # dòng có khoảng chồng lấn lớn nhất với cụm từ; nếu không chồng thì dòng gần tâm nhất
        overlaps = [max(0, min(y + h, e) - max(y, s)) for s, e in bands]
        j = int(np.argmax(overlaps)) if max(overlaps, default=0) > 0 else \
            int(np.argmin([abs(cy - (s + e) / 2) for s, e in bands]))
        rows[j].append(i)
    return [sorted(r, key=lambda k: words[k][0]) for r in rows if r]


def rows_projection(binary: np.ndarray, words: list, h: float, se_factor: float = 3.0,
                    smooth_window: int = 3, valley_ratio: float = 0.05) -> list:
    """Phương án chính: P(y) = Σ_x B(x, y) trên ảnh Dilation ngang SE ≈ se_factor·h; làm trơn W (C5 · S62);
    tách tại thung lũng P(y) ≤ T (C5 · S63–64)."""
    if not words:
        return []
    width = max(1, int(round(se_factor * h)))
    smear = cv2.dilate((binary > 0).astype(np.uint8), np.ones((1, width), np.uint8))
    prof = smooth(smear.sum(axis=1), smooth_window)
    bands = [(s, e) for s, e in segments(prof, valley_ratio * prof.max()) if e - s >= 0.3 * h]
    if not bands:
        return [list(range(len(words)))]
    return _assign(words, bands)


def rows_fixed(words: list, gap_px: float = 10.0) -> list:
    """Đối chứng: gom theo tâm y với khoảng cách cố định (pixel)."""
    order = sorted(range(len(words)), key=lambda i: words[i][1] + words[i][3] / 2)
    rows, cur, cur_cy = [], [], None
    for i in order:
        cy = words[i][1] + words[i][3] / 2
        if cur and abs(cy - cur_cy) > gap_px:
            rows.append(cur)
            cur = []
        cur.append(i)
        cur_cy = np.mean([words[k][1] + words[k][3] / 2 for k in cur])
    if cur:
        rows.append(cur)
    return [sorted(r, key=lambda k: words[k][0]) for r in rows]


def row_bbox(words: list, row: list) -> list:
    xs = [words[i][0] for i in row]
    ys = [words[i][1] for i in row]
    xe = [words[i][0] + words[i][2] for i in row]
    ye = [words[i][1] + words[i][3] for i in row]
    return [min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)]
