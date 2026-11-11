"""S-B0 chiều cao chữ, S-B2 tách dòng (KeHoach.md mục 8.1). Dòng = bbox [x, y, w, h]."""
import cv2
import numpy as np


def char_height(binary: np.ndarray, min_h: int = 4, max_ratio: float = 0.1) -> float:
    """S-B0: trung vị chiều cao thành phần liên thông (C5 · S83–84; C3 · S73). Trung vị bền với dấu chấm / khung."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    hs = hs[(hs >= min_h) & (hs <= max_ratio * binary.shape[0])]
    return float(np.median(hs)) if len(hs) else 15.0


def smooth(profile: np.ndarray, window: int) -> np.ndarray:
    """Làm trơn lược đồ h_smooth(b) = 1/W Σ h_raw(b − w) (C5 · S62)."""
    if window <= 1:
        return profile.astype(np.float64)
    return np.convolve(profile.astype(np.float64), np.ones(window) / window, mode="same")


def _band_bbox(binary: np.ndarray, y0: int, y1: int):
    cols = np.flatnonzero(binary[y0:y1].any(axis=0))
    if not len(cols):
        return None
    rows = np.flatnonzero(binary[y0:y1, cols[0]:cols[-1] + 1].any(axis=1))
    return [int(cols[0]), int(y0 + rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)]


def lines_projection(binary: np.ndarray, h: float, smooth_window: int = 3, valley_ratio: float = 0.05,
                     min_height_ratio: float = 0.3) -> list:
    """Phương án chính: P(y) = Σ_x B(x, y); làm trơn W; khe dòng P(y) ≤ T; bỏ dòng cao < 0,3h (➕ C3 · S23; C5 · S62–64).

    P(y) là lược đồ đếm pixel mực theo hàng (như p(r_k) = n_k / n nhưng đếm theo vị trí y); T = valley_ratio · max P.
    """
    b = (binary > 0).astype(np.uint8)
    prof = smooth(b.sum(axis=1), smooth_window)
    if prof.max() <= 0:
        return []
    on = prof > valley_ratio * prof.max()
    edges = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    out = []
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        box = _band_bbox(b, int(s), int(e))
        if box and box[3] >= min_height_ratio * h:
            out.append(box)
    return out


def lines_components(binary: np.ndarray, h: float) -> list:
    """Đối chứng: thành phần liên thông gom theo tâm y (C5 · S84–85): vùng nào có tâm y cách tâm dòng < 0,5h thì vào dòng đó."""
    n, _, stats, cents = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    comps = [(stats[i], cents[i][1]) for i in range(1, n)
             if stats[i, cv2.CC_STAT_HEIGHT] >= 0.3 * h and stats[i, cv2.CC_STAT_HEIGHT] <= 4 * h]
    comps.sort(key=lambda c: c[1])
    groups = []
    for st, cy in comps:
        if groups and abs(cy - groups[-1]["cy"]) < 0.5 * h:
            g = groups[-1]
            g["boxes"].append(st)
            g["cy"] = np.mean([b[1] + b[3] / 2 for b in g["boxes"]])
        else:
            groups.append({"boxes": [st], "cy": cy})
    out = []
    for g in groups:
        x0 = min(b[0] for b in g["boxes"])
        y0 = min(b[1] for b in g["boxes"])
        x1 = max(b[0] + b[2] for b in g["boxes"])
        y1 = max(b[1] + b[3] for b in g["boxes"])
        out.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
    return sorted(out, key=lambda b: b[1])
