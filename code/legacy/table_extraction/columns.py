"""K4 tìm cột (KeHoach.md mục 7.1). Trả về list khoảng [x_start, x_end] của từng cột."""
import numpy as np


def _cols_from_gaps(blank: np.ndarray, min_gap: float, words: list) -> list:
    """Cắt trục x tại các dải "trống" rộng ≥ min_gap; giữ đoạn có chứa cụm từ."""
    W = len(blank)
    cuts, x = [], 0
    while x < W:
        if blank[x]:
            s = x
            while x < W and blank[x]:
                x += 1
            if x - s >= min_gap:
                cuts.append((s, x))
        else:
            x += 1
    bounds, prev = [], 0
    for s, e in cuts:
        if s > prev:
            bounds.append([prev, s])
        prev = e
    if prev < W:
        bounds.append([prev, W])
    cols = []
    for x0, x1 in bounds:   # thu gọn về phạm vi cụm từ thật sự nằm trong cột
        inside = [w for w in words if x0 <= w[0] + w[2] / 2 < x1]
        if inside:
            cols.append([min(w[0] for w in inside), max(w[0] + w[2] for w in inside)])
    return cols


def columns_projection(binary: np.ndarray, rows: list, words: list, h: float,
                       gap_factor: float = 1.0, min_cross: float = 0.6) -> list:
    """Phương án chính: Q(x) = Σ_y B(x, y) trên thân bảng; khe cột = dải trắng rộng ≥ gap_factor·h
    xuyên qua ≥ min_cross số dòng (➕ C3 · S23; C5 · S63).

    Tính Q riêng cho từng dòng (trên khung các cụm từ của dòng) rồi đếm, ở mỗi x, tỉ lệ dòng trống.
    """
    if not words:
        return []
    W = int(max(w[0] + w[2] for w in words)) + 1
    body = [r for r in rows if len(r) >= 2] or rows
    blank_count = np.zeros(W)
    for r in body:
        occ = np.zeros(W, bool)
        for i in r:
            x, _, w, _ = words[i]
            occ[x:x + w] = True
        blank_count += ~occ
    blank = blank_count >= min_cross * len(body)
    return _cols_from_gaps(blank, gap_factor * h, words)


def columns_xycut(binary: np.ndarray, words: list, h: float, min_gap_factor: float = 1.0) -> list:
    """Đối chứng: XY-cut đệ quy — tách vùng top-down (➕ C5 · S89–90).

    Cắt tập cụm từ theo khe trắng dọc rồi khe trắng ngang, lặp tới khi không cắt được (vùng "đồng nhất");
    cột = gộp khoảng x của các lá chồng nhau.
    """
    min_gap = min_gap_factor * h

    def split(idx, axis):
        lo = [words[i][axis] for i in idx]
        hi = [words[i][axis] + words[i][axis + 2] for i in idx]
        order = np.argsort(lo)
        groups, cur, end = [], [idx[order[0]]], hi[order[0]]
        for o in order[1:]:
            if lo[o] - end >= min_gap:
                groups.append(cur)
                cur = []
            cur.append(idx[o])
            end = max(end, hi[o])
        groups.append(cur)
        return groups

    leaves = []

    def recurse(idx, axis, depth=0):
        parts = split(idx, axis)
        if len(parts) == 1:
            other = split(idx, 1 - axis)
            if len(other) == 1 or depth > 20:
                leaves.append(idx)
                return
            for p in other:
                recurse(p, axis, depth + 1)
            return
        for p in parts:
            recurse(p, 1 - axis, depth + 1)

    if not words:
        return []
    recurse(list(range(len(words))), 0)
    spans = sorted([min(words[i][0] for i in l), max(words[i][0] + words[i][2] for i in l)] for l in leaves)
    cols = [spans[0]]
    for s, e in spans[1:]:
        if s <= cols[-1][1]:
            cols[-1][1] = max(cols[-1][1], e)
        else:
            cols.append([s, e])
    return cols


def columns_fixed(words: list, gap_px: float = 20.0) -> list:
    """Đối chứng: gom theo khoảng cách x cố định (pixel)."""
    if not words:
        return []
    spans = sorted([w[0], w[0] + w[2]] for w in words)
    cols = [spans[0][:]]
    for s, e in spans[1:]:
        if s <= cols[-1][1] + gap_px:
            cols[-1][1] = max(cols[-1][1], e)
        else:
            cols.append([s, e])
    return cols
