"""S-B3 gom khoản thuốc — bước cốt lõi (KeHoach.md mục 8.1). Khoản = bbox [x, y, w, h].

Các phương án Dilation / Closing làm việc trên ảnh "dải dòng": mỗi dòng (từ S-B2) tô thành một dải đặc theo bề rộng
của nó — tương đương Dilation ngang rất dài trên ảnh mực — để chữ cùng dòng không bị đứt, rồi mới giãn theo chiều dọc.
"""
import cv2
import numpy as np

METHODS = ("dilation_rule", "each_line", "fixed_gap", "fixed_count", "dilation", "closing")


def union(boxes: list) -> list:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def _line_strips(shape, lines: list) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for x, y, w, h in lines:
        m[y:y + h, x:x + w] = 255
    return m


def _vertical_se(h: float, s: float) -> np.ndarray:
    return np.ones((max(1, int(round(s * h))), 1), np.uint8)   # SE dọc 1 × (s·h): chỉ giãn theo trục y (C7 · S5)


def _groups_from_mask(mask: np.ndarray, lines: list) -> list:
    """Gán mỗi dòng vào khối liên thông chứa tâm của nó; trả về list (chỉ số dòng, bbox khối)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    groups = {}
    for i, (x, y, w, h) in enumerate(lines):
        cy, cx = min(y + h // 2, mask.shape[0] - 1), min(x + w // 2, mask.shape[1] - 1)
        groups.setdefault(int(lab[cy, cx]), []).append(i)
    out = []
    for k, idx in groups.items():
        x, y, w, h, _ = stats[k] if k > 0 else (*union([lines[i] for i in idx]), 0)
        out.append((sorted(idx, key=lambda i: lines[i][1]), [int(x), int(y), int(w), int(h)]))
    return sorted(out, key=lambda g: g[1][1])


def items_dilation(binary: np.ndarray, lines: list, h: float, s: float = 0.5) -> list:
    """Đối chứng (d): chỉ Dilation dọc, không luật. bbox = khối sau Dilation (bị phình, C7 · S24)."""
    if not lines:
        return []
    mask = cv2.dilate(_line_strips(binary.shape, lines), _vertical_se(h, s))
    return [box for _, box in _groups_from_mask(mask, lines)]


def items_closing(binary: np.ndarray, lines: list, h: float, s: float = 0.5) -> list:
    """Đối chứng (f): Closing dọc (C7 · S35, S38) — lấp khe nhỏ mà giữ kích thước khối."""
    if not lines:
        return []
    mask = cv2.morphologyEx(_line_strips(binary.shape, lines), cv2.MORPH_CLOSE, _vertical_se(h, s))
    return [box for _, box in _groups_from_mask(mask, lines)]


def line_boldness(binary: np.ndarray, line: list, se: int = 3) -> float:
    """Độ đậm của một dòng = tỉ lệ pixel mực còn lại sau phép co SE se×se (C7 · S9, S12–14).

    Phép co làm mất nét mảnh hơn SE (S14: "cắt bớt phần dư thừa, thu nhỏ đối tượng"), nét đậm thì sống sót —
    nên dòng tên thuốc in đậm có tỉ lệ cao hơn hẳn dòng cách dùng in thường.
    """
    x, y, w, h = line
    reg = (binary[y:y + h, x:x + w] > 0).astype(np.uint8)
    ink = int(reg.sum())
    return float(cv2.erode(reg, np.ones((se, se), np.uint8)).sum() / ink) if ink else 0.0


def otsu_1d(values) -> float:
    """Ngưỡng Otsu trên một dãy số (C5 · S65–67): chọn t cực đại phương sai giữa hai nhóm ≤ t và > t."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    best_t, best = v[0], -1.0
    for i in range(1, len(v)):
        a, b = v[:i], v[i:]
        sb = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if sb > best:
            best, best_t = sb, (v[i - 1] + v[i]) / 2
    return float(best_t)


def bold_flags(binary: np.ndarray, lines: list, bold_ratio: float = 1.5) -> list:
    """Chia các dòng thành "đậm / thường" bằng ngưỡng Otsu trên độ đậm; chỉ chấp nhận khi trung bình nhóm đậm
    ≥ bold_ratio × nhóm thường (hai nhóm thật sự khác nhau). Không đạt → không dòng nào đậm."""
    if len(lines) < 2:
        return [False] * len(lines)
    b = np.array([line_boldness(binary, l) for l in lines])
    t = otsu_1d(b)
    hi, lo = b[b > t], b[b <= t]
    if not len(hi) or not len(lo) or hi.mean() < bold_ratio * max(lo.mean(), 1e-6):
        return [False] * len(lines)
    return list(b > t)


def items_dilation_rule(binary: np.ndarray, lines: list, h: float, s: float = 0.5, indent_tol: float = 1.0,
                        bold_ratio: float = 1.5) -> list:
    """Phương án chính: Dilation dọc SE 1 × (s·h) (C7 · S5, S19, S24) + luật mở khoản mới (C5 · S87 hợp vùng).

    Luật: trong mỗi khối Dilation, dòng ĐẬM (tên thuốc, nhận bằng phép co + ngưỡng Otsu) mở khoản mới; các dòng
    thường phía dưới hợp vào khoản đó. Không có dòng đậm (mẫu đơn khác) → dự phòng luật thụt lề: dòng sát lề trái
    (lệch ≤ indent_tol·h) mở khoản mới. bbox khoản = hợp các dòng của nó (không phình như Dilation).
    """
    if not lines:
        return []
    mask = cv2.dilate(_line_strips(binary.shape, lines), _vertical_se(h, s))
    bold = bold_flags(binary, lines, bold_ratio)
    use_bold = any(bold)
    items = []
    for idx, _ in _groups_from_mask(mask, lines):
        left = min(lines[i][0] for i in idx)
        cur = []
        for i in idx:
            starts = bold[i] if use_bold else lines[i][0] - left <= indent_tol * h
            if cur and starts:
                items.append(union([lines[j] for j in cur]))
                cur = []
            cur.append(i)
        items.append(union([lines[j] for j in cur]))
    return items


def items_each_line(lines: list) -> list:
    """Đối chứng (a): mỗi dòng một khoản."""
    return [list(map(int, b)) for b in lines]


def items_fixed_gap(lines: list, gap_px: float = 15.0) -> list:
    """Đối chứng (b): khe cố định — khe dọc giữa hai dòng liên tiếp > gap_px thì mở khoản mới."""
    if not lines:
        return []
    ls = sorted(lines, key=lambda b: b[1])
    items, cur = [], [ls[0]]
    for prev, b in zip(ls, ls[1:]):
        if b[1] - (prev[1] + prev[3]) > gap_px:
            items.append(union(cur))
            cur = []
        cur.append(b)
    items.append(union(cur))
    return items


def items_fixed_count(lines: list, n: int = 2) -> list:
    """Đối chứng (c): số dòng cố định mỗi khoản."""
    ls = sorted(lines, key=lambda b: b[1])
    return [union(ls[i:i + n]) for i in range(0, len(ls), n)]


def drop_fake_items(binary: np.ndarray, items: list, h: float, min_height_ratio: float = 0.5,
                    min_ink: float = 0.02) -> list:
    """S-B4 hậu xử lý: bỏ khoản cao < 0,5h hoặc ít mực (lọc vùng theo diện tích, QĐ · S7). Opening 3×3 chạy trước
    bước tách dòng trong pipeline (C7 · S28, S34)."""
    out = []
    for x, y, w, hh in items:
        region = binary[y:y + hh, x:x + w]
        if hh >= min_height_ratio * h and region.size and (region > 0).mean() >= min_ink:
            out.append([x, y, w, hh])
    return out


def group(method: str, binary: np.ndarray, lines: list, h: float, **p) -> list:
    """Gọi một phương án gom khoản theo tên (dùng trong thí nghiệm S1)."""
    if method == "dilation_rule":
        return items_dilation_rule(binary, lines, h, p.get("s", 0.5), p.get("indent_tol", 1.0),
                                   p.get("bold_ratio", 1.5))
    if method == "dilation":
        return items_dilation(binary, lines, h, p.get("s", 0.5))
    if method == "closing":
        return items_closing(binary, lines, h, p.get("s", 0.5))
    if method == "each_line":
        return items_each_line(lines)
    if method == "fixed_gap":
        return items_fixed_gap(lines, p.get("gap_px", 15.0))
    if method == "fixed_count":
        return items_fixed_count(lines, p.get("n", 2))
    raise ValueError(method)
