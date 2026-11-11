"""S-B1 khoanh vùng thuốc (KeHoach.md mục 8.1)."""
import cv2
import numpy as np

from .items import _groups_from_mask, _line_strips, bold_flags, union
from .lines import lines_projection


def text_blocks(binary: np.ndarray, h: float, k: float = 2.0, lines: list = None) -> list:
    """Khối văn bản = các dòng gần nhau theo chiều dọc. Mỗi dòng tô thành dải đặc (≈ Dilation ngang dài), rồi Dilation
    dọc SE k·h (C7 · S19, S22–24): dòng cách nhau < k·h dính thành một khối. Trả về list (chỉ số dòng, bbox)."""
    ls = lines if lines is not None else lines_projection(binary, h)
    if not ls:
        return []
    se = np.ones((max(1, int(round(k * h))), 1), np.uint8)
    mask = cv2.dilate(_line_strips(binary.shape, ls), se)
    return [(idx, union([ls[i] for i in idx])) for idx, _ in _groups_from_mask(mask, ls)]


def _gap(a, b):
    return b[1] - (a[1] + a[3])


def item_runs(lines: list, bness: list, h: float, k: float = 2.0, bold_ratio: float = 1.5,
              max_follow: int = 1, min_head: float = 0.6) -> list:
    """Chuỗi các cặp liền nhau "dòng đầu + 1..max_follow dòng theo sau".

    Dòng i là ĐẦU khoản nếu đậm hơn dòng ngay dưới ≥ bold_ratio lần — so sánh cục bộ, không chia đậm / thường cho cả
    trang vì trang có tiêu đề, mã vạch, khung bảng còn đậm hơn tên thuốc. Hai dòng kế nhau cách ≤ k·h (Dilation dọc
    SE k·h nối được). Mỗi chuỗi bỏ các cặp có dòng đầu < min_head × dòng đầu đậm nhất (ví dụ "Cộng khoản: 2 khoản").
    Trả về list chuỗi, mỗi chuỗi là list chỉ số dòng."""
    n = len(lines)
    head = [i + 1 < n and bness[i] >= bold_ratio * max(bness[i + 1], 1e-6) for i in range(n)]
    runs, cur, i = [], [], 0          # cur: list các cặp của chuỗi đang xét
    while i < n:
        if head[i] and _gap(lines[i], lines[i + 1]) <= k * h:
            pair = [i, i + 1]
            j = i + 2
            while (j < n and len(pair) < 1 + max_follow and not head[j]
                   and bness[j] < bness[i] / bold_ratio and _gap(lines[j - 1], lines[j]) <= k * h):
                pair.append(j)
                j += 1
            if cur and _gap(lines[cur[-1][-1]], lines[i]) > k * h:
                runs.append(cur)
                cur = []
            cur.append(pair)
            i = j
        else:
            if cur:
                runs.append(cur)
                cur = []
            i += 1
    if cur:
        runs.append(cur)
    out = []
    for pairs in runs:
        top = max(bness[p[0]] for p in pairs)
        kept = [i for p in pairs if bness[p[0]] >= min_head * top for i in p]
        if kept:
            out.append(kept)
    return out


def find_drug_region(binary: np.ndarray, h: float, k: float = 2.0, bold_ratio: float = 1.5,
                     max_follow: int = 1) -> list:
    """Vùng thuốc = chuỗi dài nhất các khoản liền nhau "dòng đậm (tên thuốc) + dòng thường (cách dùng)", hai dòng kế
    nhau cách ≤ k·h (Dilation dọc SE k·h, C7 · S19, S22–24). Dòng đậm nhận bằng phép co + Otsu (items.bold_flags).
    Hòa thì chọn chuỗi có dòng đậm rộng hơn (tên thuốc + "SL" trải gần hết bề ngang, tiêu đề thì hẹp).
    Không có dòng đậm → lấy cả trang (đối chứng). Trả về bbox [x, y, w, h]."""
    from .items import line_boldness
    H, W = binary.shape
    ls = lines_projection(binary, h)
    if len(ls) < 2:
        return [0, 0, W, H]
    bness = [line_boldness(binary, l) for l in ls]
    runs = item_runs(ls, bness, h, k, bold_ratio, max_follow)
    if not runs:
        return [0, 0, W, H]

    def score(run):
        heads = [i for i in run if i + 1 < len(ls) and bness[i] >= bold_ratio * max(bness[i + 1], 1e-6)]
        heads = heads or run[:1]
        return len(heads), float(np.mean([ls[i][2] for i in heads]))

    best = max(runs, key=score)
    return union([ls[i] for i in best])
