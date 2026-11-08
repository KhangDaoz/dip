"""Gom dòng thành từng khoản thuốc. Thí nghiệm S1.

Phương pháp đề xuất (kế hoạch mục 4.1): Dilation dọc nhỏ gom các dòng sát nhau thành khối, rồi tách khối bằng quy tắc
(dòng có số thứ tự ở lề trái, dòng lùi ra trái so với dòng trước, dòng tên thuốc cao hơn dòng liều).
Phương án so sánh: mỗi dòng là một khoản · khoảng cách dọc cố định (px) · mỗi khoản cố định N dòng.
"""
import re

import cv2
import numpy as np

# "1)", "1.", "2/", "10 -" ở đầu dòng — số thứ tự khoản thuốc
ORDINAL_RE = re.compile(r"^\s*\(?\d{1,2}\s*[\)\.\/:\-]")


def _union(boxes: list) -> list:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def _split(lines: list, starts: list) -> list:
    groups, cur = [], []
    for box, start in zip(lines, starts):
        if start and cur:
            groups.append(cur)
            cur = []
        cur.append(box)
    if cur:
        groups.append(cur)
    return [_union(g) for g in groups]


def _line_mask(lines: list) -> np.ndarray:
    """Mask vẽ từ bbox dòng — dùng khi không truyền ảnh nhị phân."""
    h = max(b[1] + b[3] for b in lines) + 1
    w = max(b[0] + b[2] for b in lines) + 1
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in lines:
        mask[y:y + bh, x:x + bw] = 255
    return mask


def dilation_starts(lines: list, binary=None, se_factor: float = 0.3) -> list:
    """Dilation dọc với SE cao se_factor × chiều cao chữ: hai dòng cách nhau < SE dính thành một khối.

    Trả về cờ "dòng mở đầu khối mới" cho từng dòng (lines đã sắp từ trên xuống).
    """
    if not lines:
        return []
    h_text = float(np.median([b[3] for b in lines]))
    k = max(1, int(round(se_factor * h_text)))
    mask = _line_mask(lines) if binary is None else (np.asarray(binary) > 0).astype(np.uint8) * 255
    # SE ngang 1 px: chỉ nở theo chiều dọc; khối = đoạn liên tiếp các hàng có mực sau dilation
    merged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    rows = merged.any(axis=1)
    block_id = np.cumsum(np.diff(np.concatenate([[False], rows]).astype(np.int8)) == 1)
    centers = [min(len(block_id) - 1, b[1] + b[3] // 2) for b in lines]
    return [i == 0 or block_id[centers[i]] != block_id[centers[i - 1]] for i in range(len(lines))]


def rule_starts(lines: list, texts: list = None, indent_tol: float = 1.0, height_ratio: float = 1.25) -> list:
    """Quy tắc mở khoản mới.

    * Có text OCR và ít nhất một dòng có số thứ tự → dòng có số thứ tự mở khoản.
    * Không có → dòng lùi ra trái so với dòng trước > indent_tol × chiều cao chữ (hết phần thụt lề),
      hoặc cao hơn dòng trước > height_ratio lần (dòng tên thuốc thường in đậm / chữ to hơn dòng liều).
    """
    n = len(lines)
    if n == 0:
        return []
    if texts is not None:
        ordinal = [bool(ORDINAL_RE.match(t or "")) for t in texts]
        if any(ordinal):
            ordinal[0] = True
            return ordinal
    h_text = float(np.median([b[3] for b in lines]))
    starts = [True]
    for prev, cur in zip(lines, lines[1:]):
        dedent = cur[0] < prev[0] - indent_tol * h_text
        taller = cur[3] > height_ratio * prev[3]
        starts.append(dedent or taller)
    return starts


def group_items_rules(lines: list, texts: list = None, binary=None, se_factor: float = 0.3,
                      indent_tol: float = 1.0, height_ratio: float = 1.25) -> list:
    """Phương pháp đề xuất: Dilation dọc gom khối + quy tắc tách khoản. Trả về list bbox khoản.

    binary: ảnh nhị phân cùng hệ tọa độ với lines (mực = 255); None → dilation trên mask vẽ từ bbox dòng.
    Có text OCR mang số thứ tự → tách theo số thứ tự (tin cậy nhất, xử lý được đơn giãn dòng đều nhau).
    """
    if not lines:
        return []
    by_rules = rule_starts(lines, texts, indent_tol, height_ratio)
    if texts is not None and any(ORDINAL_RE.match(t or "") for t in texts):
        return _split(lines, by_rules)
    by_dilation = dilation_starts(lines, binary, se_factor)
    return _split(lines, [a or b for a, b in zip(by_dilation, by_rules)])


def group_items_dilation(lines: list, binary=None, se_factor: float = 0.3) -> list:
    """Ablation: chỉ Dilation dọc, không có quy tắc."""
    return _split(lines, dilation_starts(lines, binary, se_factor)) if lines else []


def group_items_fixed(lines: list, lines_per_item: int = 2) -> list:
    """Phương án đối chứng: mỗi khoản = lines_per_item dòng liên tiếp (lines_per_item = 1: mỗi dòng là một khoản)."""
    return [_union(lines[i:i + lines_per_item]) for i in range(0, len(lines), lines_per_item)]


def group_items_fixed_gap(lines: list, gap_px: int = 10) -> list:
    """Phương án đối chứng: khoảng cách dọc cố định — dòng cách dòng trên > gap_px px thì mở khoản mới."""
    if not lines:
        return []
    starts = [True] + [cur[1] - (prev[1] + prev[3]) > gap_px for prev, cur in zip(lines, lines[1:])]
    return _split(lines, starts)
