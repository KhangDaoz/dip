"""Khoanh vùng bảng thuốc. Thí nghiệm S4."""
import re

import cv2
import numpy as np

from .lines import lines_projection

# Dấu hiệu dòng thuốc: số thứ tự đầu dòng hoặc đơn vị / dạng bào chế (song ngữ Việt – Anh)
MEDICINE_RE = re.compile(
    r"^\s*\(?\d{1,2}\s*[\)\.\/]|\b\d+(?:[.,]\d+)?\s*(mg|mcg|g|ml|iu)\b|\b(vi[eê]n|[oố]ng|g[oó]i|chai|tu[yý]p|"
    r"tab(let)?|cap(sule)?|syr(up)?|inj|drops?|sl\s*:)",
    re.IGNORECASE)


def text_height(binary) -> float:
    """Chiều cao chữ ước lượng = trung vị chiều cao thành phần liên thông (bỏ nhiễu)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((np.asarray(binary) > 0).astype(np.uint8), connectivity=8)
    hs = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 8]
    return float(np.median(hs)) if hs else 0.0


def text_blocks(binary, se_factor: float = 2.0) -> list:
    """Dilation SE ≈ se_factor × chiều cao chữ gom chữ thành khối văn bản; trả về list bbox khối."""
    h = text_height(binary)
    if h == 0:
        return []
    k = max(3, int(round(se_factor * h)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, k))   # gom ngang mạnh hơn dọc
    merged = cv2.dilate((np.asarray(binary) > 0).astype(np.uint8) * 255, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = merged.shape
    blocks = []
    for c in contours:
        x, y, w, bh = cv2.boundingRect(c)
        # trả về theo khung mực thật (bỏ phần nở do dilation)
        sub = np.asarray(binary)[y:y + bh, x:x + w] > 0
        ys, xs = np.nonzero(sub)
        if len(xs):
            blocks.append([int(x + xs.min()), int(y + ys.min()), int(xs.max() - xs.min() + 1),
                           int(ys.max() - ys.min() + 1)])
    return sorted(blocks, key=lambda b: b[1])


def find_medicine_region(binary, se_factor: float = 2.0, ocr_line=None):
    """Trả về bbox [x, y, w, h] vùng thuốc trên ảnh nhị phân (mực = 255), hoặc None.

    * Có `ocr_line(crop) -> str`: chọn khối có nhiều dòng mang dấu hiệu thuốc nhất (số thứ tự, mg, viên, tablet…).
    * Không có OCR: chọn khối có nhiều dòng nhất trong các khối rộng ≥ 1/3 trang (bảng thuốc là khối nhiều dòng
      liên tục nhất trên đơn); hòa thì lấy khối nhiều mực hơn.
    """
    binary = np.asarray(binary)
    blocks = text_blocks(binary, se_factor)
    if not blocks:
        return None
    W = binary.shape[1]
    best, best_score = None, None
    for b in blocks:
        x, y, w, h = b
        crop = binary[y:y + h, x:x + w]
        lines = lines_projection(crop, min_gap=2)
        if ocr_line is not None:
            hits = sum(bool(MEDICINE_RE.search(ocr_line(crop[ly:ly + lh, lx:lx + lw]) or ""))
                       for lx, ly, lw, lh in lines)
            score = (hits, len(lines))
        else:
            score = (w >= W / 3, len(lines), int((crop > 0).sum()))
        if best_score is None or score > best_score:
            best, best_score = b, score
    return best
