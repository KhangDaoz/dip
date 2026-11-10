"""Hàm dùng chung cho các thí nghiệm có OCR (T2-CER, S3, D6, T8). OCR chỉ là CÔNG CỤ ĐO (KeHoach.md mục 1.3).

* reading_order: sắp box theo thứ tự đọc (gom hàng theo tâm y, trong hàng trái → phải).
* token_prf: precision / recall / F1 trên tập từ có lặp (không phân biệt hoa thường) — đo nhận dạng mà không phạt
  thứ tự đọc; dùng kèm CER toàn tập (ocr.metrics.corpus_cer) như K5.
"""
from collections import Counter

import numpy as np

from ocr.metrics import _norm

DRUG_LABELS = ("drugname", "quantity", "usage")


def reading_order(boxes: list, key=lambda b: b) -> list:
    """boxes: danh sách phần tử; key(e) trả về [x, y, w, h]. Hai box cùng hàng khi tâm y lệch < 0,5 × chiều cao trung vị."""
    if not boxes:
        return []
    hs = np.median([key(b)[3] for b in boxes])
    items = sorted(boxes, key=lambda b: key(b)[1] + key(b)[3] / 2)
    rows, cur, cy = [], [], None
    for b in items:
        c = key(b)[1] + key(b)[3] / 2
        if cur and abs(c - cy) > 0.5 * hs:
            rows.append(cur)
            cur = []
        cur.append(b)
        cy = np.mean([key(e)[1] + key(e)[3] / 2 for e in cur])
    rows.append(cur)
    return [b for r in rows for b in sorted(r, key=lambda e: key(e)[0])]


def xyxy_to_xywh(b):
    return [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]


def tokens(s: str) -> Counter:
    return Counter(_norm(s).lower().split())


def token_counts(pred: str, gt: str) -> tuple:
    """(số từ khớp, số từ dự đoán, số từ đáp án)."""
    p, g = tokens(pred), tokens(gt)
    return sum((p & g).values()), sum(p.values()), sum(g.values())


def token_prf(tp: int, n_pred: int, n_gt: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"token_precision": p, "token_recall": r, "token_f1": 2 * p * r / (p + r) if p + r else 0.0}


def crop(img: np.ndarray, box: list, pad: int = 3) -> np.ndarray:
    x, y, w, h = box
    H, W = img.shape[:2]
    return img[max(y - pad, 0):min(y + h + pad, H), max(x - pad, 0):min(x + w + pad, W)]


def ink_to_page(binary: np.ndarray) -> np.ndarray:
    """Ảnh nhị phân của nhóm (mực = 255) → chữ đen trên nền trắng cho Tesseract."""
    return 255 - binary
