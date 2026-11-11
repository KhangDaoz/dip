"""Độ đo khoản thuốc (plan_Thang.md mục 5.1).

Khoản dự đoán KHỚP khoản đáp án khi CER(tên) ≤ 0,1 và quantity bằng nhau.
Tên so sau NFC + gộp khoảng trắng + bỏ tiền tố "N)" + casefold.
Ghép tham lam theo CER tăng dần, mỗi khoản đáp án ghép tối đa 1 lần. Cộng TP/FP/FN cả tập (micro).
Khoảng tin cậy: bootstrap theo ảnh.
"""
from __future__ import annotations

import numpy as np

from assemble.rules import clean_name
from common.boxes import precision_recall_f1
from legacy.ocr.metrics import cer

CER_THR = 0.1


def _get(it, key):
    return it.get(key) if isinstance(it, dict) else getattr(it, key, None)


def norm_name(s: str) -> str:
    return clean_name(s or "").casefold()


def match_items(pred: list, gt: list, cer_thr: float = CER_THR, name_only: bool = False) -> tuple[int, int, int]:
    """→ (tp, fp, fn) cho một ảnh."""
    pairs = []
    for i, p in enumerate(pred):
        pn = norm_name(_get(p, "name"))
        for j, g in enumerate(gt):
            gn = norm_name(_get(g, "name"))
            c = cer(pn, gn) if gn else (0.0 if not pn else 1.0)
            if c > cer_thr:
                continue
            if not name_only and _get(p, "quantity") != _get(g, "quantity"):
                continue
            pairs.append((c, i, j))
    used_p, used_g = set(), set()
    for _, i, j in sorted(pairs):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    tp = len(used_p)
    return tp, len(pred) - tp, len(gt) - tp


def items_prf(preds: dict, gts: dict, cer_thr: float = CER_THR, name_only: bool = False) -> dict:
    """preds, gts: {image_id: list[item]} (item là dict hoặc Item). Ảnh thiếu dự đoán tính là rỗng."""
    per_image = {}
    for k, g in gts.items():
        per_image[k] = match_items(preds.get(k, []), g, cer_thr, name_only)
    tp, fp, fn = (sum(v[i] for v in per_image.values()) for i in range(3))
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    exact = np.mean([v[1] == 0 and v[2] == 0 for v in per_image.values()]) if per_image else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1,
            "exact_rate": float(exact), "per_image": per_image}


def bootstrap_ci(per_image: dict, n: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Khoảng tin cậy (1 − alpha) cho F1 micro, lấy mẫu lại theo ảnh."""
    if not per_image:
        return 0.0, 0.0
    arr = np.array(list(per_image.values()), dtype=float)      # (N, 3) = tp, fp, fn
    rng = np.random.default_rng(seed)
    f1s = []
    for _ in range(n):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        f1s.append(precision_recall_f1(*s)[2])
    lo, hi = np.quantile(f1s, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def date_accuracy(pred_dates: dict, gt_dates: dict) -> float:
    """Tỉ lệ ảnh có ngày GT mà dự đoán trùng (chỉ xét ảnh GT có ngày)."""
    keys = [k for k, v in gt_dates.items() if v]
    if not keys:
        return 0.0
    return float(np.mean([pred_dates.get(k) == gt_dates[k] for k in keys]))
