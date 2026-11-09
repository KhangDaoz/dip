"""Độ đo cho S1–S4 (KeHoach.md mục 8.3): IoU box, khớp khoản thuốc, Precision / Recall / F1 (C8 · S48–49)."""
import numpy as np


def iou(a: list, b: list) -> float:
    """IoU hai box [x, y, w, h] = diện tích giao / diện tích hợp."""
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(pred: list, gt: list, thr: float = 0.5) -> dict:
    """Ghép tham lam theo IoU giảm dần, mỗi box chỉ ghép một lần; cặp IoU > thr là TP.

    Khoản dự đoán thừa = FP (ví dụ dòng hành chính bị gom nhầm); khoản thật bị bỏ sót / gộp = FN.
    """
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, ious = set(), set(), []
    for v, i, j in pairs:
        if v <= thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(v)
    tp = len(ious)
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp, "ious": ious}


def prf(tp: int, fp: int, fn: int) -> tuple:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def summarize(results: list) -> dict:
    """Gộp kết quả nhiều ảnh: P, R, F1 tính trên tổng TP / FP / FN (micro), IoU trung bình các cặp khớp."""
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    p, r, f = prf(tp, fp, fn)
    ious = [v for r in results for v in r["ious"]]
    return {"precision": p, "recall": r, "f1": f, "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "tp": tp, "fp": fp, "fn": fn}
