"""Độ đo phát hiện vùng chữ (plan.md mục 6).

* IoU của 2 box xyxy.
* Ghép tham lam: sắp mọi cặp (GT, dự đoán) có IoU ≥ ngưỡng theo IoU giảm dần, mỗi box GT và mỗi box dự đoán ghép
  tối đa một lần. Có phân lớp: chỉ ghép cặp cùng nhãn.
* P/R/F1 tính trên TOÀN TẬP: cộng dồn TP/FP/FN của mọi ảnh rồi mới chia (không lấy trung bình từng ảnh).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


def iou_matrix(a, b) -> np.ndarray:
    """a: N×4, b: M×4 (xyxy) → N×M IoU."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def iou(a, b) -> float:
    return float(iou_matrix([a], [b])[0, 0])


def match_greedy(gt_boxes, pred_boxes, iou_thr: float = 0.5, gt_labels=None, pred_labels=None):
    """Trả về list cặp (i_gt, j_pred, iou). Truyền cả gt_labels và pred_labels thì chỉ ghép cặp cùng nhãn."""
    m = iou_matrix(gt_boxes, pred_boxes)
    if gt_labels is not None and pred_labels is not None:
        same = np.asarray(gt_labels, dtype=object)[:, None] == np.asarray(pred_labels, dtype=object)[None, :]
        m = np.where(same, m, 0.0)
    gi, pj = np.nonzero(m >= iou_thr)
    order = np.argsort(-m[gi, pj], kind="stable")
    used_g, used_p, pairs = set(), set(), []
    for k in order:
        i, j = int(gi[k]), int(pj[k])
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        pairs.append((i, j, float(m[i, j])))
    return pairs


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class DetectionCounter:
    """Cộng dồn TP/FP/FN qua các ảnh. class_aware=False: bỏ nhãn lớp khi ghép."""
    iou_thr: float = 0.5
    class_aware: bool = False
    tp: int = 0
    fp: int = 0
    fn: int = 0
    per_class: dict = field(default_factory=dict)   # nhãn -> [tp, fp, fn] (chỉ khi class_aware)

    def add(self, gt_boxes, pred_boxes, gt_labels=None, pred_labels=None):
        if self.class_aware:
            pairs = match_greedy(gt_boxes, pred_boxes, self.iou_thr, gt_labels, pred_labels)
            mg = {i for i, _, _ in pairs}
            mp = {j for _, j, _ in pairs}
            for i, lab in enumerate(gt_labels):
                self.per_class.setdefault(lab, [0, 0, 0])[0 if i in mg else 2] += 1
            for j, lab in enumerate(pred_labels):
                if j not in mp:
                    self.per_class.setdefault(lab, [0, 0, 0])[1] += 1
        else:
            pairs = match_greedy(gt_boxes, pred_boxes, self.iou_thr)
        self.tp += len(pairs)
        self.fp += len(pred_boxes) - len(pairs)
        self.fn += len(gt_boxes) - len(pairs)
        return pairs

    def result(self) -> dict:
        p, r, f = prf(self.tp, self.fp, self.fn)
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": p, "recall": r, "f1": f}

    def per_class_result(self) -> dict:
        out = {}
        for lab, (tp, fp, fn) in self.per_class.items():
            p, r, f = prf(tp, fp, fn)
            out[lab] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f, "support": tp + fn}
        return out


def confusion_counts(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thr: float = 0.5) -> Counter:
    """Ma trận nhầm lẫn theo vị trí: ghép KHÔNG phân lớp rồi đếm (nhãn GT, nhãn dự đoán).
    GT không ghép được → (nhãn, "background"); dự đoán thừa → ("background", nhãn)."""
    pairs = match_greedy(gt_boxes, pred_boxes, iou_thr)
    c = Counter((gt_labels[i], pred_labels[j]) for i, j, _ in pairs)
    mg = {i for i, _, _ in pairs}
    mp = {j for _, j, _ in pairs}
    c.update((gt_labels[i], "background") for i in range(len(gt_boxes)) if i not in mg)
    c.update(("background", pred_labels[j]) for j in range(len(pred_boxes)) if j not in mp)
    return c
