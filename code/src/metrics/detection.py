from __future__ import annotations

from collections import Counter

from common.boxes import greedy_match, precision_recall_f1
from common.contracts import BOX_LABELS


def _box(o):
    return o.box if hasattr(o, "box") else o["box"]


def _label(o):
    return o.label if hasattr(o, "label") else o.get("label")


def match_counts(gt: list, pred: list, iou_threshold: float = 0.5, by_class: bool = False) -> Counter:
    c = Counter()
    groups = BOX_LABELS if by_class else (None,)
    for lab in groups:
        g = [o for o in gt if lab is None or _label(o) == lab]
        p = [o for o in pred if lab is None or _label(o) == lab]
        tp = len(greedy_match([_box(o) for o in g], [_box(o) for o in p], iou_threshold))
        prefix = f"{lab}/" if lab else ""
        c[prefix + "tp"] += tp
        c[prefix + "fp"] += len(p) - tp
        c[prefix + "fn"] += len(g) - tp
        if lab:
            c["tp"] += tp
            c["fp"] += len(p) - tp
            c["fn"] += len(g) - tp
    return c


def evaluate(gts: list, preds: list, iou_threshold: float = 0.5, by_class: bool = False) -> dict:
    total = Counter()
    for g, p in zip(gts, preds):
        total += match_counts(g, p, iou_threshold, by_class)
    out = {"all": precision_recall_f1(total["tp"], total["fp"], total["fn"]), "counts": dict(total)}
    if by_class:
        for lab in BOX_LABELS:
            out[lab] = precision_recall_f1(total[f"{lab}/tp"], total[f"{lab}/fp"], total[f"{lab}/fn"])
    return out


def confusion_matrix(gts: list, preds: list, iou_threshold: float = 0.5):
    import numpy as np
    names = list(BOX_LABELS) + ["bg"]
    idx = {n: i for i, n in enumerate(names)}
    M = np.zeros((len(names), len(names)), dtype=int)
    for g, p in zip(gts, preds):
        pairs = greedy_match([_box(o) for o in g], [_box(o) for o in p], iou_threshold)
        mg, mp = {a for a, _, _ in pairs}, {b for _, b, _ in pairs}
        for a, b, _ in pairs:
            M[idx[_label(g[a])], idx[_label(p[b]) or "bg"]] += 1
        for a in set(range(len(g))) - mg:
            M[idx[_label(g[a])], idx["bg"]] += 1
        for b in set(range(len(p))) - mp:
            M[idx["bg"], idx[_label(p[b]) or "bg"]] += 1
    return M, names
