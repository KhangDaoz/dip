from __future__ import annotations

import numpy as np

from common.boxes import precision_recall_f1

from .ocr import cer

NAME_CER = 0.1


def _get(it, k):
    return getattr(it, k) if not isinstance(it, dict) else it.get(k)


def pred_name(it, use_lexicon: bool = True) -> str:
    lm = _get(it, "lexicon_match") if use_lexicon else None
    return lm or _get(it, "name") or ""


def match_items(gt_items: list, pred_items: list, check_quantity: bool = True, name_cer: float = NAME_CER,
                use_lexicon: bool = True) -> list:
    cands = []
    for gi, g in enumerate(gt_items):
        for pi, p in enumerate(pred_items):
            c = cer(pred_name(p, use_lexicon), _get(g, "name") or "")
            if c > name_cer:
                continue
            if check_quantity and _get(g, "quantity") != _get(p, "quantity"):
                continue
            cands.append((c, gi, pi))
    pairs, ug, up = [], set(), set()
    for c, gi, pi in sorted(cands):
        if gi in ug or pi in up:
            continue
        pairs.append((gi, pi, c))
        ug.add(gi)
        up.add(pi)
    return pairs


def image_counts(gt_items: list, pred_items: list, use_lexicon: bool = True) -> dict:
    full = match_items(gt_items, pred_items, True, use_lexicon=use_lexicon)
    name = match_items(gt_items, pred_items, False, use_lexicon=use_lexicon)
    qty_ok = sum(_get(gt_items[g], "quantity") == _get(pred_items[p], "quantity") for g, p, _ in name)
    ng, np_ = len(gt_items), len(pred_items)
    return {"tp": len(full), "fp": np_ - len(full), "fn": ng - len(full),
            "name_tp": len(name), "name_fp": np_ - len(name), "name_fn": ng - len(name),
            "qty_ok": qty_ok, "n_gt": ng, "exact": int(len(full) == ng == np_)}


def aggregate(counts: list) -> dict:
    s = {k: sum(c[k] for c in counts) for k in counts[0]} if counts else {}
    if not s:
        return {"item_p": 0.0, "item_r": 0.0, "item_f1": 0.0, "name_f1": 0.0, "qty_acc": 0.0, "exact_rate": 0.0, "n_images": 0}
    p, r, f1 = precision_recall_f1(s["tp"], s["fp"], s["fn"])
    _, _, nf1 = precision_recall_f1(s["name_tp"], s["name_fp"], s["name_fn"])
    return {"item_p": p, "item_r": r, "item_f1": f1, "name_f1": nf1,
            "qty_acc": s["qty_ok"] / s["n_gt"] if s["n_gt"] else 0.0,
            "exact_rate": s["exact"] / len(counts), "n_images": len(counts),
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]}


def evaluate(gt: dict, pred: dict, use_lexicon: bool = True) -> tuple:
    ids = sorted(gt)
    counts = [image_counts(gt[i], pred.get(i, []), use_lexicon) for i in ids]
    return aggregate(counts), counts


def bootstrap_ci(counts: list, metric: str = "item_f1", n: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    base = aggregate(counts)[metric]
    if not counts:
        return base, base, base
    vals = [aggregate([counts[j] for j in rng.integers(0, len(counts), len(counts))])[metric] for _ in range(n)]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(base), float(lo), float(hi)
