from __future__ import annotations

import re
import unicodedata

import numpy as np

from metrics.items import bootstrap_ci, evaluate

_INDEX_PREFIX = re.compile(r"^\s*\d{1,2}\s*[)\]}l.]\s*")


def _get(it, k):
    return it.get(k) if isinstance(it, dict) else getattr(it, k, None)


def norm_name(s) -> str:
    s = unicodedata.normalize("NFC", s or "")
    return " ".join(_INDEX_PREFIX.sub("", s).split()).casefold()


def for_scoring(items: list) -> list[dict]:
    out = []
    for it in items:
        lm = _get(it, "lexicon_match")
        out.append({"name": norm_name(_get(it, "name")), "lexicon_match": norm_name(lm) if lm else None,
                    "quantity": _get(it, "quantity")})
    return out


def evaluate_d(gt: dict, pred: dict, use_lexicon: bool = True) -> tuple:
    return evaluate({k: for_scoring(v) for k, v in gt.items()},
                    {k: for_scoring(v) for k, v in pred.items()}, use_lexicon=use_lexicon)


def date_accuracy(pred_dates: dict, gt_dates: dict) -> float:
    keys = [k for k, v in gt_dates.items() if v]
    if not keys:
        return 0.0
    return float(np.mean([pred_dates.get(k) == gt_dates[k] for k in keys]))


__all__ = ["norm_name", "for_scoring", "evaluate_d", "date_accuracy", "bootstrap_ci"]
