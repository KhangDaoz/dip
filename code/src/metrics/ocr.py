from __future__ import annotations

import unicodedata


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gt: str) -> float:
    pred, gt = _norm(pred), _norm(gt)
    if not gt:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, gt) / len(gt)


def wer(pred: str, gt: str) -> float:
    p, g = _norm(pred).split(), _norm(gt).split()
    if not g:
        return 0.0 if not p else 1.0
    return edit_distance(p, g) / len(g)


def corpus_cer(preds: list, gts: list) -> float:
    errors = sum(edit_distance(_norm(p), _norm(g)) for p, g in zip(preds, gts))
    chars = sum(len(_norm(g)) for g in gts)
    return errors / chars if chars else 0.0


def corpus_wer(preds: list, gts: list) -> float:
    errors = sum(edit_distance(_norm(p).split(), _norm(g).split()) for p, g in zip(preds, gts))
    words = sum(len(_norm(g).split()) for g in gts)
    return errors / words if words else 0.0


def cer_by_label(preds: list, gts: list, labels: list) -> dict:
    out = {"all": corpus_cer(preds, gts)}
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        out[lab] = corpus_cer([preds[i] for i in idx], [gts[i] for i in idx])
    return out
