"""CER / WER (plan §6). Chuẩn hóa NFC + gộp khoảng trắng; phân biệt hoa/thường. Lõi lấy từ legacy/ocr/metrics.py."""
from __future__ import annotations

from legacy.ocr.metrics import _norm, cer, corpus_cer, edit_distance, wer  # noqa: F401


def corpus_wer(preds: list, gts: list) -> float:
    errors = sum(edit_distance(_norm(p).split(), _norm(g).split()) for p, g in zip(preds, gts))
    words = sum(len(_norm(g).split()) for g in gts)
    return errors / words if words else 0.0


def cer_by_label(preds: list, gts: list, labels: list) -> dict:
    """CER toàn tập theo từng lớp vùng chữ (drugname, quantity, ...) + 'all'."""
    out = {"all": corpus_cer(preds, gts)}
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        out[lab] = corpus_cer([preds[i] for i in idx], [gts[i] for i in idx])
    return out
