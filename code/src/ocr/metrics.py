"""CER / WER — độ đo chung cả nhóm dùng để so sánh trước / sau xử lý.

Chuẩn hóa trước khi so: Unicode NFC (dấu tiếng Việt có 2 cách mã hóa), gộp khoảng trắng.
"""
import re
import unicodedata


def normalize_text(text: str, lower: bool = False) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lower else text


def edit_distance(ref, hyp) -> int:
    """Khoảng cách Levenshtein giữa 2 chuỗi (hoặc 2 list từ)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, lower: bool = False) -> float:
    """Character Error Rate = số phép sửa ký tự / số ký tự của ref."""
    ref, hyp = normalize_text(ref, lower), normalize_text(hyp, lower)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str, lower: bool = False) -> float:
    """Word Error Rate = số phép sửa từ / số từ của ref."""
    ref_w, hyp_w = normalize_text(ref, lower).split(), normalize_text(hyp, lower).split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return edit_distance(ref_w, hyp_w) / len(ref_w)


def corpus_cer(refs, hyps, lower: bool = False) -> float:
    """CER gộp cả tập (tổng lỗi / tổng ký tự) — dùng cái này khi báo cáo, không lấy trung bình CER từng ảnh."""
    errors = total = 0
    for r, h in zip(refs, hyps):
        r, h = normalize_text(r, lower), normalize_text(h, lower)
        errors += edit_distance(r, h)
        total += len(r)
    return errors / total if total else 0.0
