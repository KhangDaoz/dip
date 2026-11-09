"""CER / WER dùng chung cả nhóm. Báo cáo cả tập dùng corpus_cer (tổng lỗi / tổng ký tự)."""
import unicodedata


def _norm(s: str) -> str:
    """Chuẩn hóa Unicode NFC và gộp khoảng trắng — tiếng Việt có thể mã hóa dấu theo 2 cách."""
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def edit_distance(a, b) -> int:
    """Khoảng cách Levenshtein (chèn / xóa / thay), quy hoạch động O(len(a)·len(b))."""
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
    """Character Error Rate = số lỗi ký tự / số ký tự của đáp án."""
    pred, gt = _norm(pred), _norm(gt)
    if not gt:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, gt) / len(gt)


def wer(pred: str, gt: str) -> float:
    """Word Error Rate = số lỗi từ / số từ của đáp án."""
    p, g = _norm(pred).split(), _norm(gt).split()
    if not g:
        return 0.0 if not p else 1.0
    return edit_distance(p, g) / len(g)


def corpus_cer(preds: list, gts: list) -> float:
    """CER của cả tập = tổng số lỗi / tổng số ký tự (không lấy trung bình từng mẫu)."""
    errors = sum(edit_distance(_norm(p), _norm(g)) for p, g in zip(preds, gts))
    chars = sum(len(_norm(g)) for g in gts)
    return errors / chars if chars else 0.0
