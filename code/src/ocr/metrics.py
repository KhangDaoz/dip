"""CER / WER dùng chung cả nhóm. Báo cáo cả tập dùng corpus_cer (tổng lỗi / tổng ký tự)."""


def cer(pred: str, gt: str) -> float:
    raise NotImplementedError("cer")


def wer(pred: str, gt: str) -> float:
    raise NotImplementedError("wer")


def corpus_cer(preds: list, gts: list) -> float:
    raise NotImplementedError("corpus_cer")
