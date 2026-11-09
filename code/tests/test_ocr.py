"""Bộ OCR dùng chung."""
from common.testing import todo
from ocr import metrics


@todo
def test_cer_basic():
    assert metrics.cer("abc", "abc") == 0.0
    assert metrics.cer("abd", "abc") == 1 / 3
