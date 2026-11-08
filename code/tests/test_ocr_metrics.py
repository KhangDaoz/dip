import unicodedata

import pytest

from ocr.metrics import cer, corpus_cer, edit_distance, wer


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3


def test_cer_basic():
    assert cer("abcd", "abcd") == 0
    assert cer("abcd", "abed") == pytest.approx(0.25)
    assert cer("", "") == 0


def test_cer_vietnamese_nfc_nfd_equal():
    text = "Viên uống sáng"
    assert cer(unicodedata.normalize("NFC", text), unicodedata.normalize("NFD", text)) == 0


def test_wer_and_whitespace():
    assert wer("uống  sau   ăn", "uống sau ăn") == 0
    assert wer("uống sau ăn", "uống trước ăn") == pytest.approx(1 / 3)


def test_corpus_cer_weights_by_length():
    assert corpus_cer(["a", "abcdefghij"], ["b", "abcdefghij"]) == pytest.approx(1 / 11)
