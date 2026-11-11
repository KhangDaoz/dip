"""Khối B cổ điển (plan §5.2): ảnh tổng hợp của conftest — 12 dòng chữ, mỗi dòng một cụm."""
import numpy as np

from conftest import N_LINES
from detect.classical import char_height, detect_classical, remove_lines


def test_detects_each_line(page):
    boxes = detect_classical(page)
    assert all(b.label is None and b.score == 1.0 for b in boxes)
    assert N_LINES <= len(boxes) <= 2 * N_LINES
    ys = sorted({round((b.box[1] + b.box[3]) / 2 / 55) for b in boxes})
    assert len(ys) == N_LINES                      # mỗi dòng chữ có ít nhất một box


def test_boxes_are_valid_xyxy(page):
    for b in detect_classical(page):
        b.validate()


def test_remove_lines_keeps_text(page_gray, text_mask):
    ink = text_mask.copy()
    ink[400, :] = 255                              # đường kẻ ngang dài
    cleaned = remove_lines(ink)
    assert cleaned[400, :].sum() < 0.1 * ink[400, :].sum()
    assert cleaned.sum() > 0.8 * text_mask.sum()


def test_char_height_reasonable(text_mask):
    assert 8 <= char_height(text_mask) <= 30
