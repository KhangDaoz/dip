"""D1 — phân lớp vùng chữ bằng luật trên chữ OCR (plan §5.4). Dùng khi khối B là cổ điển (label=None).

So khớp không dấu, không phân biệt hoa/thường (strip_accents) để chịu được lỗi dấu của OCR.
"""
from __future__ import annotations

import re

from .text_utils import strip_accents

RULES = (
    ("drugname", re.compile(r"^\s*\d{1,2}\s*[\)\]]")),
    ("quantity", re.compile(r"^\s*s\s*l\s*[:.;]")),
    ("date", re.compile(r"^\s*ngay\s*\d")),
    ("diagnose", re.compile(r"^\s*ch[a-z]{0,2}n\s*do[a-z]{0,2}n")),
    ("usage", re.compile(r"^\s*(sang|trua|chieu|toi|uong|ngay uong)\b.*\d")),
)


def classify_text(text: str) -> str:
    t = strip_accents(text)
    for label, pat in RULES:
        if pat.search(t):
            return label
    return "other"


def classify_boxes(boxes: list, overwrite: bool = False) -> list:
    """Gán label cho TextBox chưa có nhãn (hoặc mọi box nếu overwrite). Sau đó nối dòng chẩn đoán thứ hai:
    box `other` sát dưới một box `diagnose` và thẳng mép trái (±1 chiều cao chữ) → `diagnose`."""
    for b in boxes:
        if overwrite or b.label is None:
            b.label = classify_text(b.text)
    diag = [b for b in boxes if b.label == "diagnose"]
    for d in diag:
        h = d.box[3] - d.box[1]
        for b in boxes:
            if b.label != "other":
                continue
            below = 0 <= b.box[1] - d.box[3] <= h
            aligned = abs(b.box[0] - d.box[0]) <= 2 * h or b.box[0] > d.box[0]
            if below and aligned and b.box[0] < d.box[2]:
                b.label = "diagnose"
    return boxes
