"""Khối C — nhận dạng chữ trên từng TextBox.

    engine = get_engine("C2")           # C0 | C1 | C2 | C3
    boxes = recognize(img, boxes, engine)
"""
from __future__ import annotations

import numpy as np

from .crop import crop_box

ENGINES = {
    "C0": ("tesseract", {"mode": "gray"}),
    "C1": ("tesseract", {"mode": "sauvola"}),
    "C2": ("vietocr", {"variant": "pretrained"}),
    "C3": ("vietocr", {"variant": "finetuned"}),
}


def get_engine(name: str = "C2", **kw):
    kind, default = ENGINES[name]
    kw = {**default, **kw}
    if kind == "tesseract":
        from .tesseract_engine import TesseractEngine
        return TesseractEngine(**kw)
    from .vietocr_engine import VietOCREngine
    return VietOCREngine(**kw)


def recognize(img: np.ndarray, boxes: list, engine, pad: int = 2) -> list:
    """Điền text/text_conf cho từng TextBox (sửa tại chỗ, trả lại list). Tesseract phóng ×2, VietOCR giữ nguyên."""
    scale = getattr(engine, "scale", 1.0)
    crops = [crop_box(img, b.box, pad=pad, scale=scale) for b in boxes]
    for b, (t, c) in zip(boxes, engine.read_batch(crops)):
        b.text, b.text_conf = t.strip(), max(0.0, min(1.0, c))
    return boxes
