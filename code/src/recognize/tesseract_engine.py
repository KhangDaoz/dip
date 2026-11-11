"""C0 / C1 — Tesseract 5 `vie`, --psm 7 (một dòng). Đối chứng cổ điển (plan §5.3)."""
from __future__ import annotations

import numpy as np

from legacy.ocr.engines import OCRUnavailable, ocr_line  # noqa: F401  (OCRUnavailable re-export)

from .crop import prepare


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "vie", mode: str = "gray", scale: float = 2.0):
        """mode=gray → C0; mode=otsu/sauvola → C1. scale: phóng vùng cắt trước khi đọc."""
        self.lang, self.mode, self.scale = lang, mode, scale

    def read(self, crop: np.ndarray) -> tuple:
        if crop.size == 0:
            return "", 0.0
        return ocr_line(prepare(crop, self.mode), lang=self.lang)

    def read_batch(self, crops: list) -> list:
        return [self.read(c) for c in crops]
