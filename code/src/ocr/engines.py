"""Bọc OCR. Tiếng Anh (bộ 2): Tesseract 5 / PaddleOCR. Tiếng Việt (bộ 1, bộ 4): VietOCR.

Mọi hàm trả về (text, confidence ∈ [0, 1]). Import thư viện OCR bên trong hàm (lazy) để
các module khác không phải cài requirements-ocr.txt.
"""
import numpy as np


def ocr_word(img_crop: np.ndarray, lang: str = "eng", engine: str = "tesseract") -> tuple:
    """OCR một cụm từ (Tesseract --psm 8)."""
    raise NotImplementedError("ocr_word")


def ocr_line(img_crop: np.ndarray, lang: str = "eng", engine: str = "tesseract") -> tuple:
    """OCR một dòng (Tesseract --psm 7; VietOCR cho lang="vie")."""
    raise NotImplementedError("ocr_line")


def ocr_page(img: np.ndarray, lang: str = "eng", engine: str = "tesseract") -> tuple:
    """OCR cả trang — đối chứng "OCR cả trang" (K5, S3, ablation)."""
    raise NotImplementedError("ocr_page")
