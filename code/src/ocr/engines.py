"""Bọc OCR. Tiếng Anh (bộ 2): Tesseract 5. Tiếng Việt (bộ 1, bộ 4): Tesseract `vie` hoặc VietOCR.

Mọi hàm trả về (text, confidence ∈ [0, 1]). Import thư viện OCR bên trong hàm (lazy) để
các module khác không phải cài requirements-ocr.txt. OCR chỉ là CÔNG CỤ ĐO CER (KeHoach.md mục 1.3).

Thiếu engine thì ném OCRUnavailable; các pipeline bắt lỗi này và bỏ qua bước OCR.
"""
import os
import shutil

import numpy as np

_CACHE = {}


class OCRUnavailable(RuntimeError):
    """Máy chưa cài engine OCR được yêu cầu."""


def available(engine: str = "tesseract") -> bool:
    try:
        _get_engine(engine)
        return True
    except OCRUnavailable:
        return False


def _tesseract_cmd():
    for cand in (os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
                 r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                 r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if cand and os.path.exists(cand):
            return cand
    return None


def _get_engine(engine: str):
    if engine in _CACHE:
        return _CACHE[engine]
    if engine == "tesseract":
        try:
            import pytesseract
        except ImportError as e:
            raise OCRUnavailable("chưa cài pytesseract (pip install pytesseract)") from e
        cmd = _tesseract_cmd()
        if cmd is None:
            raise OCRUnavailable("chưa cài Tesseract 5 (https://github.com/UB-Mannheim/tesseract/wiki)")
        pytesseract.pytesseract.tesseract_cmd = cmd
        _CACHE[engine] = pytesseract
    elif engine == "vietocr":
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as e:
            raise OCRUnavailable("chưa cài vietocr") from e
        cfg = Cfg.load_config_from_name("vgg_transformer")
        cfg["device"] = "cpu"
        _CACHE[engine] = Predictor(cfg)
    else:
        raise ValueError(f"engine không hỗ trợ: {engine}")
    return _CACHE[engine]


def _tess(img: np.ndarray, lang: str, psm: int) -> tuple:
    tess = _get_engine("tesseract")
    data = tess.image_to_data(img, lang=lang, config=f"--psm {psm}", output_type=tess.Output.DICT)
    words = [(w, float(c)) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
    if not words:
        return "", 0.0
    text = " ".join(w for w, _ in words)
    if psm in (3, 4, 6):  # giữ xuống dòng giữa các dòng khi OCR cả trang / khối
        lines = {}
        for w, c, b, p, ln in zip(data["text"], data["conf"], data["block_num"], data["par_num"], data["line_num"]):
            if w.strip() and float(c) >= 0:
                lines.setdefault((b, p, ln), []).append(w)
        text = "\n".join(" ".join(v) for v in lines.values())
    return text, float(np.mean([c for _, c in words])) / 100.0


def ocr_word(img_crop: np.ndarray, lang: str = "eng", engine: str = "tesseract") -> tuple:
    """OCR một cụm từ (Tesseract --psm 8)."""
    return _tess(img_crop, lang, 8)


def ocr_line(img_crop: np.ndarray, lang: str = "eng", engine: str = "tesseract") -> tuple:
    """OCR một dòng (Tesseract --psm 7; VietOCR cho engine="vietocr")."""
    if engine == "vietocr":
        from PIL import Image
        pred = _get_engine("vietocr")
        rgb = img_crop if img_crop.ndim == 2 else img_crop[:, :, ::-1]
        text, prob = pred.predict(Image.fromarray(rgb), return_prob=True)
        return text, float(prob)
    return _tess(img_crop, lang, 7)


def ocr_block(img: np.ndarray, lang: str = "eng") -> tuple:
    """OCR một khối nhiều dòng (một khoản thuốc, một ô) — Tesseract --psm 6."""
    return _tess(img, lang, 6)


def ocr_page(img: np.ndarray, lang: str = "eng", engine: str = "tesseract") -> tuple:
    """OCR cả trang — đối chứng "OCR cả trang" (K5, S3, ablation). Tesseract --psm 3."""
    return _tess(img, lang, 3)
