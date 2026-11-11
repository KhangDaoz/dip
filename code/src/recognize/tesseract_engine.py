from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import cv2
import numpy as np

from common.io import to_gray
from enhance.binarize import sauvola

from .vietocr_engine import PAD, crop_box

SCALE = 2.0
BORDER = 10
WIN_PATHS = (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe")


class OCRUnavailable(RuntimeError):
    pass


def _tesseract():
    try:
        import pytesseract
    except ImportError as e:
        raise OCRUnavailable("chưa cài pytesseract (pip install pytesseract)") from e
    for cand in (os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"), *WIN_PATHS):
        if cand and os.path.exists(cand):
            pytesseract.pytesseract.tesseract_cmd = cand
            return pytesseract
    raise OCRUnavailable("chưa cài Tesseract 5 (Windows: https://github.com/UB-Mannheim/tesseract/wiki)")


@contextmanager
def _one_thread():
    old = os.environ.get("OMP_THREAD_LIMIT")
    os.environ["OMP_THREAD_LIMIT"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("OMP_THREAD_LIMIT", None)
        else:
            os.environ["OMP_THREAD_LIMIT"] = old


def available(lang: str = "vie") -> bool:
    try:
        return lang in _tesseract().get_languages(config="")
    except Exception:
        return False


def prepare(crop: np.ndarray, mode: str = "gray", scale: float = SCALE) -> np.ndarray:
    g = to_gray(crop)
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if mode == "binary":
        g = 255 - sauvola(g, window=31, k=0.2)
    elif mode != "gray":
        raise ValueError("mode phải là 'gray' (C0) hoặc 'binary' (C1)")
    return cv2.copyMakeBorder(g, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, value=255)


class TesseractEngine:

    def __init__(self, mode: str = "gray", lang: str = "vie", psm: int = 7, workers: int = 8):
        if mode not in ("gray", "binary"):
            raise ValueError("mode phải là 'gray' hoặc 'binary'")
        self.mode, self.lang, self.psm, self.workers = mode, lang, psm, workers
        self.name = "C0_tesseract" if mode == "gray" else "C1_tesseract_sauvola"
        self.tess = _tesseract()
        if lang not in self.tess.get_languages(config=""):
            raise OCRUnavailable(f"Tesseract chưa có gói ngôn ngữ '{lang}' (tải {lang}.traineddata vào tessdata)")

    def read(self, crop: np.ndarray) -> tuple:
        with _one_thread():
            return self._read(crop)

    def _read(self, crop: np.ndarray) -> tuple:
        if crop.size == 0:
            return "", 0.0
        d = self.tess.image_to_data(prepare(crop, self.mode), lang=self.lang, config=f"--psm {self.psm}",
                                    output_type=self.tess.Output.DICT)
        words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if w.strip() and float(c) >= 0]
        if not words:
            return "", 0.0
        return " ".join(w for w, _ in words), float(np.mean([c for _, c in words])) / 100.0

    def read_batch(self, crops: list) -> list:
        with _one_thread(), ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self._read, crops))

    def recognize(self, img_bgr: np.ndarray, text_boxes: list, pad: int = PAD) -> list:
        results = self.read_batch([crop_box(img_bgr, tb.box, pad) for tb in text_boxes])
        for tb, (text, conf) in zip(text_boxes, results):
            tb.text, tb.text_conf = text, min(max(conf, 0.0), 1.0)
        return text_boxes
