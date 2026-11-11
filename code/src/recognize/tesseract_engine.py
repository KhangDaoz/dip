"""Khối C — đối chứng cổ điển: Tesseract 5 `vie` (plan §5.3).

* C0: ảnh xám, phóng ×2 nội suy bậc ba (ch. 2), --psm 7 (một dòng chữ).
* C1: như C0 nhưng đầu vào nhị phân Sauvola (ch. 5) — kiểm tra lại kết quả v1 "ảnh xám tốt hơn nhị phân".

Cùng giao diện với VietOCREngine:

    from recognize.tesseract_engine import TesseractEngine
    ocr = TesseractEngine("gray")                    # C0; "binary" → C1
    ocr.read(crop_bgr) -> (text, conf)
    ocr.read_batch([crop, ...]) -> [(text, conf), ...]
    ocr.recognize(img_bgr, text_boxes)               # điền TextBox.text / text_conf tại chỗ

Cần cài Tesseract 5 + gói ngôn ngữ `vie` (Windows: bản UB-Mannheim) và `pip install pytesseract`.
Không tìm thấy thì ném OCRUnavailable. Đường dẫn tự dò: biến môi trường TESSERACT_CMD → PATH → thư mục cài mặc định.
"""
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
BORDER = 10        # viền trắng quanh vùng cắt — Tesseract nhận dòng kém khi chữ sát mép
WIN_PATHS = (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe")


class OCRUnavailable(RuntimeError):
    """Máy chưa cài Tesseract / pytesseract / gói ngôn ngữ."""


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
    """Tesseract tự mở nhiều luồng OpenMP; chạy song song nhiều vùng thì mỗi tiến trình con chỉ nên dùng 1 luồng.
    Chỉ đặt OMP_THREAD_LIMIT trong lúc gọi rồi trả lại — đặt toàn cục sẽ bóp torch (VietOCR) xuống 1 luồng."""
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
    """Vùng BGR → ảnh xám (C0) hoặc nhị phân chữ đen nền trắng (C1), phóng `scale` lần, thêm viền trắng."""
    g = to_gray(crop)
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if mode == "binary":
        g = 255 - sauvola(g, window=31, k=0.2)            # sauvola: mực = 255 → đảo thành chữ đen
    elif mode != "gray":
        raise ValueError("mode phải là 'gray' (C0) hoặc 'binary' (C1)")
    return cv2.copyMakeBorder(g, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, value=255)


class TesseractEngine:
    """mode: 'gray' (C0, mặc định) | 'binary' (C1)."""

    def __init__(self, mode: str = "gray", lang: str = "vie", psm: int = 7, workers: int = 8):
        if mode not in ("gray", "binary"):
            raise ValueError("mode phải là 'gray' hoặc 'binary'")
        self.mode, self.lang, self.psm, self.workers = mode, lang, psm, workers
        self.name = "C0_tesseract" if mode == "gray" else "C1_tesseract_sauvola"
        self.tess = _tesseract()
        if lang not in self.tess.get_languages(config=""):
            raise OCRUnavailable(f"Tesseract chưa có gói ngôn ngữ '{lang}' (tải {lang}.traineddata vào tessdata)")

    def read(self, crop: np.ndarray) -> tuple:
        """Một vùng BGR → (text, conf ∈ [0, 1]); conf = trung bình độ tin cậy các từ."""
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
        """Nhiều vùng → [(text, conf)]. Mỗi vùng một tiến trình tesseract, chạy song song `workers` luồng."""
        with _one_thread(), ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self._read, crops))

    def recognize(self, img_bgr: np.ndarray, text_boxes: list, pad: int = PAD) -> list:
        """Điền `text`, `text_conf` cho list[TextBox] (sửa tại chỗ) và trả về chính list đó."""
        results = self.read_batch([crop_box(img_bgr, tb.box, pad) for tb in text_boxes])
        for tb, (text, conf) in zip(text_boxes, results):
            tb.text, tb.text_conf = text, min(max(conf, 0.0), 1.0)
        return text_boxes
