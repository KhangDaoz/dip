"""Khối C — VietOCR `vgg_transformer` (plan §5.3).

* C3 (chính): trọng số fine-tune trên VAIPE — code/weights/vietocr_finetuned.pth
  (notebooks/finetune_vietocr.ipynb; tải: tools/download_weights.py).
* C2 (đối chứng): trọng số pretrained chính thức (tự tải từ vocr.vn lần đầu).

Cả hai nạp bằng configs/vietocr_vgg_transformer.yml với image_max_width = 1024 — PHẢI khớp lúc
fine-tune (mặc định VietOCR là 512, sẽ bóp méo ~5 % vùng dài như dòng chẩn đoán).
Vùng cắt giống lúc train: box chuẩn/dự đoán + đệm 2 px, không phóng (`crop_box`).

    from recognize.vietocr_engine import VietOCREngine
    ocr = VietOCREngine("finetuned")                 # hoặc "pretrained"
    ocr.read(crop_bgr) -> (text, prob)
    ocr.read_batch([crop, ...]) -> [(text, prob), ...]
    ocr.recognize(img_bgr, text_boxes)               # điền TextBox.text / text_conf tại chỗ
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from common.boxes import clip_box
from common.paths import VIETOCR_CONFIG, VIETOCR_FINETUNED

PRETRAINED_URL = "https://vocr.vn/data/vietocr/vgg_transformer.pth"
MAX_WIDTH = 1024
PAD = 2

_PREDICTORS: dict = {}


def _stub_imgaug():
    """vietocr import imgaug ở mức module nhưng chỉ dùng albumentations; imgaug 0.4 hỏng với NumPy 2."""
    try:
        import imgaug  # noqa: F401
    except Exception:
        stub = types.ModuleType("imgaug")
        stub.augmenters = types.ModuleType("imgaug.augmenters")
        sys.modules["imgaug"] = stub
        sys.modules["imgaug.augmenters"] = stub.augmenters


def _device(device):
    if device:
        return device
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_predictor(weights: Path | str = VIETOCR_FINETUNED, device=None, max_width: int = MAX_WIDTH,
                   config: Path | str = VIETOCR_CONFIG):
    """Nạp (và cache) vietocr Predictor. weights: đường dẫn .pth, hoặc PRETRAINED_URL cho C2."""
    weights = str(weights)
    device = _device(device)
    key = (weights, device, max_width)
    if key in _PREDICTORS:
        return _PREDICTORS[key]
    if not weights.startswith("http") and not Path(weights).exists():
        raise FileNotFoundError(f"không thấy {weights} — chạy: python tools/download_weights.py")
    _stub_imgaug()
    import os
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")   # torch>=2.6 + vietocr torch.load
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor
    cfg = Cfg.load_config_from_file(str(config))
    cfg["device"] = device
    cfg["weights"] = weights
    cfg["cnn"]["pretrained"] = False
    cfg["predictor"]["beamsearch"] = False
    cfg["dataset"]["image_max_width"] = max_width
    _PREDICTORS[key] = Predictor(cfg)
    return _PREDICTORS[key]


def crop_box(img_bgr: np.ndarray, box, pad: int = PAD) -> np.ndarray:
    """Cắt box xyxy với đệm `pad` px, kẹp trong ảnh — giống lúc tạo dữ liệu train (tools/to_vietocr.py)."""
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = clip_box(box, w, h)
    return img_bgr[max(y0 - pad, 0):min(y1 + pad, h), max(x0 - pad, 0):min(x1 + pad, w)]


def _pil(crop: np.ndarray):
    from PIL import Image
    rgb = np.stack([crop] * 3, -1) if crop.ndim == 2 else crop[:, :, ::-1]
    return Image.fromarray(np.ascontiguousarray(rgb))


class VietOCREngine:
    """variant: 'finetuned' (C3, mặc định) | 'pretrained' (C2). `weights` ghi đè đường dẫn."""

    def __init__(self, variant: str = "finetuned", weights: Path | str | None = None, device=None,
                 batch_size: int = 64):
        if variant not in ("finetuned", "pretrained"):
            raise ValueError("variant phải là 'finetuned' hoặc 'pretrained'")
        self.variant = variant
        self.name = "C3_finetune" if variant == "finetuned" else "C2_pretrained"
        self.batch_size = batch_size
        weights = weights or (VIETOCR_FINETUNED if variant == "finetuned" else PRETRAINED_URL)
        self.predictor = load_predictor(weights, device=device)

    def read(self, crop: np.ndarray) -> tuple:
        """Một vùng BGR → (text, prob)."""
        if crop.size == 0:
            return "", 0.0
        text, prob = self.predictor.predict(_pil(crop), return_prob=True)
        return text, float(prob)

    def read_batch(self, crops: list) -> list:
        """Nhiều vùng → [(text, prob)]. Gom theo tỉ lệ rộng/cao để mỗi batch ít kích thước (nhanh hơn)."""
        out = [("", 0.0)] * len(crops)
        idx = sorted((i for i, c in enumerate(crops) if c.size),
                     key=lambda i: crops[i].shape[1] / crops[i].shape[0])
        for s in range(0, len(idx), self.batch_size):
            chunk = idx[s:s + self.batch_size]
            texts, probs = self.predictor.predict_batch([_pil(crops[i]) for i in chunk], return_prob=True)
            for i, t, p in zip(chunk, texts, probs):
                out[i] = (t, float(p))
        return out

    def recognize(self, img_bgr: np.ndarray, text_boxes: list, pad: int = PAD) -> list:
        """Điền `text`, `text_conf` cho list[TextBox] (sửa tại chỗ) và trả về chính list đó."""
        results = self.read_batch([crop_box(img_bgr, tb.box, pad) for tb in text_boxes])
        for tb, (text, prob) in zip(text_boxes, results):
            tb.text, tb.text_conf = text, min(max(prob, 0.0), 1.0)
        return text_boxes
