"""C2 / C3 — VietOCR `vgg_transformer` (plan §5.3).

C2: trọng số pretrained của VietOCR (tự tải lần đầu).
C3: trọng số fine-tune trên VAIPE (code/weights/vietocr_finetuned.pth, train ở notebooks/finetune_vietocr.ipynb).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.paths import VIETOCR_FINETUNED
from legacy.ocr.engines import OCRUnavailable

_PREDICTORS = {}


def _device(device):
    if device:
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_predictor(weights: Path | str | None = None, arch: str = "vgg_transformer", device=None,
                   beamsearch: bool = False):
    key = (str(weights), arch, device, beamsearch)
    if key in _PREDICTORS:
        return _PREDICTORS[key]
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
    except ImportError as e:
        raise OCRUnavailable("chưa cài vietocr (pip install -r requirements-dl.txt)") from e
    cfg = Cfg.load_config_from_name(arch)
    cfg["device"] = _device(device)
    cfg["predictor"]["beamsearch"] = beamsearch
    cfg["cnn"]["pretrained"] = False
    if weights is not None:
        if not Path(weights).exists():
            raise FileNotFoundError(f"không thấy trọng số VietOCR: {weights}")
        cfg["weights"] = str(weights)
    _PREDICTORS[key] = Predictor(cfg)
    return _PREDICTORS[key]


class VietOCREngine:
    name = "vietocr"

    def __init__(self, variant: str = "pretrained", weights: Path | str | None = None, device=None,
                 batch_size: int = 32):
        """variant: pretrained (C2) | finetuned (C3). `weights` ghi đè đường dẫn mặc định."""
        if variant == "finetuned" and weights is None:
            weights = VIETOCR_FINETUNED
        self.variant = variant
        self.batch_size = batch_size
        self.predictor = load_predictor(weights, device=device)

    @staticmethod
    def _pil(crop: np.ndarray):
        from PIL import Image
        rgb = crop if crop.ndim == 2 else crop[:, :, ::-1]
        return Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")

    def read(self, crop: np.ndarray) -> tuple:
        if crop.size == 0:
            return "", 0.0
        text, prob = self.predictor.predict(self._pil(crop), return_prob=True)
        return text, float(prob)

    def read_batch(self, crops: list) -> list:
        out = [("", 0.0)] * len(crops)
        idx = [i for i, c in enumerate(crops) if c.size]
        for s in range(0, len(idx), self.batch_size):
            chunk = idx[s:s + self.batch_size]
            imgs = [self._pil(crops[i]) for i in chunk]
            try:
                texts, probs = self.predictor.predict_batch(imgs, return_prob=True)
            except (AttributeError, TypeError):   # bản vietocr cũ không có predict_batch(return_prob)
                texts, probs = zip(*[self.predictor.predict(im, return_prob=True) for im in imgs])
            for i, t, p in zip(chunk, texts, probs):
                out[i] = (t, float(p))
        return out
