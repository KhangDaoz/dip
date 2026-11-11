"""Khối B — phương pháp chính: YOLOv8 6 lớp (plan §5.2). Weights train trên Kaggle → code/weights/yolo_vaipe.pt."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.contracts import BOX_LABELS, TextBox
from common.paths import YOLO_WEIGHTS

_MODELS = {}


class WeightsMissing(FileNotFoundError):
    """Chưa có trọng số YOLO (chạy notebooks/train_yolo.ipynb hoặc scripts/download_weights.sh)."""


def load_model(weights: Path | str = YOLO_WEIGHTS):
    weights = Path(weights)
    key = str(weights.resolve())
    if key not in _MODELS:
        if not weights.exists():
            raise WeightsMissing(f"không thấy {weights}")
        from ultralytics import YOLO   # import muộn: module khác không cần cài ultralytics
        _MODELS[key] = YOLO(str(weights))
    return _MODELS[key]


def detect_yolo(img: np.ndarray, weights: Path | str = YOLO_WEIGHTS, conf: float = 0.25, iou: float = 0.5,
                imgsz: int = 1024, device=None, class_agnostic: bool = False) -> list:
    """Ảnh BGR → list[TextBox] (box xyxy pixel ảnh gốc, label 6 lớp, score = conf).

    conf chọn trên valid (tối đa F1); class_agnostic=True bỏ nhãn lớp (dùng cho E-B1).
    """
    model = load_model(weights)
    res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
    out = []
    if res.boxes is None:
        return out
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    score = res.boxes.conf.cpu().numpy()
    for b, c, s in zip(xyxy, cls, score):
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        label = None if class_agnostic else BOX_LABELS[c]
        out.append(TextBox(box=[x0, y0, max(x1, x0), max(y1, y0)], label=label, score=float(s)))
    out.sort(key=lambda t: (t.box[1], t.box[0]))
    return out
