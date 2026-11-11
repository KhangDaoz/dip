from __future__ import annotations

from pathlib import Path

import numpy as np

from common.contracts import BOX_LABELS, TextBox
from common.paths import YOLO_WEIGHTS

IMGSZ = 1024
CONF = 0.443
IOU = 0.7

_MODELS: dict = {}


class WeightsMissing(FileNotFoundError):
    pass


def load_yolo(weights: Path | str = YOLO_WEIGHTS):
    weights = Path(weights)
    key = str(weights.resolve())
    if key not in _MODELS:
        if not weights.exists():
            raise WeightsMissing(f"không thấy {weights} — chạy: python tools/download_weights.py")
        from ultralytics import YOLO
        model = YOLO(str(weights))
        names = [model.names[i] for i in range(len(model.names))]
        if tuple(names) != BOX_LABELS:
            raise ValueError(f"lớp của {weights.name} là {names}, cần {BOX_LABELS}")
        _MODELS[key] = model
    return _MODELS[key]


def detect_yolo(img: np.ndarray, weights: Path | str = YOLO_WEIGHTS, conf: float = CONF,
                iou: float = IOU, imgsz: int = IMGSZ, device=None,
                class_agnostic: bool = False) -> list:
    res = load_yolo(weights).predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device,
                                     verbose=False)[0]
    h, w = img.shape[:2]
    out = []
    if res.boxes is not None:
        for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                           res.boxes.conf.cpu().numpy()):
            x0, y0, x1, y1 = (int(round(float(v))) for v in b)
            x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
            y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
            out.append(TextBox(box=[x0, y0, max(x0, x1), max(y0, y1)],
                               label=None if class_agnostic else BOX_LABELS[c], score=float(s)))
    out.sort(key=lambda t: (t.box[1], t.box[0]))
    return out
