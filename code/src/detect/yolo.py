"""Khối B — YOLOv8s phát hiện + phân lớp vùng chữ, 6 lớp (plan §5.2).

Trọng số: code/weights/yolo_vaipe.pt (notebooks/train_yolo.ipynb; tải: tools/download_weights.py).

    from detect.yolo import detect_yolo
    boxes = detect_yolo(img_bgr)          # list[TextBox], xyxy pixel ảnh gốc, sắp trên→dưới, trái→phải
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.contracts import BOX_LABELS, TextBox
from common.paths import YOLO_WEIGHTS

IMGSZ = 1024          # = imgsz lúc train
CONF = 0.443          # ngưỡng tối đa F1 trung bình 6 lớp trên VALID (test_metrics.json: conf_max_f1_on_val)
IOU = 0.7             # NMS, mặc định ultralytics (= lúc chấm val/test)

_MODELS: dict = {}


class WeightsMissing(FileNotFoundError):
    """Chưa có trọng số — chạy `python tools/download_weights.py`."""


def load_yolo(weights: Path | str = YOLO_WEIGHTS):
    """Nạp (và cache) mô hình ultralytics YOLO."""
    weights = Path(weights)
    key = str(weights.resolve())
    if key not in _MODELS:
        if not weights.exists():
            raise WeightsMissing(f"không thấy {weights} — chạy: python tools/download_weights.py")
        from ultralytics import YOLO      # import muộn: module khác không cần ultralytics
        model = YOLO(str(weights))
        names = [model.names[i] for i in range(len(model.names))]
        if tuple(names) != BOX_LABELS:
            raise ValueError(f"lớp của {weights.name} là {names}, cần {BOX_LABELS}")
        _MODELS[key] = model
    return _MODELS[key]


def detect_yolo(img: np.ndarray, weights: Path | str = YOLO_WEIGHTS, conf: float = CONF,
                iou: float = IOU, imgsz: int = IMGSZ, device=None,
                class_agnostic: bool = False) -> list:
    """Ảnh BGR → list[TextBox] (box xyxy int, label 1 trong 6 lớp, score = độ tin cậy).

    class_agnostic=True bỏ nhãn lớp (label=None) — để so công bằng với detector cổ điển (E-B1).
    """
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
