"""Phương pháp chính của khối B: YOLOv8s phát hiện + phân 6 lớp vùng chữ (plan.md mục 5.2).

Trọng số: code/weights/yolo_vaipe.pt (train Kaggle T4×2, imgsz 1024, 50 epoch; tải từ Drive nhóm, không commit).
Ngưỡng `conf` chọn trên valid bằng experiments/e_b1.py --sweep-conf, ghi ở results/e_b1/conf_sweep.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .textbox import CLASSES, TextBox

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "yolo_vaipe.pt"
DEFAULT_CONF = 0.40   # chọn trên valid (max F1 không phân lớp 0,995), results/e_b1/conf_sweep.csv


class YoloDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS, imgsz: int = 1024, conf: float = DEFAULT_CONF,
                 iou: float = 0.6, device: str | None = None):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"Không thấy trọng số {weights} — tải yolo_vaipe.pt từ Drive nhóm về code/weights/")
        from ultralytics import YOLO   # import muộn: các phần khác của repo không cần cài ultralytics
        import torch

        self.model = YOLO(str(weights))
        self.names = {int(k): v for k, v in self.model.names.items()}
        if tuple(self.names[i] for i in range(len(self.names))) != CLASSES:
            raise ValueError(f"Thứ tự lớp của trọng số {self.names} khác CLASSES {CLASSES}")
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict_raw(self, img_bgr: np.ndarray, conf: float | None = None):
        """Trả về (boxes N×4 xyxy float, cls N int, score N float) — dùng cho quét ngưỡng."""
        r = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf if conf is None else conf,
                               iou=self.iou, device=self.device, verbose=False)[0]
        b = r.boxes
        return (b.xyxy.cpu().numpy().astype(float), b.cls.cpu().numpy().astype(int),
                b.conf.cpu().numpy().astype(float))

    def detect(self, img_bgr: np.ndarray) -> list[TextBox]:
        """Ảnh BGR uint8 → list[TextBox], sắp theo dòng rồi theo cột."""
        if img_bgr is None or getattr(img_bgr, "ndim", 0) not in (2, 3):
            raise ValueError("img_bgr phải là ảnh numpy (cv2.imread trả None khi sai đường dẫn)")
        boxes, cls, score = self.predict_raw(img_bgr)
        h, w = img_bgr.shape[:2]
        out = []
        for (x0, y0, x1, y1), c, s in zip(boxes, cls, score):
            box = [int(max(0, round(x0))), int(max(0, round(y0))), int(min(w, round(x1))), int(min(h, round(y1)))]
            out.append(TextBox(box=box, label=self.names[int(c)], score=float(s)))
        return sorted(out, key=lambda t: (t.box[1], t.box[0]))
