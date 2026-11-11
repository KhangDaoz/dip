"""Khối B — phát hiện vùng chữ. `detect(img, method)` chọn biến thể."""
from __future__ import annotations


def detect(img, method: str = "yolo", **kw) -> list:
    if method == "yolo":
        from .yolo import detect_yolo
        return detect_yolo(img, **kw)
    if method == "classical":
        from .classical import detect_classical
        return detect_classical(img, **kw)
    raise ValueError(f"method phải là yolo/classical, nhận {method}")
