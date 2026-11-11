"""Khối B — phát hiện + phân loại vùng chữ trên đơn thuốc VAIPE (plan.md mục 5.2, Trường).

* `yolo.YoloDetector`          phương pháp chính: YOLOv8s 6 lớp, imgsz 1024.
* `classical.ClassicalDetector` đối chứng cổ điển: Sauvola → giãn theo h → thành phần liên thông (không phân lớp).

Cả hai trả về `list[TextBox]` với box = [x0, y0, x1, y1] (cùng quy ước nhãn VAIPE).
"""
from .textbox import CLASSES, TextBox

__all__ = ["CLASSES", "TextBox"]
