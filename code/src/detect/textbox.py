"""Kiểu dữ liệu đầu ra của khối B.

Plan.md mục 4.2 đặt `TextBox` trong `common/contracts.py` (Hiếu viết lại). Khi file đó đã có `TextBox` thì dùng bản
chung; chưa có thì dùng bản tạm dưới đây, khai báo đúng các trường của plan để đổi sang không phải sửa code gọi.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# Thứ tự lớp cố định, trùng tools/to_yolo.py và model.names của yolo_vaipe.pt
CLASSES = ("drugname", "quantity", "usage", "diagnose", "date", "other")

try:  # bản chung khi Hiếu đã viết lại contracts
    from common.contracts import TextBox  # type: ignore[attr-defined]
except ImportError:
    @dataclass
    class TextBox:
        """Đầu ra B, đầu vào C."""
        box: list            # [x0, y0, x1, y1] pixel, cùng quy ước nhãn VAIPE
        label: str | None    # một trong CLASSES, hoặc None nếu bộ phát hiện không phân lớp
        score: float
        text: str = ""       # C điền
        text_conf: float = 0.0

        def to_dict(self) -> dict:
            return asdict(self)
