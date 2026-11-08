"""Tách trường trong một khoản thuốc từ text OCR."""
from common.contracts import MedicineItem


def parse_item_fields(text: str, bbox: list) -> MedicineItem:
    """Tên thuốc, hàm lượng, liều sáng / trưa / tối, số ngày, ghi chú (song ngữ Việt – Anh)."""
    raise NotImplementedError("parse_item_fields")
