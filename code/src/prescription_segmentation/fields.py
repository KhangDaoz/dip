"""S-B5 tách trường bằng regex — hậu xử lý văn bản, KHÔNG tính là đóng góp DIP (KeHoach.md mục 1.3)."""


def parse_item_fields(text: str) -> dict:
    """Văn bản OCR một khoản → {"name", "strength", "dose": {"morning", "noon", "evening"}, "days", "note"}."""
    raise NotImplementedError("parse_item_fields")
