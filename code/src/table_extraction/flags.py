"""K7 vai trò cột + cờ bất thường — hậu xử lý văn bản, KHÔNG tính là đóng góp DIP (KeHoach.md mục 1.3, 7.1)."""


def parse_number(text: str):
    """"14,9" / "14.9" → 14.9; không đọc được → None."""
    raise NotImplementedError("parse_number")


def parse_range(text: str):
    """"13.5 - 17.5" → [13.5, 17.5]; không đọc được → None."""
    raise NotImplementedError("parse_range")


def flag(result, value_range) -> str:
    """So kết quả với khoảng tham chiếu → "LOW" / "NORMAL" / "HIGH" / "UNKNOWN"."""
    raise NotImplementedError("flag")


def column_roles(cells: list, texts: dict) -> dict:
    """Luật thứ tự cột + kiểu chữ (số, đơn vị) → {col: "parameter" | "result" | "unit" | "range"}."""
    raise NotImplementedError("column_roles")
