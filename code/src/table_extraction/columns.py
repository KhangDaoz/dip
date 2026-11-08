"""Tìm cột từ phân bố cụm từ. Thí nghiệm K2. Mọi hàm trả về list khoảng [x_trái, x_phải] của từng cột."""


def columns_projection(binary, words: list) -> list:
    """Projection dọc trên dải dòng của table body. `words` là list bounding box cụm từ."""
    raise NotImplementedError("columns_projection")


def columns_xycut(binary, words: list) -> list:
    """Phân tách cột bằng Recursive XY-Cut trên các cụm từ."""
    raise NotImplementedError("columns_xycut")


def columns_fixed(words: list, gap: float = 2.0) -> list:
    """Gom cột theo phân bố tọa độ x của các cụm từ."""
    raise NotImplementedError("columns_fixed")
