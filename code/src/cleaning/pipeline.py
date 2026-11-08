"""Ghép các bước của ② thành một hàm, trả về CleaningResult (hợp đồng ② → ③④)."""
from pathlib import Path

import numpy as np

from common.contracts import CleaningResult
from common.paths import OUT_DIR


def clean(img: np.ndarray, doc_type: str, image_id: str, out_dir: Path = OUT_DIR) -> CleaningResult:
    """Nhận ảnh đã nắn của ①: khử bóng → tách mực → xóa dấu (giữ chữ đỏ) → nhị phân hóa → thành phần liên thông.

    Lưu ảnh xám sạch, ảnh nhị phân, các mask vào out_dir/<image_id>_*.png.
    """
    raise NotImplementedError("clean")
