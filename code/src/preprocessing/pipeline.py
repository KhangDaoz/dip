"""① Ghép H1–H11 theo hợp đồng PreprocessResult (common/contracts.py)."""
from pathlib import Path

import numpy as np

from common.contracts import PreprocessResult


def resize_long_side(img: np.ndarray, long_side: int = 1000) -> tuple:
    """H1: đưa cạnh dài về long_side để dò; trả về (ảnh nhỏ, hệ số tỉ lệ). Nắn vẫn chạy trên ảnh gốc."""
    raise NotImplementedError("resize_long_side")


def preprocess(img_bgr: np.ndarray, image_id: str, out_dir: Path, doc_type: str = None) -> PreprocessResult:
    """Ảnh chụp → ảnh đã nắn + chỉnh nghiêng (chưa CLAHE), ma trận H tổng, 4 góc, loại tài liệu, điểm chất lượng.

    doc_type khác None thì bỏ qua bước phân loại (dùng khi classification chưa xong).
    Ghi ảnh đã nắn vào out_dir/<image_id>_rect.png.
    """
    raise NotImplementedError("preprocess")
