"""② Ghép R1–R9 theo hợp đồng CleaningResult (common/contracts.py)."""
from pathlib import Path

import numpy as np

from common.contracts import CleaningResult


def clean(img_bgr: np.ndarray, doc_type: str, image_id: str, out_dir: Path) -> CleaningResult:
    """Ảnh đã nắn → ảnh xám sạch, ảnh nhị phân, mask (stamp, handwriting, red_text), components có nhãn.

    Đơn thuốc: chạy xóa dấu / chữ ký (R6). Phiếu xét nghiệm: KHÔNG xóa, giữ chữ đỏ cảnh báo.
    Giữ nguyên kích thước và hệ tọa độ ảnh đầu vào.
    """
    raise NotImplementedError("clean")
