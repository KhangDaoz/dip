"""Đọc bộ 5 (benchmark tổng hợp của Thắng) cho các script thí nghiệm.

Cấu trúc (từ 02/11): data/5_synthetic/<split>/{images,labels,clean,clean_binary,shadow_mask,page_mask}/<sample_id>
Mọi đường dẫn lấy từ manifest_v1.csv (cột rel_*_path), nên không phụ thuộc cách đặt thư mục.
"""
import json

import numpy as np
import pandas as pd

from common.io import read_image
from common.paths import DATASETS

ROOT = DATASETS[5]


def manifest(splits=("test",)) -> pd.DataFrame:
    m = pd.read_csv(ROOT / "manifest_v1.csv")
    return m[m.split.isin(splits)].reset_index(drop=True)


def label(row) -> dict:
    with open(ROOT / row.rel_gt_path, encoding="utf-8") as f:
        return json.load(f)


def degraded(row) -> np.ndarray:
    return read_image(ROOT / row.rel_degraded_path)


def clean(row) -> np.ndarray:
    return read_image(ROOT / row.rel_clean_path)


def corners_inside(gt: dict, tol: float = 1.0) -> bool:
    """4 góc giấy đều nằm trong khung ảnh hỏng (mẫu chỉ xoay thường bị mất góc)."""
    w, h = gt["image_size"]
    c = np.asarray(gt["corners"])
    return bool(((c[:, 0] >= -tol) & (c[:, 0] <= w - 1 + tol) & (c[:, 1] >= -tol) & (c[:, 1] <= h - 1 + tol)).all())
