"""Đọc bộ 5 đã sinh (data/5_synthetic): manifest, nhãn, ảnh — dùng chung cho thí nghiệm D*, T*, S*."""
import numpy as np
import pandas as pd

from common.io import load_json, read_image
from common.paths import DATASETS

ROOT = DATASETS[5]


def manifest(split: str = None) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "manifest_v2.csv")
    df["degradations"] = df["applied_degradations"].fillna("").astype(str)
    df["geometric"] = (df["perspective_strength"].fillna(0) > 0) | (df["angle_deg"].fillna(0) != 0)
    return df if split is None else df[df["split"] == split].reset_index(drop=True)


def label(row) -> dict:
    return load_json(ROOT / row["rel_gt_path"])


def image(row, kind: str = "degraded", gray: bool = False) -> np.ndarray:
    rel = {"degraded": row["rel_degraded_path"], "clean": row["rel_clean_path"],
           "clean_binary": row["rel_clean_binary_path"]}.get(kind)
    if rel is None:
        rel = f"{row['split']}/{kind}/{row['sample_id']}.png"
    return read_image(ROOT / rel, gray=gray)


def mask(row, kind: str) -> np.ndarray:
    """kind ∈ glare_mask, stamp_mask, pen_mask, shadow_mask, page_mask, ignore_band."""
    p = ROOT / row["split"] / kind / f"{row['sample_id']}.png"
    return read_image(p, gray=True) if p.exists() else None


def items_band(gt: dict, space: str = "clean") -> list:
    """GT khoản thuốc theo định nghĩa DẢI NGANG (như S0 cho VAIPE): x trải hết bề ngang vùng thuốc, y theo khoản."""
    reg = gt.get(f"medicine_region_{space}")
    reg = reg["bbox"] if isinstance(reg, dict) else reg
    out = []
    for it in gt.get(f"items_{space}", []):
        if space == "degraded" and not it.get("valid", True):
            continue
        x, y, w, h = it["bbox"]
        out.append([reg[0], y, reg[2], h] if reg else [x, y, w, h])
    return out
