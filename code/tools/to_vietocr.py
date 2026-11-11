"""Cắt crops cho VietOCR từ nhãn VAIPE (plan §5.3).

Mỗi vùng chữ → 1 PNG trong <out>/<split>/, đệm 2px, phóng ×2 (nội suy bicubic).
Kèm file nhãn `<split>_label.txt` theo định dạng VietOCR: `<path>\t<text>`.

Chạy:
    python tools/to_vietocr.py                  # cả train+valid+test
    python tools/to_vietocr.py --split train     # một split
    python tools/to_vietocr.py --split train --pad 3 --scale 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common.paths import VAIPE_CROPS
from vaipe.dataset import list_images


def crop_one(img_bgr: np.ndarray, box: list, pad: int = 2, scale: float = 2.0) -> np.ndarray:
    """Cắt vùng box xyxy (pixel) trên ảnh BGR, đệm `pad` pixel, phóng `scale` lần.

    Nội suy bicubic (ch. 2 · S23-24). Nếu box sát mép, đệm bằng nền (giá trị biên).
    """
    h, w = img_bgr.shape[:2]
    x0 = max(0, int(box[0]) - pad)
    y0 = max(0, int(box[1]) - pad)
    x1 = min(w, int(box[2]) + pad)
    y1 = min(h, int(box[3]) + pad)
    crop = img_bgr[y0:y1, x0:x1]
    if scale != 1.0:
        new_w = max(1, int(crop.shape[1] * scale))
        new_h = max(1, int(crop.shape[0] * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return crop


def convert_split(split: str, out_root: Path, pad: int = 2, scale: float = 2.0) -> int:
    """Sinh crops PNG + file nhãn VietOCR cho split. Trả về số crops."""
    img_out = out_root / split
    label_file = out_root / f"{split}_label.txt"
    img_out.mkdir(parents=True, exist_ok=True)
    images = list_images(split)
    rows = []
    for img in images:
        bgr = cv2.imread(str(img.image_path))
        if bgr is None:
            continue
        for a in img.annotations:
            crop = crop_one(bgr, a.box, pad=pad, scale=scale)
            if crop.size == 0:
                continue
            name = f"{img.image_id}_{a.id}.png"
            cv2.imwrite(str(img_out / name), crop)
            rows.append(f"{(img_out / name).as_posix()}\t{a.text.strip()}")
    label_file.write_text("\n".join(rows), encoding="utf-8")
    print(f"[to_vietocr] {split}: {len(rows)} crops → {label_file}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Cắt crops cho VietOCR từ nhãn VAIPE")
    ap.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    ap.add_argument("--pad", type=int, default=2, help="đệm pixel quanh box")
    ap.add_argument("--scale", type=float, default=2.0, help="phóng to crop")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_root = args.out or VAIPE_CROPS
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    for s in splits:
        convert_split(s, out_root, pad=args.pad, scale=args.scale)


if __name__ == "__main__":
    main()
