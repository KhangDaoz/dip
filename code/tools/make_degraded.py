"""Sinh ảnh test suy biến 3 mức L0/L2/L4 (plan §2.4).

Dùng bọc `legacy/synthetic/degrade.py` — các phép: rotate, perspective, gaussian_blur,
motion_blur, shadow, glare, jpeg, gaussian_noise.

L0: ảnh gốc (không biến đổi).
L2: xoay ±2°, Gauss blur σ=1-1.5, bóng nhẹ, JPEG q=60-80.
L4: xoay ±5°, phối cảnh nhẹ, bóng đậm, motion blur, JPEG q=25-40, noise σ=5-10.

Kết quả: data/1_vaipe/test_degraded/<L0|L2|L4>/<image_id>.png + params.csv.
Seed cố định 42 để tái lập.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common.io import read_image, write_image
from common.paths import VAIPE_DEGRADED, vaipe_images
from legacy.synthetic.degrade import (
    rotate, perspective, gaussian_blur, motion_blur, shadow, jpeg,
)


def make_level(img: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Áp chuỗi biến đổi theo mức. L0 = ảnh gốc."""
    if level == 0:
        return img
    if level == 2:
        # L2: nhẹ
        ang = rng.uniform(-2, 2)
        img, _ = rotate(img, ang)
        sigma = rng.uniform(1.0, 1.5)
        img, _ = gaussian_blur(img, sigma)
        img, _ = shadow(img, 1, rng)
        q = int(rng.uniform(60, 80))
        img, _ = jpeg(img, q)
        return img
    if level == 4:
        # L4: nặng — L2 + phối cảnh + motion blur + bóng nặng + noise
        ang = rng.uniform(-5, 5)
        img, _ = rotate(img, ang)
        img, _ = perspective(img, 0.05, rng)
        img, _ = gaussian_blur(img, rng.uniform(1.0, 1.5))
        img, _ = motion_blur(img, 2, rng)
        img, _ = shadow(img, 3, rng)
        img, _ = jpeg(img, int(rng.uniform(25, 40)))
        return img
    raise ValueError(f"level phải 0/2/4, nhận {level}")


def main():
    ap = argparse.ArgumentParser(description="Sinh ảnh suy biến L0/L2/L4")
    ap.add_argument("--src", choices=["test", "valid", "train"], default="test",
                    help="split gốc để biến đổi (mặc định test)")
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 2, 4],
                    help="các mức cần sinh")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_root = args.out or VAIPE_DEGRADED
    out_root.mkdir(parents=True, exist_ok=True)
    src_dir = vaipe_images(args.src)
    paths = sorted(src_dir.glob("*.png"))
    rng = np.random.default_rng(args.seed)
    params_log = []
    n_done = 0
    for lv in args.levels:
        out_dir = out_root / f"L{lv}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            img = read_image(p)
            out_img = make_level(img, lv, rng)
            write_image(out_dir / p.name, out_img)
            params_log.append({"image_id": p.stem, "level": lv})
            n_done += 1
        print(f"[make_degraded] L{lv}: {len(paths)} ảnh → {out_dir}")
    # ghi params.csv
    import csv
    with open(out_root / "params.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "level"])
        w.writeheader()
        w.writerows(params_log)
    print(f"[make_degraded] xong {n_done} ảnh, params.csv tại {out_root}")


if __name__ == "__main__":
    main()
