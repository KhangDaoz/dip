"""E-A2 — chỉnh nghiêng đúng không? Xoay ảnh test góc biết trước (0–10°), ước lượng lại (plan §5.1). Người làm: Hiếu.

    python experiments/e_a2_deskew.py --limit 50
Độ đo: sai số góc tuyệt đối (trung vị, P90). Không cần trọng số.
"""
import argparse

import _v2  # noqa: F401
import numpy as np
import pandas as pd

from common.io import read_image, save_json
from enhance.deskew import estimate_skew, rotate
from vaipe.dataset import list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--angles", nargs="+", type=float, default=[0, 1, 2, 3, 5, 7, 10])
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    out = _v2.out_dir("e_a2")
    rows = []
    for img in list_images("test")[:args.limit]:
        im = read_image(img.image_path)
        for a in args.angles:
            a = float(a) * rng.choice([-1, 1])
            est = estimate_skew(rotate(im, a), max_angle=12)
            rows.append({"image_id": img.image_id, "angle": a, "est": est, "abs_err": abs(est - a)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "raw.csv", index=False, encoding="utf-8-sig")
    df["abs_angle"] = df["angle"].abs()
    summ = df.groupby("abs_angle")["abs_err"].agg(median="median", p90=lambda s: s.quantile(0.9), mean="mean")
    summ.to_csv(out / "table.csv", encoding="utf-8-sig")
    save_json(out / "config.json", {**vars(args), "commit": _v2.git_hash()})
    print(summ.to_string(), f"\nTổng: trung vị {df['abs_err'].median():.3f}°")


if __name__ == "__main__":
    main()
