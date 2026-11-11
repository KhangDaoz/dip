"""E-A2 — chỉnh nghiêng (A2) ước lượng góc đúng không? (plan §5.1)

Xoay ảnh test một góc biết trước, ước lượng lại bằng lược đồ chiếu ngang. Góc thật = (góc danh nghĩa a + u) với
u ~ U[0; 0,5), dấu ngẫu nhiên — góc lẻ, không trùng lưới tìm 0,25° (nếu lấy đúng số nguyên thì sai số luôn 0).
Độ đo: sai số góc tuyệt đối — trung vị, P90, max, % ảnh sai ≤ 0,5°. Không cần trọng số.

    python experiments/e_a2_deskew.py               # 176 ảnh test × 7 góc
    python experiments/e_a2_deskew.py --limit 30    # thử nhanh
Kết quả: results/e_a2/{raw.csv, table.csv, error_by_angle.png, config.json}
"""
import argparse
import time

import numpy as np
import pandas as pd

from common.experiment import results_dir, save_config, save_figure, save_table
from common.io import read_image
from enhance.deskew import estimate_skew
from vaipe.dataset import list_images
from vaipe.degrade import rotate

EXP_ID = "e_a2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--angles", nargs="+", type=float, default=[0, 1, 2, 3, 5, 7, 10])
    ap.add_argument("--max-angle", type=float, default=12.0, help="miền tìm góc ±")
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    results_dir(EXP_ID)
    save_config(vars(args), EXP_ID)

    images = list_images(args.split)[:args.limit]
    rows, t0 = [], time.perf_counter()
    for k, img in enumerate(images, 1):
        im = read_image(img.image_path)
        for a in args.angles:
            true = (float(a) + rng.uniform(0, 0.5)) * rng.choice([-1, 1])
            tilted, _ = rotate(im, true)
            t = time.perf_counter()
            est = estimate_skew(tilted, max_angle=args.max_angle, step=args.step)
            rows.append({"image_id": img.image_id, "width": im.shape[1], "nominal": float(a), "angle": true, "est": est,
                         "abs_err": abs(est - true), "sec": time.perf_counter() - t})
        if k % 25 == 0:
            print(f"  {k}/{len(images)} ảnh ({time.perf_counter() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID, "raw")

    df["abs_angle"] = df["nominal"]
    summ = df.groupby("abs_angle").agg(
        n=("abs_err", "size"), median=("abs_err", "median"), p90=("abs_err", lambda s: s.quantile(0.9)),
        max=("abs_err", "max"), pct_le_0_5=("abs_err", lambda s: 100 * (s <= 0.5).mean()),
        sec_per_img=("sec", "mean")).reset_index()
    total = {"abs_angle": "all", "n": len(df), "median": df["abs_err"].median(),
             "p90": df["abs_err"].quantile(0.9), "max": df["abs_err"].max(),
             "pct_le_0_5": 100 * (df["abs_err"] <= 0.5).mean(), "sec_per_img": df["sec"].mean()}
    summ = pd.concat([summ, pd.DataFrame([total])], ignore_index=True)
    save_table(summ, EXP_ID, "table")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    groups = [df.loc[df["abs_angle"] == a, "abs_err"].values for a in sorted(df["abs_angle"].unique())]
    ax.boxplot(groups, tick_labels=[f"{a:g}" for a in sorted(df["abs_angle"].unique())], showfliers=True)
    ax.axhline(args.step / 2, ls="--", lw=0.8, color="gray", label=f"nửa bước tìm ({args.step / 2:g}°)")
    ax.set_xlabel("Góc nghiêng danh nghĩa a (thật: a + U[0; 0,5)°), độ")
    ax.set_ylabel("Sai số |θ̂ − θ| (độ)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, EXP_ID, "error_by_angle")


if __name__ == "__main__":
    main()
