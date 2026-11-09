"""S6 — Kiểm chứng bộ 5 + dải viền bỏ qua khi chấm nhị phân. Người làm: Thắng. Kế hoạch: KeHoach.md mục 6.4, 8.2, 8.3.

1. Dải viền bỏ qua: với mỗi mẫu, ignore_band = (A ⊕ B) − (A ⊖ B), A = ảnh nhị phân chuẩn (Otsu trên ảnh sạch), B = 3×3
   (mở rộng tách biên C7 · S40). Ghi data/5_synthetic/<split>/ignore_band/<sample_id>.png cho T2, T3 của Trường.
2. Đáp án Otsu có thiên vị không: trên ẢNH SẠCH, đếm tỉ lệ pixel Otsu và Sauvola (tham số pipeline ②) cho kết quả khác
   nhau — trong và ngoài dải viền. Ngoài dải viền gần 0 % ⇒ chấm bỏ dải viền là khách quan.
3. Bộ 5 có "giống thật": so phân bố độ nét Var(∇²f) (C5 · S28–29), độ sáng / tương phản (lược đồ xám C3 · S23) của
   ảnh hỏng bộ 5 (test) với ảnh chụp thật bộ 4; PSNR ảnh hỏng so với ảnh sạch theo loại hỏng (C6 · S39).
Kết quả: results/s6/

Chạy (từ code/): python experiments/s6_bo5_check.py
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cleaning import binarize, metrics
from cleaning.pipeline import PARAMS as CLEAN_PARAMS, estimate_char_height
from common.experiment import results_dir, save_config, save_figure, save_table
from common.io import read_image, write_image
from common.paths import DATASETS
from common.seed import set_seed
from preprocessing import quality
from preprocessing.pipeline import resize_long_side
from synthetic import dataset as D, degrade

EXP_ID = "s6"
CONFIG = {"seed": 42, "band_ksize": 3, "sauvola_window_factor": CLEAN_PARAMS["sauvola_window_factor"],
          "sauvola_k": CLEAN_PARAMS["sauvola_k"], "stats_long_side": 1000}


def sauvola_pipeline(gray):
    h = estimate_char_height(gray)
    return binarize.sauvola(gray, int(max(15, round(CONFIG["sauvola_window_factor"] * h))) | 1, CONFIG["sauvola_k"])


def stats(img):
    small, _ = resize_long_side(img, CONFIG["stats_long_side"])
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return {"laplacian_var": quality.blur_laplacian(g), "mean": float(g.mean()), "std": float(g.std())}


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    man = D.manifest()

    # 1 + 2: dải viền và bất đồng Otsu / Sauvola trên ảnh sạch
    dis, seen = [], set()
    for _, row in man.iterrows():
        gt_bin = D.image(row, "clean_binary", gray=True)
        band = degrade.ignore_band(gt_bin, CONFIG["band_ksize"])
        write_image(D.ROOT / row["split"] / "ignore_band" / f"{row['sample_id']}.png", band)
        src = row["source_id"]
        if src in seen:
            continue
        seen.add(src)
        clean = D.image(row, "clean", gray=True)
        o, s = binarize.otsu(clean) > 0, sauvola_pipeline(clean) > 0
        b = band > 0
        diff = o != s
        dis.append({"source_id": src, "split": row["split"], "band_ratio": float(b.mean()),
                    "disagree_all": float(diff.mean()), "disagree_in_band": float(diff[b].mean()) if b.any() else 0,
                    "disagree_outside_band": float(diff[~b].mean()),
                    "share_of_disagreement_in_band": float(diff[b].sum() / max(diff.sum(), 1)),
                    "f_otsu_vs_gt_ignore": metrics.f_measure(o.astype(np.uint8) * 255, gt_bin, band),
                    "f_sauvola_vs_gt_ignore": metrics.f_measure(s.astype(np.uint8) * 255, gt_bin, band),
                    "f_sauvola_vs_gt_full": metrics.f_measure(s.astype(np.uint8) * 255, gt_bin)})
    dis = pd.DataFrame(dis)
    save_table(dis, EXP_ID, "disagreement")
    print("Bất đồng Otsu / Sauvola trên ảnh sạch (trung bình", len(dis), "ảnh gốc):")
    print(dis.drop(columns=["source_id", "split"]).mean().round(4).to_string())

    # 3: thống kê "giống thật"
    rows = []
    for _, row in man[man["split"] == "test"].iterrows():
        img = D.image(row)
        clean = D.image(row, "clean")
        rows.append({"source": "bộ 5 (test, ảnh hỏng)", "recipe": row["recipe_name"],
                     "psnr_vs_clean": metrics.psnr(img, clean) if img.shape == clean.shape else np.nan, **stats(img)})
    for p in sorted(DATASETS[4].rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            rows.append({"source": "bộ 4 (ảnh chụp thật)", "recipe": p.parent.name, "psnr_vs_clean": np.nan,
                         **stats(read_image(p))})
    st = pd.DataFrame(rows)
    save_table(st, EXP_ID, "realism")
    summ = st.groupby("source")[["laplacian_var", "mean", "std"]].describe(percentiles=[0.25, 0.5, 0.75]).T
    save_table(summ.reset_index(), EXP_ID, "realism_summary")
    print(st.groupby("source")[["laplacian_var", "mean", "std"]].median().round(1).to_string())
    print(st.dropna(subset=["psnr_vs_clean"]).groupby(st["recipe"].str.replace(r"_[^_]*$", "", regex=True))
          ["psnr_vs_clean"].mean().round(2).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col, title in zip(axes, ["laplacian_var", "mean", "std"],
                              ["Độ nét Var(∇²f) (log)", "Độ sáng trung bình", "Độ tương phản (độ lệch chuẩn)"]):
        for src, g in st.groupby("source"):
            v = np.log10(g[col] + 1) if col == "laplacian_var" else g[col]
            ax.hist(v, bins=30, alpha=0.55, density=True, label=src)
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("S6 — Bộ 5 so với ảnh chụp thật bộ 4")
    fig.tight_layout()
    save_figure(fig, EXP_ID)


if __name__ == "__main__":
    main()
