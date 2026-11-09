"""D1 — Đo mờ / đo lóa như một bộ phát hiện. Người làm: Hiếu. Kế hoạch: KeHoach.md mục 5.1 (H3–H4), 5.2, 5.3.

Mờ: bộ 5 không biến dạng hình học; nhãn "mờ" = blur_level ≥ 1. Điểm: Var(∇²f) (C5 · S28–29, đề xuất) ·
Tenengrad (Sobel, C5 · S15, S19) · năng lượng tần số cao (C3 · S8). Ngưỡng chọn F1 lớn nhất trên val, báo F1 trên test;
AUC của đường đánh đổi theo ngưỡng (C8 · S49: tăng ngưỡng thì P tăng, R giảm).
Lóa: mẫu glare_level > 0; mask {V > τ_V} ∩ {S < τ_S} (C4 · S39–41); chọn τ trên val theo IoU, báo IoU trên test.
Kết quả: results/d1/

Chạy (từ code/): python experiments/d1_iqa.py
"""
import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cleaning.metrics import mask_iou
from common.experiment import results_dir, save_config, save_figure, save_table
from common.seed import set_seed
from preprocessing import quality
from preprocessing.pipeline import resize_long_side
from synthetic import dataset as D

EXP_ID = "d1"
CONFIG = {"seed": 42, "long_side": 1000, "tau_v": [220, 230, 240, 250], "tau_s": [20, 30, 45, 60]}
SCORES = {"laplacian": quality.blur_laplacian, "tenengrad": quality.blur_tenengrad, "fft": quality.blur_fft}


def roc(scores, labels):
    """Điểm càng NHỎ càng mờ → dự đoán mờ khi điểm < t. Trả về (fpr, tpr, auc, các ngưỡng)."""
    s, y = np.asarray(scores), np.asarray(labels).astype(bool)
    ts = np.unique(s)
    tpr = [np.mean(s[y] < t) for t in ts] + [1.0]
    fpr = [np.mean(s[~y] < t) for t in ts] + [1.0]
    order = np.argsort(fpr)
    auc = float(np.trapezoid(np.array(tpr)[order], np.array(fpr)[order]))
    return np.array(fpr), np.array(tpr), auc, ts


def f1_at(s, y, t):
    pred = np.asarray(s) < t
    y = np.asarray(y).astype(bool)
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    man = D.manifest()
    man = man[~man["geometric"] & man["split"].isin(["val", "test"])]
    rows = []
    for _, r in man.iterrows():
        small, _ = resize_long_side(D.image(r), CONFIG["long_side"])
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        rows.append({"sample_id": r["sample_id"], "split": r["split"], "blurry": r["blur_level"] > 0,
                     "blur_level": r["blur_level"], **{k: fn(g) for k, fn in SCORES.items()}})
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID, "blur_scores")
    res = []
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for k in SCORES:
        v, t = df[df["split"] == "val"], df[df["split"] == "test"]
        _, _, _, ts = roc(v[k], v["blurry"])
        best_t = max(ts, key=lambda th: f1_at(v[k], v["blurry"], th))
        fpr, tpr, auc, _ = roc(t[k], t["blurry"])
        res.append({"score": k, "threshold": float(best_t), "val_f1": f1_at(v[k], v["blurry"], best_t),
                    "test_f1": f1_at(t[k], t["blurry"], best_t), "test_auc": auc,
                    "n_test": len(t), "n_blurry_test": int(t["blurry"].sum())})
        ax.plot(np.sort(fpr), np.sort(tpr), label=f"{k} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("Tỉ lệ báo nhầm (FPR)")
    ax.set_ylabel("Tỉ lệ phát hiện (TPR)")
    ax.set_title("D1 — Phát hiện ảnh mờ (test bộ 5)")
    ax.legend(fontsize=8)
    save_figure(fig, EXP_ID, "roc_blur")
    blur = pd.DataFrame(res)
    save_table(blur, EXP_ID, "blur_summary")
    print(blur.to_string())

    gl = D.manifest()
    gl = gl[(gl["glare_level"] > 0) & (~gl["geometric"]) & gl["split"].isin(["val", "test"])]
    cache = [(r["split"], r["glare_level"], D.image(r), D.mask(r, "glare_mask")) for _, r in gl.iterrows()]
    grid = []
    for tv, ts_ in itertools.product(CONFIG["tau_v"], CONFIG["tau_s"]):
        ious = [mask_iou(quality.glare_mask(img, tv, ts_), gt) for sp, _, img, gt in cache if sp == "val" and gt is not None]
        grid.append({"tau_v": tv, "tau_s": ts_, "val_iou": float(np.mean(ious)) if ious else 0})
    grid = pd.DataFrame(grid).sort_values("val_iou", ascending=False)
    save_table(grid, EXP_ID, "glare_sweep")
    tv, ts_ = int(grid.iloc[0]["tau_v"]), int(grid.iloc[0]["tau_s"])
    test = [(lv, mask_iou(quality.glare_mask(img, tv, ts_), gt)) for sp, lv, img, gt in cache
            if sp == "test" and gt is not None]
    g = pd.DataFrame(test, columns=["glare_level", "iou"]).groupby("glare_level")["iou"].agg(["mean", "count"])
    save_table(g.reset_index(), EXP_ID, "glare_summary")
    print("Lóa: τ_V =", tv, "τ_S =", ts_)
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
