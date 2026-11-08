"""D1 (phần lóa) — Phát hiện lóa: dò ngưỡng V, S của glare_mask; IoU với mask lóa và phát hiện ảnh có lóa
Người làm: Hiếu. Kế hoạch: team/KeHoach_Hieu.md (mục 3.2).

Dữ liệu: bộ 6 v2.1 (manifest_v2.csv) — dò tham số trên val, báo cáo trên test. Ảnh có lóa: v10, v14
(glare_level 1–3, đã giảm phơi sáng 0,80–0,92 nên giấy ≈ 175–235 như ảnh chụp); mẫu âm: còn lại (có cả mờ, bóng).
Vùng giấy lấy từ page_mask của bộ 6 (tách sai số tìm góc ra khỏi phép đo lóa). Ảnh resize cạnh dài 1000 px như assess_quality.
Độ đo: IoU mask lóa (ảnh dương, theo mức); phát hiện ảnh có lóa: AUC của glare_ratio, F1 tại ngưỡng diện tích chọn trên val.
Đầu ra: results/d1_glare/table.csv, summary.csv, by_level.csv, figure.png

Chạy: python experiments/d1_glare.py
"""
import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

import bo6
from common.experiment import save_config, save_figure, save_table
from common.io import read_image
from common.seed import set_seed
from preprocessing.quality import GLARE_S, GLARE_V, glare_mask, resize_long_side

EXP_ID = "d1_glare"
CONFIG = {"seed": 42, "glare_v": [220, 230, 240, 250], "glare_s": [20, 30, 40, 60], "long_side": 1000,
          "select_split": "val", "report_split": "test"}


def load_data(config: dict):
    m = pd.read_csv(bo6.ROOT / "manifest_v2.csv")
    m = m[m.split.isin([config["select_split"], config["report_split"]])].reset_index(drop=True)
    data = []
    for _, r in m.iterrows():
        img, scale = resize_long_side(read_image(bo6.ROOT / r.rel_degraded_path), config["long_side"])
        size = (img.shape[1], img.shape[0])
        page = cv2.resize(read_image(bo6.ROOT / r.rel_degraded_path.replace("/images/", "/page_mask/"), gray=True),
                          size, interpolation=cv2.INTER_NEAREST)
        gt = np.zeros(img.shape[:2], bool)
        if r.glare_level > 0:
            gt = cv2.resize(read_image(bo6.ROOT / r.rel_glare_mask_path, gray=True), size, interpolation=cv2.INTER_NEAREST) > 0
        data.append((r, img, page, gt))
    return data


def run(data, config: dict) -> pd.DataFrame:
    out = []
    for r, img, page, gt in data:
        for v, s in itertools.product(config["glare_v"], config["glare_s"]):
            pred = glare_mask(img, v_thresh=v, s_thresh=s, region=page) > 0
            union = (pred | gt).sum()
            out.append({"sample_id": r.sample_id, "split": r.split, "glare_level": int(r.glare_level), "v": v, "s": s,
                        "glare_ratio": float(pred[page > 0].mean()) if (page > 0).any() else float(pred.mean()),
                        "iou": float((pred & gt).sum() / union) if union else np.nan})
    return pd.DataFrame(out)


def _best_ratio_threshold(score, y):
    cands = np.unique(np.concatenate([[0.0], score]))
    f1 = [f1_score(y, score > t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1))])


def summarize(df: pd.DataFrame, config: dict):
    sel, rep = df[df.split == config["select_split"]], df[df.split == config["report_split"]]
    grid = sel[sel.glare_level > 0].groupby(["v", "s"]).iou.mean().rename("val_iou").reset_index()
    best = grid.sort_values("val_iou", ascending=False).iloc[0]
    rows = []
    for label, (v, s) in (("chọn trên val", (best.v, best.s)), ("mặc định hiện tại", (GLARE_V, GLARE_S))):
        a = sel[(sel.v == v) & (sel.s == s)]
        b = rep[(rep.v == v) & (rep.s == s)]
        t = _best_ratio_threshold(a.glare_ratio.values, (a.glare_level > 0).values)
        y = (b.glare_level > 0).values
        rows.append({"config": label, "v": int(v), "s": int(s), "ratio_threshold_val": round(t, 5),
                     "iou_test": b[b.glare_level > 0].iou.mean(), "auc_test": roc_auc_score(y, b.glare_ratio),
                     "f1_test": f1_score(y, b.glare_ratio > t),
                     "f1_test_level2plus": f1_score(b.glare_level >= 2, b.glare_ratio > t)})
    by_level = rep[(rep.v == best.v) & (rep.s == best.s) & (rep.glare_level > 0)].groupby("glare_level").iou.agg(["mean", "median", "count"])
    return pd.DataFrame(rows).round(4), grid.round(4), by_level.round(3).reset_index(), best


def plot(df: pd.DataFrame, best, config: dict):
    rep = df[(df.split == config["report_split"]) & (df.v == best.v) & (df.s == best.s)]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    fpr, tpr, _ = roc_curve(rep.glare_level > 0, rep.glare_ratio)
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(rep.glare_level > 0, rep.glare_ratio):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("Tỉ lệ dương tính giả")
    axes[0].set_ylabel("Tỉ lệ phát hiện đúng ảnh có lóa")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.3)
    pos = rep[rep.glare_level > 0]
    axes[1].boxplot([pos[pos.glare_level == k].iou for k in (1, 2, 3)], tick_labels=["1", "2", "3"])
    axes[1].set_xlabel("Mức lóa")
    axes[1].set_ylabel("IoU mask lóa")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    set_seed(CONFIG["seed"])
    save_config(CONFIG, EXP_ID)
    df = run(load_data(CONFIG), CONFIG)
    print("Bảng:", save_table(df, EXP_ID))
    summary, grid, by_level, best = summarize(df, CONFIG)
    save_table(summary, EXP_ID, "summary")
    save_table(grid, EXP_ID, "grid_val")
    save_table(by_level, EXP_ID, "by_level")
    print(grid.pivot(index="v", columns="s", values="val_iou").to_string())
    print(summary.to_string(index=False))
    print(by_level.to_string(index=False))
    print("Hình:", save_figure(plot(df, best, CONFIG), EXP_ID))


if __name__ == "__main__":
    main()
