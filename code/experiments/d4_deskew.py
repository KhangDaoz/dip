"""D4 — Chỉnh nghiêng: Projection profile vs Hough vs Radon
Người làm: Hiếu. Kế hoạch: team/KeHoach_Hieu.md (mục 3.2).

Dữ liệu: bộ 5 tập test, tập chỉ xoay (±2, ±5, ±10, ±15°, mỗi góc 10 ảnh) — chạy chỉnh nghiêng trực tiếp,
không nắn phối cảnh trước. Mức 0°: bộ 5 không có mẫu chỉ xoay 0° → đo trên ảnh sạch (clean/) của
các ảnh gốc trong tập test.
Độ đo: sai số góc tuyệt đối (độ); tỉ lệ sai số ≤ 0,5°; thời gian ms/ảnh.
Quy ước dấu: bộ 5 angle_deg dương = ngược chiều kim đồng hồ, trùng quy ước của deskew_* → đáp án = angle_deg.
Đầu ra: results/d4/table.csv, results/d4/summary.csv, results/d4/by_angle.csv, results/d4/figure.png

Chạy: python experiments/d4_deskew.py
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bo5
from common.experiment import save_config, save_figure, save_table
from common.seed import set_seed
from preprocessing import deskew

EXP_ID = "d4"
CONFIG = {"seed": 42, "split": "test", "angle_range": 20.0, "step": 0.25, "ok_deg": 0.5}
METHODS = {"projection": deskew.deskew_projection, "hough": deskew.deskew_hough, "radon": deskew.deskew_radon}
LABELS = {"projection": "Projection profile", "hough": "Hough", "radon": "Radon"}


def load_data(config: dict):
    m = bo5.manifest((config["split"],))
    rot = m[(m.applied_degradations == "rotate") & (m.perspective_strength == 0)]
    data = [(r.sample_id, float(r.angle_deg), lambda r=r: bo5.degraded(r)) for _, r in rot.iterrows()]
    for _, r in m.drop_duplicates("source_id").iterrows():          # 0°: ảnh sạch của mỗi ảnh gốc
        data.append((f"{r.source_id.split(':')[-1]}_clean", 0.0, lambda r=r: bo5.clean(r)))
    return data


def run(data, config: dict) -> pd.DataFrame:
    out = []
    for sample_id, truth, load in data:
        img = load()
        for name, fn in METHODS.items():
            kwargs = {"angle_range": config["angle_range"]}
            if name != "hough":
                kwargs["step"] = config["step"]
            t0 = time.perf_counter()
            _, est = fn(img, **kwargs)
            ms = (time.perf_counter() - t0) * 1000
            out.append({"sample_id": sample_id, "true_deg": truth, "method": name, "est_deg": est,
                        "abs_err": abs(est - truth), "ms": ms})
    df = pd.DataFrame(out)
    df["ok"] = df.abs_err <= config["ok_deg"]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("method", sort=False)
    return pd.DataFrame({
        "n": g.size(), "mae_deg": g.abs_err.mean(), "median_deg": g.abs_err.median(),
        "max_deg": g.abs_err.max(), "ok_pct": 100 * g.ok.mean(), "ms_mean": g.ms.mean(),
    }).round(3).reset_index()


def plot(df: pd.DataFrame):
    by = df.assign(abs_true=df.true_deg.abs()).groupby(["abs_true", "method"]).abs_err.mean().unstack()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in METHODS:
        ax.plot(by.index, by[m], marker="o", label=LABELS[m])
    ax.set_xlabel("Góc nghiêng thật |θ| (độ)")
    ax.set_ylabel("Sai số trung bình (độ)")
    ax.set_xticks(by.index)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    set_seed(CONFIG["seed"])
    save_config(CONFIG, EXP_ID)
    df = run(load_data(CONFIG), CONFIG)
    print("Bảng:", save_table(df, EXP_ID))
    summary = summarize(df)
    save_table(summary, EXP_ID, "summary")
    by_angle = df.assign(abs_true=df.true_deg.abs()).pivot_table(index="abs_true", columns="method",
                                                                 values="abs_err", aggfunc="mean").round(3)
    save_table(by_angle.reset_index(), EXP_ID, "by_angle")
    print(summary.to_string(index=False))
    print(by_angle.to_string())
    print("Hình:", save_figure(plot(df), EXP_ID))


if __name__ == "__main__":
    main()
