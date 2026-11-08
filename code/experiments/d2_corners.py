"""D2 — Tìm 4 góc: contour vs Hough vs minAreaRect vs kết hợp có dự phòng
Người làm: Hiếu. Kế hoạch: team/KeHoach_Hieu.md (mục 3.2).

Dữ liệu: bộ 6 tập test, bỏ mẫu có góc giấy nằm ngoài khung (mẫu chỉ xoay — dùng cho D4).
Mẫu không biến dạng hình học (chỉ mờ / bóng) có 4 góc = khung ảnh: đáp án đúng là "full_image".
Độ đo: sai số góc lớn nhất (% đường chéo) so với ground truth; thành công khi < 2% đường chéo;
       tỉ lệ không tìm được (phương án đơn lẻ trả None); thời gian ms/ảnh.
Đầu ra: results/d2/table.csv (từng mẫu × phương án), results/d2/summary.csv, results/d2/figure.png

Chạy: python experiments/d2_corners.py
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bo6
from common.experiment import save_config, save_figure, save_table
from common.seed import set_seed
from preprocessing import corners as C
from preprocessing.quality import resize_long_side

EXP_ID = "d2"
CONFIG = {"seed": 42, "split": "test", "success_pct_diag": 2.0, "long_side": 1000}
METHODS = ["contour", "hough", "minrect", "combined"]
LABELS = {"contour": "Contour", "hough": "Hough", "minrect": "minAreaRect", "combined": "Kết hợp có dự phòng"}


def load_data(config: dict):
    m = bo6.manifest((config["split"],))
    rows = []
    for _, r in m.iterrows():
        gt = bo6.label(r)
        if bo6.corners_inside(gt):
            rows.append((r, gt))
    return rows


def _detect(img, method: str, long_side: int):
    """Trả về 4 góc theo tọa độ ảnh gốc, hoặc None nếu phương án không tìm được."""
    if method == "combined":
        return C.find_corners(img, long_side)[0]
    small, scale = resize_long_side(img, long_side)
    quad = dict(C.FINDERS)[method](small)
    return None if quad is None else quad / scale


def run(data, config: dict) -> pd.DataFrame:
    out = []
    for r, gt in data:
        img = bo6.degraded(r)
        diag = float(np.hypot(*img.shape[:2]))
        truth = C.order_corners(np.asarray(gt["corners"], np.float32))
        for method in METHODS:
            t0 = time.perf_counter()
            quad = _detect(img, method, config["long_side"])
            ms = (time.perf_counter() - t0) * 1000
            err = np.nan if quad is None else float(np.linalg.norm(C.order_corners(quad) - truth, axis=1).max())
            out.append({"sample_id": r.sample_id, "recipe": r.recipe_name, "perspective": r.perspective_strength,
                        "angle": abs(r.angle_deg), "blur": r.blur_level, "shadow": r.shadow_level,
                        "method": method, "found": quad is not None, "err_px": err, "err_pct_diag": 100 * err / diag,
                        "ms": ms})
    df = pd.DataFrame(out)
    df["success"] = df.err_pct_diag < config["success_pct_diag"]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("method", sort=False)
    s = pd.DataFrame({
        "n": g.size(),
        "found_pct": 100 * g.found.mean(),
        "success_pct": 100 * g.success.mean(),
        "err_pct_diag_median": g.err_pct_diag.median(),
        "err_px_mean_when_found": g.err_px.mean(),
        "ms_mean": g.ms.mean(),
    }).reindex(METHODS).round(2)
    return s.reset_index()


def plot(df: pd.DataFrame):
    geo = df[df.perspective > 0]
    rate = 100 * geo.groupby(["perspective", "method"]).success.mean().unstack()[METHODS]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(rate))
    w = 0.2
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - 1.5) * w, rate[m], w, label=LABELS[m])
    ax.set_xticks(x, [f"{100 * p:.0f}%" for p in rate.index])
    ax.set_xlabel("Mức phối cảnh")
    ax.set_ylabel("Tìm góc thành công (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    set_seed(CONFIG["seed"])
    save_config(CONFIG, EXP_ID)
    df = run(load_data(CONFIG), CONFIG)
    print("Bảng:", save_table(df, EXP_ID))
    summary = summarize(df)
    print("Tóm tắt:", save_table(summary, EXP_ID, "summary"))
    print(summary.to_string(index=False))
    print("Hình:", save_figure(plot(df), EXP_ID))


if __name__ == "__main__":
    main()
