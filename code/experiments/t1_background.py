"""T1 — Ước lượng nền nào khử bóng tốt nhất? Người làm: Trường. Kế hoạch: KeHoach.md mục 6.1 (R1–R2), 6.2, 6.3.

Dữ liệu: bộ 5, mẫu có bóng (shadow_level > 0) và không biến dạng hình học; val chọn kích thước SE, test báo kết quả.
Phương án: không khử · trung vị cỡ lớn (C3 · S73) · Closing ảnh xám = Max rồi Min (C3 · S74, C7 · S35) ·
Closing + trung vị. Khử bóng = toán tử điểm I / B̂ · 255 (C3 · S10–11).
Độ đo: PSNR (C6 · S39), SSIM so với ảnh sạch (xám).
Kết quả: results/t1/

Chạy (từ code/): python experiments/t1_background.py
"""
import cv2
import pandas as pd

from cleaning import background, metrics
from common.experiment import results_dir, save_config, save_table
from common.seed import set_seed
from synthetic import dataset as D

EXP_ID = "t1"
CONFIG = {"seed": 42, "kernels": [31, 51, 81, 101], "median_after": 21,
          "methods": ["none", "median", "closing", "closing_median"]}


def run(row, method, k):
    g = D.image(row, gray=True)
    ref = D.image(row, "clean", gray=True)
    if method == "none":
        out = g
    elif method == "median":
        out = background.remove_shadow(g, background.background_median(g, k))
    elif method == "closing":
        out = background.remove_shadow(g, background.background_closing(g, k))
    else:
        out = background.remove_shadow(g, background.background_closing(g, k, CONFIG["median_after"]))
    return metrics.psnr(out, ref), metrics.ssim(out, ref)


def subset(split):
    m = D.manifest(split)
    return m[(m["shadow_level"] > 0) & (~m["geometric"])]


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    val, test = subset("val"), subset("test")
    if len(val) < 10:   # val có ít mẫu bóng thuần → chọn trên train (bộ 5 train không dùng để học gì)
        val = subset("train")
    sweep = []
    for method in CONFIG["methods"]:
        for k in ([0] if method == "none" else CONFIG["kernels"]):
            vals = [run(r, method, k) for _, r in val.iterrows()]
            sweep.append({"method": method, "kernel": k, "n": len(vals),
                          "psnr": sum(v[0] for v in vals) / len(vals), "ssim": sum(v[1] for v in vals) / len(vals)})
    sw = pd.DataFrame(sweep)
    save_table(sw, EXP_ID, "sweep")
    rows = []
    for method, g in sw.groupby("method"):
        best = g.sort_values("psnr", ascending=False).iloc[0]
        vals = [run(r, method, int(best["kernel"])) for _, r in test.iterrows()]
        rows.append({"method": method, "kernel": int(best["kernel"]), "n_test": len(vals),
                     "psnr": sum(v[0] for v in vals) / len(vals), "ssim": sum(v[1] for v in vals) / len(vals),
                     "val_psnr": best["psnr"]})
    df = pd.DataFrame(rows).sort_values("psnr", ascending=False)
    save_table(df, EXP_ID, "summary")
    print(sw.to_string())
    print(df.to_string())


if __name__ == "__main__":
    main()
