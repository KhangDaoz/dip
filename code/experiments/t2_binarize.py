"""T2 — Cách nhị phân hóa nào tốt nhất? (bảng chính của Trường). Kế hoạch: KeHoach.md mục 6.1 (R7, R8), 6.2, 6.3, 6.4.

Dữ liệu: bộ 5, mẫu KHÔNG biến dạng hình học (ảnh nhị phân chuẩn khớp từng pixel): val chọn tham số Sauvola (w, k) và
Opening; test báo kết quả. Đầu vào = ảnh hỏng → xám → (khử bóng Closing 81 hoặc không) → nhị phân.
Phương án: Sauvola (chính) · Otsu (đối chứng chính) · ISODATA · Tam giác · Đối xứng nền (C5 · S65–78) · Niblack · Wolf.
Độ đo: F-measure và pseudo-F CHẤM BỎ DẢI VIỀN 1 px (ignore_band từ S6), PSNR ảnh nhị phân; kèm F-measure không bỏ
dải viền để thấy thiên lệch của đáp án Otsu (KeHoach.md mục 6.4).
Kết quả: results/t2/{sweep.csv, table.csv (từng mẫu, T3 dùng lại), summary.csv}

Chạy (từ code/): python experiments/t2_binarize.py   (cần chạy s6_bo5_check.py trước để có ignore_band)
"""
import itertools

import numpy as np
import pandas as pd

from cleaning import background, binarize, metrics, postprocess
from cleaning.pipeline import estimate_char_height
from common.experiment import results_dir, save_config, save_table
from common.seed import set_seed
from synthetic import dataset as D

EXP_ID = "t2"
CONFIG = {"seed": 42, "background_kernel": 81, "min_area": 10,
          "sauvola_grid": {"window_factor": [1.5, 2.0, 2.5, 3.0, 4.0], "k": [0.1, 0.2, 0.3, 0.4, 0.5]},
          "opening_grid": [1, 2], "niblack_k": -0.2, "wolf_k": 0.5}
LEVELS = ["shadow_level", "blur_level", "stamp_level", "glare_level", "jpeg_quality", "noise_sigma", "pen_level"]


def prepare(row, shadow_removal):
    g = D.image(row, gray=True)
    if shadow_removal:
        g = background.remove_shadow(g, background.background_closing(g, CONFIG["background_kernel"]))
    return g


def run_method(g, method, p, h):
    if method in ("sauvola", "niblack", "wolf"):
        w = int(max(15, round(p.get("window_factor", 2.5) * h))) | 1
        k = {"sauvola": p.get("k", 0.2), "niblack": CONFIG["niblack_k"], "wolf": CONFIG["wolf_k"]}[method]
        b = binarize.binarize(g, method, window=w, k=k)
    else:
        b = binarize.binarize(g, method)
    return postprocess.remove_small(postprocess.opening(b, p.get("opening", 1)), CONFIG["min_area"])


def score(b, gt, band):
    return {"f": metrics.f_measure(b, gt, band), "pf": metrics.pseudo_f_measure(b, gt, band),
            "f_full": metrics.f_measure(b, gt), "psnr": metrics.psnr(b, gt)}


def subset(split):
    m = D.manifest(split)
    return m[~m["geometric"]].reset_index(drop=True)


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    val, test = subset("val"), subset("test")

    # chọn w, k, opening của Sauvola trên val (khử bóng bật, đúng cấu hình pipeline)
    cache = []
    for _, r in val.iterrows():
        g = prepare(r, True)
        cache.append((g, estimate_char_height(g), D.image(r, "clean_binary", gray=True), D.mask(r, "ignore_band")))
    sweep = []
    for wf, k, op in itertools.product(CONFIG["sauvola_grid"]["window_factor"], CONFIG["sauvola_grid"]["k"],
                                       CONFIG["opening_grid"]):
        p = {"window_factor": wf, "k": k, "opening": op}
        f = np.mean([score(run_method(g, "sauvola", p, h), gt, band)["f"] for g, h, gt, band in cache])
        sweep.append({**p, "f_val": f})
    sw = pd.DataFrame(sweep).sort_values("f_val", ascending=False)
    save_table(sw, EXP_ID, "sweep")
    best = {k: sw.iloc[0][k] for k in ("window_factor", "k", "opening")}
    best["opening"] = int(best["opening"])
    print("Sauvola chọn trên val:", best, "F =", round(sw.iloc[0]["f_val"], 4))

    rows = []
    for _, r in test.iterrows():
        gt, band = D.image(r, "clean_binary", gray=True), D.mask(r, "ignore_band")
        for sr in (True, False):
            g = prepare(r, sr)
            h = estimate_char_height(g)
            for method in binarize.METHODS:
                p = best if method == "sauvola" else {"window_factor": best["window_factor"], "opening": best["opening"]}
                rows.append({"sample_id": r["sample_id"], "recipe": r["recipe_name"], "method": method,
                             "shadow_removal": sr, **{l: r[l] for l in LEVELS},
                             **score(run_method(g, method, p, h), gt, band)})
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID)
    summ = df.groupby(["shadow_removal", "method"])[["f", "pf", "f_full", "psnr"]].mean().reset_index()
    summ = summ.sort_values(["shadow_removal", "f"], ascending=[False, False])
    save_table(summ, EXP_ID, "summary")
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
