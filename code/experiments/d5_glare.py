"""D5 — Khử lóa (H5) có giúp không? Người làm: Hiếu. Kế hoạch: KeHoach.md mục 5.1 (H4–H5), 5.2 (D5).

Dữ liệu: 30 mẫu test bộ 5 chỉ có lóa (v10 `single_glare`, mức 1–3; không biến dạng hình học nên so từng pixel được).
Ảnh tham chiếu = clean_exposed (ảnh sạch cùng mức phơi sáng, không lóa).
Mask lóa: phát hiện bằng H4 ({V > τ_V} ∩ {S < τ_S}, τ chọn ở D1) hoặc mask GT (trần trên của bước phát hiện).
Phương án: không khử · inpainting Telea (đề xuất) · Navier–Stokes · Telea với mask GT · pipeline (chỉ khử khi tỉ lệ lóa 1–8 %).
Độ đo: PSNR (C6 · S39) trong vùng lóa GT và trên cả ảnh; F-measure nhị phân (Sauvola sau ②) trong vùng lóa so với
clean_binary — đo chữ dưới lóa còn đọc được không.
Kết quả: results/d5/{table.csv, summary.csv, config.json}

Chạy (từ code/): python experiments/d5_glare.py
"""
import cv2
import numpy as np
import pandas as pd

from cleaning import metrics
from cleaning.pipeline import PARAMS as CP, binarize_clean, shadow_free
from common.experiment import results_dir, save_config, save_table
from common.seed import set_seed
from preprocessing import glare as G, quality
from preprocessing.pipeline import PARAMS as PP, resize_long_side
from synthetic import dataset as D

EXP_ID = "d5"
CONFIG = {"seed": 42, "split": "test", "recipe": "single_glare", "tau_v": PP["tau_v"], "tau_s": PP["tau_s"],
          "fix_range": PP["glare_fix_range"], "radius": 5}


def psnr_in(a, b, m):
    a, b = a.astype(float), b.astype(float)
    sel = m > 0
    if not sel.any():
        return np.nan
    mse = np.mean((a[sel] - b[sel]) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def f_in(b, gt, m):
    sel = m > 0
    return metrics.f_measure(np.where(sel, b, 0), np.where(sel, gt, 0))


def detect(img):
    small, _ = resize_long_side(img, 1000)
    m = quality.glare_mask(small, CONFIG["tau_v"], CONFIG["tau_s"])
    return cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST), quality.glare_ratio(m)


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    m = D.manifest(CONFIG["split"])
    m = m[m["recipe_name"] == CONFIG["recipe"]]
    rows = []
    for _, r in m.iterrows():
        img = D.image(r)
        ref = D.image(r, "clean_exposed")
        gt_mask = D.mask(r, "glare_mask")
        gt_bin = D.image(r, "clean_binary", gray=True)
        det, ratio = detect(img)
        lo, hi = CONFIG["fix_range"]
        out = {"none": img,
               "telea": G.remove_glare(img, det, "telea", CONFIG["radius"]),
               "ns": G.remove_glare(img, det, "ns", CONFIG["radius"]),
               "telea_gt_mask": G.remove_glare(img, gt_mask, "telea", CONFIG["radius"]),
               "pipeline": G.remove_glare(img, det, "telea", CONFIG["radius"]) if lo <= ratio <= hi else img}
        for k, x in out.items():
            b, _ = binarize_clean(shadow_free(x, CP), CP)
            rows.append({"sample_id": r["sample_id"], "glare_level": r["glare_level"], "method": k,
                         "detected_ratio": ratio, "mask_iou": metrics.mask_iou(det, gt_mask),
                         "psnr_glare": psnr_in(x, ref, gt_mask), "psnr_full": metrics.psnr(x, ref),
                         "f_glare": f_in(b, gt_bin, gt_mask)})
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID)
    s = df.groupby("method")[["psnr_glare", "psnr_full", "f_glare"]].mean().reset_index()
    lv = df.pivot_table(index="glare_level", columns="method", values="f_glare", aggfunc="mean")
    save_table(s, EXP_ID, "summary")
    save_table(lv.reset_index(), EXP_ID, "by_level")
    print(s.round(3).to_string())
    print(lv.round(3).to_string())
    print("Tỉ lệ mẫu pipeline chạy khử lóa:", round(float(((df.method == "pipeline") & (df.detected_ratio.between(lo, hi))).sum()
                                                       / (df.method == "pipeline").sum()), 3))


if __name__ == "__main__":
    main()
