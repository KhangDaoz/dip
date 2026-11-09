"""D4 — Chỉnh nghiêng: Projection vs Hough vs Radon. Người làm: Hiếu. Kế hoạch: KeHoach.md mục 5.1 (H8), 5.2.

Dữ liệu: bộ 5 test, mẫu CHỈ XOAY (d4_pure_rotate: ±2, ±5, ±10, ±15°) + ảnh sạch tương ứng (0°). Chạy chỉnh nghiêng
trực tiếp trên ảnh nhị phân (không nắn phối cảnh trước, vì nắn xong thì không còn góc để đo).
Phương án: projection θ* = argmax Var(P_θ) (➕ C3 · S23, đề xuất) · Hough · Radon.
Độ đo: sai số góc tuyệt đối (°), thời gian (ms).
Kết quả: results/d4/

Chạy (từ code/): python experiments/d4_deskew.py
"""
import time

import cv2
import numpy as np
import pandas as pd

from common.experiment import results_dir, save_config, save_table
from common.seed import set_seed
from preprocessing import deskew
from preprocessing.pipeline import ink_binary, resize_long_side
from synthetic import dataset as D

EXP_ID = "d4"
CONFIG = {"seed": 42, "long_side": 1000, "max_angle": 20.0, "step": 0.25}
METHODS = {"projection": lambda b: deskew.skew_projection(b, CONFIG["max_angle"], CONFIG["step"]),
           "hough": lambda b: deskew.skew_hough(b, CONFIG["max_angle"]),
           "radon": lambda b: deskew.skew_radon(b, CONFIG["max_angle"], 0.5)}


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    man = D.manifest("test")
    rot = man[man["recipe_name"].str.startswith("d4_pure_rotate")]
    items = [(r["sample_id"], float(r["angle_deg"]), D.image(r)) for _, r in rot.iterrows()]
    seen = set()
    for _, r in rot.iterrows():           # 0°: ảnh sạch của cùng ảnh gốc
        if r["source_id"] not in seen:
            seen.add(r["source_id"])
            items.append((r["sample_id"] + "_clean", 0.0, D.image(r, "clean")))
    rows = []
    for sid, ang, img in items:
        small, _ = resize_long_side(img, CONFIG["long_side"])
        b = ink_binary(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        for m, fn in METHODS.items():
            t0 = time.perf_counter()
            est = fn(b)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"sample_id": sid, "angle": ang, "method": m, "estimate": est,
                         "abs_err": abs(est - ang), "ms": ms})
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID)
    summ = df.groupby("method").agg(n=("abs_err", "size"), mean_err=("abs_err", "mean"),
                                    median_err=("abs_err", "median"), within_0_5=("abs_err", lambda v: (v <= 0.5).mean()),
                                    ms=("ms", "median")).reset_index()
    save_table(summ, EXP_ID, "summary")
    print(summ.round(3).to_string())
    print(df.pivot_table(index="angle", columns="method", values="abs_err").round(2).to_string())


if __name__ == "__main__":
    main()
