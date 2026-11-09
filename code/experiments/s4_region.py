"""S4 — Khoanh vùng thuốc có giúp không? Người làm: Thắng. Kế hoạch: KeHoach.md mục 8.1 (S-B1), 8.3, 8.4.

Dữ liệu: VAIPE `valid/` chọn k (SE Dilation dọc k·h) và max_follow; `test/` báo kết quả. Vùng chuẩn = khung bao box
`drugname` / `usage` / `quantity` (S0). Tham số gom khoản lấy từ S1 (results/s1/summary.csv).
So sánh: khoanh vùng tự động (đề xuất) vs lấy cả trang (đối chứng) vs vùng GT (trần trên).
Độ đo: IoU vùng; F1 khoản đầu-cuối (sau ②, S-B1 → S-B4).
Kết quả: results/s4/

Chạy (từ code/): python experiments/s4_region.py
"""
import ast

import numpy as np
import pandas as pd

from cleaning.pipeline import PARAMS as CLEAN_PARAMS, binarize_clean, shadow_free
from common.experiment import results_dir, save_config, save_table
from common.io import load_json, read_image
from common.paths import DATASETS, RESULTS_DIR
from common.seed import set_seed
from prescription_segmentation import lines as L, metrics as M, region as R
from prescription_segmentation.pipeline import segment

EXP_ID = "s4"
CONFIG = {"seed": 42, "grid": {"region_k": [1.0, 1.5, 2.0, 3.0, 4.0], "max_follow": [1, 2]}}


def s1_params():
    df = pd.read_csv(RESULTS_DIR / "s1" / "summary.csv")
    row = df[(df["method"] == "dilation_rule") & (df["lines"] == "projection")].iloc[0]
    return ast.literal_eval(row["params"])


def load(split):
    gt = load_json(RESULTS_DIR / "s0" / f"gt_items_{split}.json")
    out = {}
    for img_id, g in gt.items():
        img = read_image(DATASETS[1] / split / "images" / f"{img_id}.png")
        out[img_id] = (binarize_clean(shadow_free(img, CLEAN_PARAMS), CLEAN_PARAMS)[0], g)
    return out


def run(data, p, use_region=True):
    reg_iou, res = [], []
    for b, g in data.values():
        h = L.char_height(b)
        reg = R.find_drug_region(b, h, p["region_k"], p.get("bold_ratio", 1.5), p["max_follow"]) \
            if use_region else [0, 0, b.shape[1], b.shape[0]]
        x, y, w, hh = reg
        sub = np.zeros_like(b)
        sub[y:y + hh, x:x + w] = b[y:y + hh, x:x + w]
        s = segment(sub, {**p, "use_region": False})
        reg_iou.append(M.iou(reg, g["region"]))
        res.append(M.match(s["items"], [it["bbox"] for it in g["items"]]))
    return {"region_iou": float(np.mean(reg_iou)), **M.summarize(res)}


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    p1 = s1_params()
    save_config({**CONFIG, "s1_params": p1}, EXP_ID)
    valid, test = load("valid"), load("test")
    sweep = []
    for k in CONFIG["grid"]["region_k"]:
        for mf in CONFIG["grid"]["max_follow"]:
            p = {**p1, "region_k": k, "max_follow": mf}
            sweep.append({"region_k": k, "max_follow": mf, **run(valid, p)})
    sw = pd.DataFrame(sweep)
    save_table(sw, EXP_ID, "sweep")
    best = sw.sort_values(["f1", "region_iou"], ascending=False).iloc[0]
    p = {**p1, "region_k": float(best["region_k"]), "max_follow": int(best["max_follow"])}
    rows = [{"method": "khoanh vùng tự động (đề xuất)", **p, **run(test, p, True)},
            {"method": "lấy cả trang (đối chứng)", **p, **run(test, p, False)}]
    gtrow = M.summarize([M.match(segment(np.where(_mask(b, g["region"]), b, 0), {**p, "use_region": False})["items"],
                                 [it["bbox"] for it in g["items"]]) for b, g in test.values()])
    rows.append({"method": "vùng GT (trần trên)", **p, "region_iou": 1.0, **gtrow})
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID, "summary")
    print(sw.to_string())
    print(df[["method", "region_k", "max_follow", "region_iou", "precision", "recall", "f1", "mean_iou"]].to_string())


def _mask(b, reg):
    m = np.zeros(b.shape, bool)
    x, y, w, h = reg
    m[max(y - 10, 0):y + h + 10, max(x - 10, 0):x + w + 10] = True
    return m


if __name__ == "__main__":
    main()
