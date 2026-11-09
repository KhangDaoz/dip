"""S1 — Cách gom khoản thuốc nào tốt nhất? Người làm: Thắng. Kế hoạch: KeHoach.md mục 8.1 (S-B2, S-B3, S-B4), 8.3, 8.4.

Dữ liệu: VAIPE `valid/` (176 ảnh) chọn tham số, `test/` (176 ảnh) báo kết quả — GT khoản từ S0 (dải ngang).
Cô lập bước gom khoản: cắt theo vùng thuốc GT (lỗi khoanh vùng đo riêng ở S4). Ảnh nhị phân = bước ② (khử bóng + Sauvola).
Phương án (mục 8.1 S-B3): dilation_rule (đề xuất) · each_line (a) · fixed_gap (b) · fixed_count (c) · dilation (d) ·
closing (f); tách dòng bằng projection (đề xuất) hoặc components (đối chứng). Hậu xử lý S-B4: opening + bỏ khoản giả.
Độ đo: F1 khoản (khớp khi IoU > 0,5), IoU trung bình các cặp khớp (C8 · S48–49).
Kết quả: results/s1/{sweep.csv, table.csv, summary.csv, figure.png, config.json}

Chạy (từ code/): python experiments/s1_items.py
"""
import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cleaning.pipeline import PARAMS as CLEAN_PARAMS, binarize_clean, shadow_free
from common.experiment import results_dir, save_config, save_figure, save_table
from common.io import load_json, read_image
from common.paths import DATASETS, RESULTS_DIR
from common.seed import set_seed
from prescription_segmentation import items as I, lines as L, metrics as M

EXP_ID = "s1"
CONFIG = {
    "seed": 42, "gt": "s0/gt_items_<split>.json", "crop": "gt_region", "pad": 10, "iou_threshold": 0.5,
    "line_methods": ["projection", "components"],
    "grid": {
        "dilation_rule": {"s": [0.5, 1, 1.5, 2, 3, 4], "bold_ratio": [1.15, 1.25, 1.5, 2.0], "opening": [1, 2]},
        "dilation": {"s": [0.5, 1, 1.5, 2, 3, 4]},
        "closing": {"s": [0.5, 1, 1.5, 2, 3, 4]},
        "fixed_gap": {"gap_px": [3, 5, 10, 15, 20, 30]},
        "fixed_count": {"n": [1, 2, 3]},
        "each_line": {},
    },
}
LABELS = {"dilation_rule": "Dilation dọc + luật dòng đậm (đề xuất)", "each_line": "(a) mỗi dòng một khoản",
          "fixed_gap": "(b) khe cố định", "fixed_count": "(c) số dòng cố định", "dilation": "(d) chỉ Dilation dọc",
          "closing": "(f) Closing dọc"}


def load(split):
    gt = load_json(RESULTS_DIR / "s0" / f"gt_items_{split}.json")
    data = {}
    for img_id, g in gt.items():
        img = read_image(DATASETS[1] / split / "images" / f"{img_id}.png")
        b, _ = binarize_clean(shadow_free(img, CLEAN_PARAMS), CLEAN_PARAMS)
        data[img_id] = (b, g)
    return data


def run_one(b, g, method, line_method, p):
    pad = CONFIG["pad"]
    x, y, w, h = g["region"]
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    sub = b[y0:y + h + pad, x0:x + w + pad]
    op = p.get("opening", 1)
    if op > 1:
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((op, op), np.uint8))
    ch = L.char_height(b)
    ls = L.lines_projection(sub, ch) if line_method == "projection" else L.lines_components(sub, ch)
    its = I.drop_fake_items(sub, I.group(method, sub, ls, ch, **p), ch)
    its = [[a + x0, c + y0, d, e] for a, c, d, e in its]
    return M.match(its, [it["bbox"] for it in g["items"]], CONFIG["iou_threshold"])


def evaluate(data, method, line_method, p):
    return M.summarize([run_one(b, g, method, line_method, p) for b, g in data.values()])


def grid(params):
    keys = list(params)
    return [dict(zip(keys, v)) for v in itertools.product(*params.values())] if keys else [{}]


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    valid, test = load("valid"), load("test")

    sweep, best = [], {}
    for method, params in CONFIG["grid"].items():
        for lm in CONFIG["line_methods"]:
            runs = [(p, evaluate(valid, method, lm, p)) for p in grid(params)]
            for p, s in runs:
                sweep.append({"method": method, "lines": lm, **p, **s})
            # nhiều cấu hình HÒA trên valid → chọn cấu hình ở GIỮA vùng hòa (ổn định nhất, ít nhạy sai lệch dữ liệu)
            top = max(s["f1"] for _, s in runs)
            tied = sorted([(p, s) for p, s in runs if s["f1"] >= top - 1e-9], key=lambda t: tuple(t[0].values()))
            best[(method, lm)] = tied[len(tied) // 2]
    save_table(pd.DataFrame(sweep), EXP_ID, "sweep")

    rows = []
    for (method, lm), (p, sv) in best.items():
        st = evaluate(test, method, lm, p)
        rows.append({"method": method, "label": LABELS[method], "lines": lm, "params": p,
                     "valid_f1": sv["f1"], **{f"test_{k}": v for k, v in st.items()}})
    df = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    save_table(df, EXP_ID, "summary")
    print(df[["label", "lines", "params", "valid_f1", "test_precision", "test_recall", "test_f1",
              "test_mean_iou"]].to_string())

    proj = df[df["lines"] == "projection"].sort_values("test_f1")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(proj["label"], proj["test_f1"], color=["#c44" if m == "dilation_rule" else "#89a" for m in proj["method"]])
    for i, (f, u) in enumerate(zip(proj["test_f1"], proj["test_mean_iou"])):
        ax.text(f + 0.01, i, f"F1 {f:.3f} · IoU {u:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("F1 khoản thuốc trên test VAIPE (IoU > 0,5)")
    ax.set_title("S1 — Gom khoản thuốc (tách dòng bằng projection)")
    fig.tight_layout()
    save_figure(fig, EXP_ID)


if __name__ == "__main__":
    main()
