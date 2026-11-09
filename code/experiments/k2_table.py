"""K1, K2, K4 — Gom dòng, tìm cột, SE thích nghi. Người làm: Khang. Kế hoạch: KeHoach.md mục 7.1–7.3.

Dữ liệu: 57 ảnh gốc bộ 2, GT từ K0. Kiểm định chéo 5-fold (C8 · S41): chọn tham số trên 4 fold (grid search),
đo trên fold còn lại; mỗi ảnh được đánh giá đúng 1 lần ⇒ tập đánh giá = 57 ảnh.
Đầu vào: ảnh nhị phân của ② (khử bóng + Sauvola) + box cụm từ GT (cô lập bước gom dòng / cột khỏi bước tách cụm từ).
K1 gom dòng: projection trên ảnh Dilation ngang (đề xuất) vs gom tâm y với khe cố định.
K2 tìm cột (dòng GT cho trước): projection dọc (đề xuất) vs XY-cut đệ quy vs khe x cố định.
K4: tham số theo chiều cao chữ h của từng ảnh (đề xuất) vs h cố định = trung vị cả bộ (pixel cố định).
Độ đo: pairwise F1 quan hệ "cùng dòng" (K1) / "cùng cột" (K2) (C8 · S49); tỉ lệ ảnh gom dòng đúng hoàn toàn.
Kết quả: results/k2/

Chạy (từ code/): python experiments/k2_table.py   (sau k0_gt.py)
"""
import itertools

import numpy as np
import pandas as pd

from cleaning.pipeline import PARAMS as CLEAN_PARAMS, binarize_clean, shadow_free
from common.experiment import results_dir, save_config, save_table
from common.io import load_json, read_image
from common.paths import DATASETS, RESULTS_DIR
from common.seed import set_seed
from table_extraction import cells as C, columns as COL, geometry, rows as ROWS

EXP_ID = "k2"
CONFIG = {
    "seed": 42, "folds": 5,
    "rows": {"projection": {"se_factor": [2, 3, 4, 5], "valley_ratio": [0.02, 0.05, 0.1], "smooth_window": [1, 3, 5]},
             "fixed": {"gap_px": [3, 5, 8, 10, 15]}},
    "cols": {"projection": {"gap_factor": [0.5, 1.0, 1.5, 2.0], "min_cross": [0.4, 0.6, 0.8]},
             "xycut": {"min_gap_factor": [0.5, 1.0, 1.5, 2.0]},
             "fixed": {"gap_px": [5, 10, 20, 30]}},
}


def pair_f1(pred_groups, true_groups):
    """F1 trên tập cặp (i, j) cùng nhóm. pred_groups / true_groups: list nhãn nhóm theo chỉ số cụm từ."""
    n = len(true_groups)
    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            p, t = pred_groups[i] == pred_groups[j], true_groups[i] == true_groups[j]
            tp += p and t
            fp += p and not t
            fn += t and not p
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return p, r, (2 * p * r / (p + r) if p + r else 0)


def rows_labels(rows, n):
    lab = [-1] * n
    for k, r in enumerate(rows):
        for i in r:
            lab[i] = k
    return [l if l >= 0 else 10_000 + i for i, l in enumerate(lab)]


def run_rows(item, method, p, h):
    words = [w["bbox"] for w in item["words"]]
    if method == "projection":
        rws = ROWS.rows_projection(item["binary"], words, h, p["se_factor"], p["smooth_window"], p["valley_ratio"])
    else:
        rws = ROWS.rows_fixed(words, p["gap_px"])
    return rows_labels(rws, len(words))


def run_cols(item, method, p, h):
    words = [w["bbox"] for w in item["words"]]
    true_rows = {}
    for i, w in enumerate(item["words"]):
        true_rows.setdefault(w["row"], []).append(i)
    rws = list(true_rows.values())
    if method == "projection":
        cols = COL.columns_projection(item["binary"], rws, words, h, p["gap_factor"], p["min_cross"])
    elif method == "xycut":
        cols = COL.columns_xycut(item["binary"], words, h, p["min_gap_factor"])
    else:
        cols = COL.columns_fixed(words, p["gap_px"])
    return [C.column_of(w, cols) if cols else 0 for w in words]


def score(items, task, method, p, h_mode):
    tp = fp = fn = 0
    exact = []
    for it in items:
        h = it["h"] if h_mode == "adaptive" else it["h_fixed"]
        if task == "rows":
            pred, true = run_rows(it, method, p, h), [w["row"] for w in it["words"]]
        else:
            pred, true = run_cols(it, method, p, h), [w["role"] for w in it["words"]]
        a, b, c = pair_f1(pred, true)
        tp, fp, fn = tp + a, fp + b, fn + c
        exact.append(b == 0 and c == 0)
    P, R, F = prf(tp, fp, fn)
    return {"precision": P, "recall": R, "f1": F, "exact": float(np.mean(exact))}


def grid(d):
    return [dict(zip(d, v)) for v in itertools.product(*d.values())]


def main():
    rng = set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    gt = load_json(RESULTS_DIR / "k0" / "gt.json")
    items = []
    for key, g in gt.items():
        img = read_image(DATASETS[2] / g["image"])
        b, _ = binarize_clean(shadow_free(img, CLEAN_PARAMS), CLEAN_PARAMS)
        items.append({"key": key, "binary": b, "words": g["words"], "h": geometry.char_height(b)})
    h_fixed = float(np.median([it["h"] for it in items]))
    for it in items:
        it["h_fixed"] = h_fixed
    fold = rng.permutation(len(items)) % CONFIG["folds"]

    rows_out, chosen = [], []
    for task in ("rows", "cols"):
        for method, params in CONFIG[task].items():
            for h_mode in (("adaptive", "fixed") if method != "fixed" else ("adaptive",)):
                tot = {"tp": 0}
                per_fold = []
                for f in range(CONFIG["folds"]):
                    tr = [it for it, k in zip(items, fold) if k != f]
                    te = [it for it, k in zip(items, fold) if k == f]
                    best = max(grid(params), key=lambda p: score(tr, task, method, p, h_mode)["f1"])
                    s = score(te, task, method, best, h_mode)
                    per_fold.append(s)
                    chosen.append({"task": task, "method": method, "h_mode": h_mode, "fold": f, **best, **s})
                rows_out.append({"task": task, "method": method, "h_mode": h_mode,
                                 "f1_mean": np.mean([s["f1"] for s in per_fold]),
                                 "f1_std": np.std([s["f1"] for s in per_fold]),
                                 "precision": np.mean([s["precision"] for s in per_fold]),
                                 "recall": np.mean([s["recall"] for s in per_fold]),
                                 "exact_mean": np.mean([s["exact"] for s in per_fold])})
    save_table(pd.DataFrame(chosen), EXP_ID, "folds")
    df = pd.DataFrame(rows_out)
    save_table(df, EXP_ID, "summary")
    print(f"{len(items)} ảnh; h cố định = {h_fixed:.1f} px")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
