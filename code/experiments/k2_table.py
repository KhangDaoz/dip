"""K1, K2, K4 — Gom dòng, tìm cột, SE thích nghi. Người làm: Khang. Kế hoạch: KeHoach.md mục 7.1–7.3.

Dữ liệu: GT chính thức của K0 (results/k0/gt.json — 32 ảnh bộ 2 giữ lại sau soát tay). Kiểm định chéo 5-fold (C8 · S41): chọn tham số trên 4 fold (grid search),
đo trên fold còn lại; mỗi ảnh được đánh giá đúng 1 lần ⇒ tập đánh giá = toàn bộ ảnh GT.
Đầu vào: ảnh nhị phân của ② (khử bóng + Sauvola) + box cụm từ GT (cô lập bước gom dòng / cột khỏi bước tách cụm từ).
Tách cụm từ (bước 2 của ③): Dilation SE (v·h) × (a·h) ± Opening 2×2, so khung dự đoán với khung GT (IoU ≥ 0,5).
K1 gom dòng: projection trên ảnh Dilation ngang (đề xuất) vs gom tâm y với khe cố định.
K2 tìm cột (dòng GT cho trước): projection dọc (đề xuất) vs XY-cut đệ quy vs khe x cố định.
K4: tham số theo chiều cao chữ h của từng ảnh (đề xuất) vs h cố định = trung vị cả bộ (pixel cố định).
"+nl": chiếu trên ảnh đã xóa đường kẻ (Opening SE đường thẳng W/20, H/20 — C7 · S6, S28; postprocess.remove_rule_lines).
Thêm sau phân tích lỗi: một số phiếu bộ 2 CÓ kẻ, kẻ dọc lấp mọi thung lũng của P(y) → cả trang thành 1 dòng.
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
from table_extraction import cells as C, columns as COL, geometry, postprocess as PP, rows as ROWS

EXP_ID = "k2"
CONFIG = {
    "seed": 42, "folds": 5,
    "rows": {"projection": {"se_factor": [2, 3, 4, 5], "valley_ratio": [0.02, 0.05, 0.1], "smooth_window": [1, 3, 5]},
             "projection+nl": {"se_factor": [2, 3, 4, 5], "valley_ratio": [0.02, 0.05, 0.1], "smooth_window": [1, 3, 5]},
             "fixed": {"gap_px": [3, 5, 8, 10, 15]}},
    # bước tách cụm từ (KeHoach.md 7.3 dòng đầu): SE Dilation (v·h) × (a·h), có / không Opening 2×2 trước
    "words": {"dilation": {"a": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], "v": [0.0, 0.5, 1.0], "opening": [0, 2]}},
    "cols": {"projection": {"gap_factor": [0.5, 1.0, 1.5, 2.0], "min_cross": [0.4, 0.6, 0.8]},
             "projection+nl": {"gap_factor": [0.5, 1.0, 1.5, 2.0], "min_cross": [0.4, 0.6, 0.8]},
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
    if method.startswith("projection"):
        b = item["binary_nl"] if method.endswith("+nl") else item["binary"]
        rws = ROWS.rows_projection(b, words, h, p["se_factor"], p["smooth_window"], p["valley_ratio"])
    else:
        rws = ROWS.rows_fixed(words, p["gap_px"])
    return rows_labels(rws, len(words))


def run_cols(item, method, p, h):
    words = [w["bbox"] for w in item["words"]]
    true_rows = {}
    for i, w in enumerate(item["words"]):
        true_rows.setdefault(w["row"], []).append(i)
    rws = list(true_rows.values())
    if method.startswith("projection"):
        b = item["binary_nl"] if method.endswith("+nl") else item["binary"]
        cols = COL.columns_projection(b, rws, words, h, p["gap_factor"], p["min_cross"])
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


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i) if i else 0.0


def score_words(items, p, h_mode):
    """F1 khớp khung cụm từ dự đoán ↔ GT (IoU ≥ 0,5, mỗi khung dự đoán khớp tối đa một GT), trong vùng thân bảng."""
    tp = fp = fn = 0
    exact = []
    for it in items:
        h = it["h"] if h_mode == "adaptive" else it["h_fixed"]
        b = it["binary_open_nl"] if p["opening"] else it["binary_nl"]
        X0, Y0, X1, Y1 = it["body"]
        pred = [w for w in geometry.words_from_binary(b, h, p["a"], v=p["v"])
                if w[0] >= X0 and w[1] >= Y0 and w[0] + w[2] <= X1 and w[1] + w[3] <= Y1]
        used, m = set(), 0
        for g in (w["bbox"] for w in it["words"]):
            best = max(range(len(pred)), key=lambda i: iou(g, pred[i]), default=None)
            if best is not None and best not in used and iou(g, pred[best]) >= 0.5:
                used.add(best)
                m += 1
        tp, fp, fn = tp + m, fp + len(pred) - m, fn + len(it["words"]) - m
        exact.append(m == len(it["words"]) == len(pred))
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
        bb = [w["bbox"] for w in g["words"]]
        body = (min(x for x, _, _, _ in bb) - 6, min(y for _, y, _, _ in bb) - 6,
                max(x + w for x, _, w, _ in bb) + 6, max(y + hh for _, y, _, hh in bb) + 6)
        items.append({"key": key, "binary": b, "binary_nl": PP.remove_rule_lines(b),
                      "binary_open_nl": PP.remove_rule_lines(PP.opening_before_projection(b, 2)),
                      "words": g["words"], "h": geometry.char_height(b), "body": body})
    h_fixed = float(np.median([it["h"] for it in items]))
    for it in items:
        it["h_fixed"] = h_fixed
    fold = rng.permutation(len(items)) % CONFIG["folds"]

    rows_out, chosen = [], []
    for task in ("words", "rows", "cols"):
        for method, params in CONFIG[task].items():
            for h_mode in (("adaptive", "fixed") if method != "fixed" else ("adaptive",)):
                tot = {"tp": 0}
                per_fold = []
                for f in range(CONFIG["folds"]):
                    tr = [it for it, k in zip(items, fold) if k != f]
                    te = [it for it, k in zip(items, fold) if k == f]
                    sc = (lambda its, p: score_words(its, p, h_mode)) if task == "words" else \
                        (lambda its, p: score(its, task, method, p, h_mode))
                    best = max(grid(params), key=lambda p: sc(tr, p)["f1"])
                    s = sc(te, best)
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
