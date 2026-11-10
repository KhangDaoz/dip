"""K3 — Gom dòng còn đúng không khi ảnh còn nghiêng 0→5°? Người làm: Khang. Kế hoạch: team/KeHoach.md mục 7.2.

Dữ liệu: GT chính thức K0 (results/k0/gt.json, 32 ảnh bộ 2). Mỗi ảnh nhị phân của ② được xoay thêm ±θ quanh tâm
(θ = 0, 0,5, 1, …, 5°); khung cụm từ GT xoay cùng ma trận rồi lấy hình chữ nhật bao (như đầu vào thật sau tách cụm từ).
Tham số cuối = bộ được chọn ở cả 5 fold của K1 (results/k2/folds.csv), không chọn lại trên ảnh xoay.
Phương án: projection ngang + xóa kẻ (đề xuất) · gom tâm y khe cố định (đối chứng) · projection + xóa kẻ SAU KHI chỉnh
nghiêng bằng ① (preprocessing.deskew.skew_projection — xoay ngược đúng góc ước lượng) để xem ① gỡ được bao nhiêu.
Độ đo: F1 cặp cụm từ cùng dòng (như K1), tỉ lệ ảnh gom đúng hoàn toàn; trung bình hai chiều xoay +θ và −θ.
Kết quả: results/k3/ (table.csv, figure.png).

Chạy (từ code/): python experiments/k3_skew.py   (sau k0_gt.py, k2_table.py)
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cleaning.pipeline import PARAMS as CLEAN_PARAMS, binarize_clean, shadow_free
from common.experiment import results_dir, save_config, save_figure, save_table
from common.io import load_json, read_image
from common.paths import DATASETS, RESULTS_DIR
from common.seed import set_seed
from preprocessing.deskew import skew_projection
from table_extraction import geometry, postprocess as PP, rows as ROWS

EXP_ID = "k3"
CONFIG = {"seed": 42, "angles": [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5],
          # tham số cuối: K1 chọn giống nhau ở cả 5 fold (results/k2/folds.csv)
          "projection": {"se_factor": 2.0, "smooth_window": 1, "valley_ratio": 0.1},
          "fixed": {"gap_px": 5}}


def pair_f1(pred, true):
    tp = fp = fn = 0
    n = len(true)
    for i in range(n):
        for j in range(i + 1, n):
            p, t = pred[i] == pred[j], true[i] == true[j]
            tp += p and t
            fp += p and not t
            fn += t and not p
    return tp, fp, fn


def labels(rows, n):
    lab = [10_000 + i for i in range(n)]
    for k, r in enumerate(rows):
        for i in r:
            lab[i] = k
    return lab


def rotate(binary, boxes, angle):
    """Xoay ảnh + khung quanh tâm ảnh (góc dương = ngược chiều kim đồng hồ, như deskew._rotate)."""
    H, W = binary.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    img = cv2.warpAffine(binary, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    out = []
    for x, y, w, h in boxes:
        pts = np.array([[x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]], float) @ M.T
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        out.append([int(round(x0)), int(round(y0)), max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))])
    return img, out


def rows_proj(binary, boxes, h):
    p = CONFIG["projection"]
    return ROWS.rows_projection(PP.remove_rule_lines(binary), boxes, h, p["se_factor"], p["smooth_window"],
                                p["valley_ratio"])


def run():
    gt = load_json(RESULTS_DIR / "k0" / "gt.json")
    recs = []
    for key, g in gt.items():
        img = read_image(DATASETS[2] / g["image"])
        b, _ = binarize_clean(shadow_free(img, CLEAN_PARAMS), CLEAN_PARAMS)
        h = geometry.char_height(b)
        boxes = [w["bbox"] for w in g["words"]]
        true = [w["row"] for w in g["words"]]
        for a in CONFIG["angles"]:
            for sign in ((1,) if a == 0 else (1, -1)):
                rb, rboxes = rotate(b, boxes, sign * a)
                est = skew_projection(rb)                       # ① ước lượng góc nghiêng
                db, dboxes = rotate(rb, rboxes, -est)            # ① xoay ngược
                for method, pred in (("projection", rows_proj(rb, rboxes, h)),
                                     ("fixed", ROWS.rows_fixed(rboxes, CONFIG["fixed"]["gap_px"])),
                                     ("deskew+projection", rows_proj(db, dboxes, h))):
                    tp, fp, fn = pair_f1(labels(pred, len(boxes)), true)
                    recs.append({"image": key, "angle": a, "sign": sign, "method": method, "tp": tp, "fp": fp,
                                 "fn": fn, "exact": fp == 0 and fn == 0, "est_angle": est})
    return pd.DataFrame(recs)


def summarize(df):
    g = df.groupby(["method", "angle"]).agg(tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
                                            exact=("exact", "mean")).reset_index()
    g["precision"] = g.tp / (g.tp + g.fp)
    g["recall"] = g.tp / (g.tp + g.fn)
    g["f1"] = 2 * g.precision * g.recall / (g.precision + g.recall)
    return g


def plot(s):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    names = {"projection": "Projection + xóa kẻ (đề xuất)", "fixed": "Tâm y, khe cố định",
             "deskew+projection": "Chỉnh nghiêng (1) rồi projection"}
    style = {"projection": "o-", "fixed": "s--", "deskew+projection": "^-"}
    for m in ("projection", "fixed", "deskew+projection"):
        d = s[s.method == m]
        ax[0].plot(d.angle, d.f1, style[m], label=names[m])
        ax[1].plot(d.angle, 100 * d.exact, style[m], label=names[m])
    ax[0].set(xlabel="Góc nghiêng thêm (°)", ylabel="F1 cặp cùng dòng", ylim=(0, 1.02))
    ax[1].set(xlabel="Góc nghiêng thêm (°)", ylabel="% ảnh gom dòng đúng hoàn toàn", ylim=(0, 102))
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    df = run()
    save_table(df, EXP_ID, "per_image")
    s = summarize(df)
    save_table(s, EXP_ID)
    save_figure(plot(s), EXP_ID)
    print(f"{df.image.nunique()} ảnh")
    print(s[["method", "angle", "f1", "precision", "recall", "exact"]].round(3).to_string())
    d = df[df.method == "projection"].copy()
    d["err"] = (d.est_angle - d.sign * d.angle).abs()          # góc còn sót sau ① (ảnh gốc coi như ~0°)
    print("Sai số ước lượng góc của ① (|θ_ước lượng − θ_xoay|, trung vị theo góc):")
    print(d.groupby("angle").err.median().round(2).to_string())


if __name__ == "__main__":
    main()
