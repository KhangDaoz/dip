"""S2 — Tách khoản thuốc bền đến đâu khi ảnh bị hỏng? Người làm: Thắng. Kế hoạch: KeHoach.md mục 8.3 (hình chính).

Dữ liệu: test bộ 5 (450 mẫu, 30 ảnh gốc × biến thể). Chạy ĐẦU-CUỐI như app: ① preprocess (doc_type cho trước) →
② clean → ④ segment (tham số từ S1, S4). Khoản dự đoán (trên ảnh đã nắn) chiếu về ảnh hỏng bằng H⁻¹ của ①, rồi về
ảnh sạch bằng H_degraded_to_clean của bộ 5, lấy khung bao → so với GT khoản trên ảnh sạch (dải ngang, như S0).
Độ đo: F1 khoản (IoU > 0,5) theo từng loại / mức hỏng.
Kết quả: results/s2/{table.csv (từng mẫu), by_level.csv, figure.png}

Chạy (từ code/): python experiments/s2_robustness.py
"""
import ast
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cleaning.pipeline import clean
from common.experiment import results_dir, save_config, save_figure, save_table
from common.io import read_image
from common.paths import REPO_DIR, RESULTS_DIR
from common.seed import set_seed
from prescription_segmentation import metrics as M
from prescription_segmentation.pipeline import segment
from preprocessing.pipeline import preprocess
from synthetic import dataset as D

EXP_ID = "s2"
CONFIG = {"seed": 42, "split": "test", "iou_threshold": 0.5}
FACTORS = [("angle_deg", "Góc nghiêng (°)", lambda v: abs(v)), ("perspective_strength", "Mức phối cảnh", None),
           ("blur_level", "Mức mờ", None), ("shadow_level", "Mức bóng", None), ("jpeg_quality", "Chất lượng JPEG", None),
           ("glare_level", "Mức lóa", None), ("stamp_level", "Mức dấu đỏ", None), ("noise_sigma", "Nhiễu σ", None)]


def params():
    p = {}
    s1 = pd.read_csv(RESULTS_DIR / "s1" / "summary.csv")
    r = s1[(s1["method"] == "dilation_rule") & (s1["lines"] == "projection")].iloc[0]
    p.update(ast.literal_eval(r["params"]))
    s4 = RESULTS_DIR / "s4" / "summary.csv"
    if s4.exists():
        r4 = pd.read_csv(s4).iloc[0]
        p.update(region_k=float(r4["region_k"]), max_follow=int(r4["max_follow"]))
    return p


def project(boxes, H):
    out = []
    for x, y, w, h in boxes:
        pts = np.array([[x, y, 1], [x + w, y, 1], [x + w, y + h, 1], [x, y + h, 1]], float) @ np.asarray(H).T
        pts = pts[:, :2] / pts[:, 2:3]
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        out.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
    return out


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    p = params()
    save_config({**CONFIG, "segment_params": p}, EXP_ID)
    man = D.manifest(CONFIG["split"])
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for _, row in man.iterrows():
            gt = D.label(row)
            gt_items = D.items_band(gt, "clean")
            rec = {k: row[k] for k in ("sample_id", "recipe_name", *[f for f, _, _ in FACTORS])}
            try:
                img = D.image(row)
                pre = preprocess(img, row["sample_id"], tmp, doc_type="prescription")
                cl = clean(read_image(REPO_DIR / pre.rectified_path), "prescription", row["sample_id"], tmp)
                seg = segment(read_image(REPO_DIR / cl.binary_path, gray=True), p)
                # ảnh đã nắn → ảnh hỏng (H⁻¹ của ①) → ảnh sạch (H_degraded_to_clean của bộ 5)
                H = np.asarray(gt["H_degraded_to_clean"]) @ np.linalg.inv(np.asarray(pre.homography))
                pred = project(seg["items"], H)
                m = M.match(pred, gt_items, CONFIG["iou_threshold"])
                rec.update(tp=m["tp"], fp=m["fp"], fn=m["fn"], mean_iou=float(np.mean(m["ious"])) if m["ious"] else 0,
                           corner_method=pre.corner_method, error="")
            except Exception as e:
                rec.update(tp=0, fp=0, fn=len(gt_items), mean_iou=0, corner_method="", error=f"{type(e).__name__}: {e}")
            rows.append(rec)
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID)
    aggregate(df)


CLEAN = {"angle_deg": 0, "perspective_strength": 0, "blur_level": 0, "shadow_level": 0, "jpeg_quality": 100,
         "glare_level": 0, "stamp_level": 0, "noise_sigma": 0}
PLOT = ["angle_deg", "perspective_strength", "blur_level", "shadow_level", "glare_level", "stamp_level"]


def aggregate(df):
    """F1 theo mức của từng loại hỏng. Mỗi đường chỉ gồm mẫu mà MỌI loại hỏng khác ở mức sạch (mức 0 = mẫu sạch hoàn
    toàn); riêng góc nghiêng và phối cảnh xét chung vì lưới d3 luôn có cả hai."""
    P, R, F = M.prf(df["tp"].sum(), df["fp"].sum(), df["fn"].sum())
    print(f"Toàn bộ test bộ 5: P {P:.3f} R {R:.3f} F1 {F:.3f}; lỗi chạy: {(df['error'].fillna('') != '').sum()}")
    lv = []
    for f, title, fn in FACTORS:
        geo = {"angle_deg", "perspective_strength"}
        others = [o for o in CLEAN if o != f and not (f in geo and o in geo)]
        sub = df[(df[others] == pd.Series({o: CLEAN[o] for o in others})).all(axis=1)]
        key = sub[f].map(fn) if fn else sub[f]
        for v, g in sub.groupby(key):
            P, R, F = M.prf(g["tp"].sum(), g["fp"].sum(), g["fn"].sum())
            lv.append({"factor": f, "level": v, "n": len(g), "precision": P, "recall": R, "f1": F})
    lvdf = pd.DataFrame(lv)
    save_table(lvdf, EXP_ID, "by_level")
    print(lvdf.round(3).to_string())
    fig, axes = plt.subplots(1, len(PLOT), figsize=(16, 3.4), sharey=True)
    for ax, f in zip(axes, PLOT):
        title = dict((a, b) for a, b, _ in FACTORS)[f]
        g = lvdf[lvdf["factor"] == f]
        ax.plot(g["level"], g["f1"], "o-", color="#c44")
        for x, y, n in zip(g["level"], g["f1"], g["n"]):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("F1 khoản thuốc")
    fig.suptitle("S2 — Tách khoản thuốc đầu-cuối (1)→(2)→(4) theo mức hỏng (test bộ 5, mỗi đường một loại hỏng)")
    fig.tight_layout()
    save_figure(fig, EXP_ID)

if __name__ == "__main__":
    import sys
    if "--aggregate" in sys.argv:   # chỉ tính lại bảng / hình từ results/s2/table.csv
        aggregate(pd.read_csv(RESULTS_DIR / "s2" / "table.csv"))
    else:
        main()
