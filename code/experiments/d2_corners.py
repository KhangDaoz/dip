"""D2 + D3 — Tìm 4 góc và sai số nắn theo mức méo. Người làm: Hiếu. Kế hoạch: KeHoach.md mục 5.1 (H6–H7), 5.2, 5.3.

Dữ liệu: bộ 5, mẫu có phối cảnh (perspective_strength > 0; mẫu chỉ xoay bị mất góc ngoài khung nên bỏ).
Val chọn ngưỡng Canny (T2, tỉ lệ T1 = 2·T2 theo C5 · S47 hoặc 3·T2); test báo kết quả.
D2: contour (Canny → đường bao → tứ giác, đề xuất) · Hough · minAreaRect · kết hợp có dự phòng. Độ đo: sai số trung bình
4 góc (px, % đường chéo); thành công khi sai số < 2 % đường chéo.
D3: sai số của phương án kết hợp theo góc nghiêng × mức phối cảnh (lưới d3_grid của test) → đường cong (hình chính).
Kết quả: results/d2/

Chạy (từ code/): python experiments/d2_corners.py
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.experiment import results_dir, save_config, save_figure, save_table
from common.seed import set_seed
from preprocessing import corners as C
from preprocessing import denoise
from preprocessing.pipeline import resize_long_side
from synthetic import dataset as D

EXP_ID = "d2"
CONFIG = {"seed": 42, "long_side": 1000, "canny_low": [30, 50, 70, 100], "canny_ratio": [2, 3],
          "success_ratio": 0.02}


def detect(gray_s, method, lo, hi):
    if method == "contour":
        return C.corners_contour(gray_s, lo, hi), "contour"
    if method == "hough":
        return C.corners_hough(gray_s, lo, hi), "hough"
    if method == "minrect":
        return C.corners_minrect(gray_s), "minrect"
    return C.find_corners(gray_s, lo, hi)


def error(pred, gt, scale, diag):
    if pred is None:
        return np.inf, np.inf
    e = np.linalg.norm(np.asarray(pred) / scale - np.asarray(gt), axis=1).mean()
    return float(e), float(e / diag)


def run(rows, methods, lo, hi):
    out = []
    for _, r in rows.iterrows():
        gt = np.asarray(D.label(r)["corners"])
        img = D.image(r)
        small, s = resize_long_side(img, CONFIG["long_side"])
        g = denoise.median(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 3)
        diag = float(np.hypot(*img.shape[:2]))
        for m in methods:
            q, used = detect(g, m, lo, hi)
            if used == "full_image":
                q = None
            e, er = error(q, gt, s, diag)
            out.append({"sample_id": r["sample_id"], "recipe": r["recipe_name"], "angle": abs(r["angle_deg"]),
                        "strength": r["perspective_strength"], "method": m, "used": used, "err_px": e,
                        "err_ratio": er, "success": er < CONFIG["success_ratio"]})
    return pd.DataFrame(out)


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    man = D.manifest()
    man = man[man["perspective_strength"] > 0]
    val, test = man[man["split"] == "val"], man[man["split"] == "test"]

    sweep = []
    for lo in CONFIG["canny_low"]:
        for ratio in CONFIG["canny_ratio"]:
            d = run(val, ["combined"], lo, lo * ratio)
            sweep.append({"canny_low": lo, "canny_high": lo * ratio, "ratio": ratio, "success": d["success"].mean(),
                          "median_err_px": d["err_px"].replace(np.inf, np.nan).median()})
    sw = pd.DataFrame(sweep).sort_values(["success", "median_err_px"], ascending=[False, True])
    save_table(sw, EXP_ID, "sweep")
    lo, hi = int(sw.iloc[0]["canny_low"]), int(sw.iloc[0]["canny_high"])
    print(sw.to_string())

    df = run(test, ["contour", "hough", "minrect", "combined"], lo, hi)
    save_table(df, EXP_ID)
    fin = df.replace(np.inf, np.nan)
    summ = fin.groupby("method").agg(n=("success", "size"), success=("success", "mean"),
                                     found=("err_px", lambda v: v.notna().mean()),
                                     median_err_px=("err_px", "median"), mean_err_ratio=("err_ratio", "mean"))
    save_table(summ.reset_index(), EXP_ID, "summary")
    print(f"Canny ({lo}, {hi})")
    print(summ.round(4).to_string())

    grid = fin[(fin["method"] == "combined") & fin["recipe"].str.startswith("d3_grid")]
    d3 = grid.groupby(["strength", "angle"]).agg(err_px=("err_px", "median"), success=("success", "mean"),
                                                 n=("success", "size")).reset_index()
    save_table(d3, EXP_ID, "d3_grid")
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for s, g in d3.groupby("strength"):
        ax.plot(g["angle"], g["err_px"], "o-", label=f"phối cảnh {int(s * 100)} %")
    ax.set_xlabel("Góc nghiêng (°)")
    ax.set_ylabel("Sai số 4 góc, trung vị (px)")
    ax.set_title("D3 — Sai số tìm góc theo mức méo (test bộ 5)")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, EXP_ID, "d3_curve")
    print(d3.round(2).to_string())


if __name__ == "__main__":
    main()
