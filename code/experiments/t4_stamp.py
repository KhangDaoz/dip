"""T4 — Phát hiện và xóa dấu đỏ. Người làm: Trường. Kế hoạch: KeHoach.md mục 6.1 (R3–R6), 6.2.

Dữ liệu: bộ 5, mẫu có dấu đỏ (stamp_level > 0), không biến dạng hình học; GT = stamp_mask.
Phát hiện: mask đỏ HSV (C4 · S37–41) → thành phần liên thông, độ tròn C = 4πA/P² (C8 · S16), vành kín, làm đầy vùng
(C7 · S40, S43–45). Độ đo: IoU mask dấu; tỉ lệ dấu phát hiện được (IoU > 0,5).
Xóa: Telea · Navier–Stokes · tô màu nền (C3 · S10) · không xóa → PSNR trong vùng dấu so với ảnh sạch (C6 · S39).
Kết quả: results/t4/

Chạy (từ code/): python experiments/t4_stamp.py
"""
import cv2
import numpy as np
import pandas as pd

from cleaning import ink, inpaint, metrics, stamp
from common.experiment import results_dir, save_config, save_table
from common.seed import set_seed
from synthetic import dataset as D

EXP_ID = "t4"
CONFIG = {"seed": 42, "inpaint": ["telea", "ns", "fill", "none"], "detect_iou": 0.5}


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    m = D.manifest()
    m = m[(m["stamp_level"] > 0) & (~m["geometric"]) & (m["split"].isin(["val", "test"]))]
    rows = []
    for _, r in m.iterrows():
        img, clean, gt = D.image(r), D.image(r, "clean"), D.mask(r, "stamp_mask")
        if gt is None or img.shape != clean.shape:
            continue
        masks = ink.ink_masks(img)
        pred, red_text = stamp.split_stamp_vs_text(masks["red"])
        # GT stamp_mask chỉ gồm NÉT mực của dấu; kết quả R5 là vùng dấu đã làm đầy → chấm nét đỏ nằm trong vùng dấu
        stroke = np.where(pred > 0, masks["red"], 0).astype(np.uint8)
        rec = {"sample_id": r["sample_id"], "split": r["split"], "stamp_level": r["stamp_level"],
               "iou_red_hsv": metrics.mask_iou(masks["red"], gt), "iou_detect": metrics.mask_iou(stroke, gt),
               "gt_covered": float(((pred > 0) & (gt > 0)).sum() / max((gt > 0).sum(), 1)),
               "red_text_kept_px": int((red_text > 0).sum())}
        region = cv2.dilate(gt, np.ones((5, 5), np.uint8)) > 0
        for meth in CONFIG["inpaint"]:
            out = img if meth == "none" else inpaint.remove_marks(img, cv2.dilate(pred, np.ones((3, 3), np.uint8)), meth)
            g1, g0 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
            rec[f"psnr_stamp_{meth}"] = metrics.psnr(g1[region], g0[region])
        rows.append(rec)
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID)
    df["detected"] = df["gt_covered"] > 0.8
    summ = df.groupby("stamp_level")[["iou_red_hsv", "iou_detect", "gt_covered", "detected"] +
                                     [f"psnr_stamp_{m}" for m in CONFIG["inpaint"]]].mean()
    summ.loc["tất cả"] = df[summ.columns].mean()
    save_table(summ.reset_index(), EXP_ID, "summary")
    print(f"n = {len(df)}")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
