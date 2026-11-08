"""D5 — Khử lóa: không khử vs Telea vs Navier–Stokes (inpainting), với mask lóa thật và mask phát hiện được
Người làm: Hiếu. Kế hoạch: team/KeHoach_Hieu.md (mục 3.2).

Dữ liệu: bộ 6 v2.1 tập test, ảnh chỉ lóa (v10 single_glare, mức 1–3) — cô lập tác động của lóa.
Ảnh tham chiếu: clean_exposed/ (ảnh sạch cùng mức phơi sáng với ảnh hỏng).
Mask: "gt" = glare_mask thật (cận trên, như có bộ phát hiện hoàn hảo); "detected" = quality.glare_mask (thực tế).
Độ đo trong vùng lóa thật: PSNR, SSIM (ảnh xám), F-measure nét chữ so với Otsu của clean_exposed, nhị phân hóa ảnh
kết quả bằng Otsu toàn cục và bằng Sauvola (thích nghi cục bộ — cách bước ② dùng).
CER sau OCR: thêm khi chạy D6 (bộ OCR Tesseract của Khang).
Đầu ra: results/d5/table.csv, summary.csv, figure.png (ví dụ trước / sau)

Chạy: python experiments/d5_glare_inpaint.py
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.filters import threshold_sauvola
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import bo6
from common.experiment import save_config, save_figure, save_table
from common.io import read_image, to_gray
from common.seed import set_seed
from preprocessing.glare import remove_glare
from preprocessing.quality import glare_mask

EXP_ID = "d5"
CONFIG = {"seed": 42, "split": "test", "recipe": "single_glare", "methods": ["none", "telea", "ns"], "radius": 5}
LABELS = {"none": "Không khử", "telea": "Telea", "ns": "Navier–Stokes"}


def load_data(config: dict) -> pd.DataFrame:
    m = pd.read_csv(bo6.ROOT / "manifest_v2.csv")
    return m[(m.split == config["split"]) & (m.recipe_name == config["recipe"])].reset_index(drop=True)


def ink(gray: np.ndarray) -> np.ndarray:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0


def ink_sauvola(gray: np.ndarray) -> np.ndarray:
    return gray < threshold_sauvola(gray, window_size=25, k=0.2)


def f_in(pred: np.ndarray, gt: np.ndarray, region: np.ndarray) -> float:
    p, g = pred & region, gt & region
    tp = (p & g).sum()
    if g.sum() == 0:
        return np.nan
    return 0.0 if tp == 0 else float(2 * tp / (p.sum() + g.sum()))


def run(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = []
    for _, r in data.iterrows():
        img = read_image(bo6.ROOT / r.rel_degraded_path)
        ref = read_image(bo6.ROOT / r.rel_clean_exposed_path)
        gt_mask = read_image(bo6.ROOT / r.rel_glare_mask_path, gray=True)
        region = gt_mask > 0
        g_ref = to_gray(ref)
        ink_ref = ink(g_ref)
        masks = {"gt": gt_mask, "detected": glare_mask(img)}
        for mask_name, mask in masks.items():
            for m in config["methods"]:
                res = img if m == "none" else remove_glare(img, mask, method=m, radius=config["radius"])
                g = to_gray(res)
                ys, xs = np.nonzero(region)
                box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
                out.append({"sample_id": r.sample_id, "glare_level": int(r.glare_level), "mask": mask_name, "method": m,
                            "psnr_in": peak_signal_noise_ratio(g_ref[region], g[region], data_range=255),
                            "ssim_box": structural_similarity(g_ref[box], g[box], data_range=255),
                            "f_ink_in": f_in(ink(g), ink_ref, region),
                            "f_ink_sauvola_in": f_in(ink_sauvola(g), ink_ref, region)})
    return pd.DataFrame(out)


def examples(data: pd.DataFrame, config: dict):
    r = data[data.glare_level == 3].iloc[0]
    img = read_image(bo6.ROOT / r.rel_degraded_path)
    ref = read_image(bo6.ROOT / r.rel_clean_exposed_path)
    gm = read_image(bo6.ROOT / r.rel_glare_mask_path, gray=True)
    ys, xs = np.nonzero(gm)
    pad = 20
    crop = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
    panels = [("Ảnh sạch (cùng phơi sáng)", ref), ("Lóa mức 3", img),
              ("Telea (mask thật)", remove_glare(img, gm, "telea")), ("Navier–Stokes (mask thật)", remove_glare(img, gm, "ns"))]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (t, im) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(im[crop], cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig


def main():
    set_seed(CONFIG["seed"])
    save_config(CONFIG, EXP_ID)
    data = load_data(CONFIG)
    df = run(data, CONFIG)
    print("Bảng:", save_table(df, EXP_ID))
    summary = df.groupby(["mask", "method", "glare_level"])[["psnr_in", "ssim_box", "f_ink_in", "f_ink_sauvola_in"]].mean().round(3).reset_index()
    save_table(summary, EXP_ID, "summary")
    print(df.groupby(["mask", "method"])[["psnr_in", "ssim_box", "f_ink_in", "f_ink_sauvola_in"]].mean().round(3).to_string())
    print(summary.pivot_table(index=["mask", "method"], columns="glare_level", values="f_ink_sauvola_in").round(3).to_string())
    print("Hình:", save_figure(examples(data, CONFIG), EXP_ID))


if __name__ == "__main__":
    main()
