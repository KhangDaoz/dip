"""D11 — Khử nhiễu muối tiêu: không khử vs trung vị 3×3 / 5×5 vs trung vị chọn lọc vs trung vị thích nghi
vs Gauss / bilateral (đối chứng). Chương 3, bài giảng slide 56, 73–78.
Người làm: Hiếu. Kế hoạch: team/KeHoach_Hieu.md (mục 3.2).

Dữ liệu: ảnh sạch bộ 6 (30 ảnh gốc của tập test), thu nhỏ cạnh dài về 640 px như ảnh Roboflow (chữ nhỏ —
trường hợp lọc trung vị dễ làm mòn nét nhất), thêm muối tiêu mật độ d rồi nén JPEG q=90 như Roboflow.
d = 0 để kiểm tra phương án có làm hỏng ảnh sạch không.
Độ đo: PSNR, SSIM (ảnh xám) so với ảnh sạch; F-measure nét chữ: Otsu trên ảnh đã khử so với Otsu trên ảnh
sạch (mực = 255); thời gian ms/ảnh. Bộ phát hiện nhiễu: tỉ lệ điểm nhiễu đo được theo d.
Đầu ra: results/d11/table.csv, results/d11/summary.csv, results/d11/figure.png, results/d11/examples.png

Chạy: python experiments/d11_denoise.py
"""
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import bo6
from common.experiment import save_config, save_figure, save_table
from common.io import to_gray
from common.seed import set_seed
from preprocessing import denoise
from preprocessing.quality import resize_long_side

EXP_ID = "d11"
CONFIG = {"seed": 42, "split": "test", "long_side": 640, "densities": [0, 0.005, 0.01, 0.03, 0.05, 0.10],
          "jpeg_quality": 90}
METHODS = ["none", "median3", "median5", "switching", "adaptive", "gaussian", "bilateral"]
LABELS = {"none": "Không khử", "median3": "Trung vị 3×3", "median5": "Trung vị 5×5",
          "switching": "Trung vị chọn lọc", "adaptive": "Trung vị thích nghi", "gaussian": "Gauss 3×3",
          "bilateral": "Bilateral"}


def add_salt_pepper(img: np.ndarray, d: float, rng, jpeg_quality: int) -> np.ndarray:
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < d / 2] = 0
    out[(r >= d / 2) & (r < d)] = 255
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def ink(gray: np.ndarray) -> np.ndarray:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0


def f_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = np.logical_and(pred, gt).sum()
    if tp == 0:
        return 0.0
    p, r = tp / pred.sum(), tp / gt.sum()
    return float(2 * p * r / (p + r))


def load_data(config: dict):
    m = bo6.manifest((config["split"],)).drop_duplicates("source_id")
    return [(r.source_id.split(":")[-1], resize_long_side(bo6.clean(r), config["long_side"])[0]) for _, r in m.iterrows()]


def run(data, config: dict) -> pd.DataFrame:
    rng = np.random.default_rng(config["seed"])
    out = []
    for name, clean in data:
        g_clean = to_gray(clean)
        ink_clean = ink(g_clean)
        for d in config["densities"]:
            noisy = add_salt_pepper(clean, d, rng, config["jpeg_quality"]) if d > 0 else clean
            detected = denoise.noise_ratio(noisy)
            for m in METHODS:
                t0 = time.perf_counter()
                res = noisy if m == "none" else denoise.METHODS[m](noisy)
                ms = (time.perf_counter() - t0) * 1000
                g = to_gray(res)
                out.append({"image": name, "density": d, "detected_ratio": detected, "method": m,
                            "psnr": peak_signal_noise_ratio(g_clean, g, data_range=255),
                            "ssim": structural_similarity(g_clean, g, data_range=255),
                            "f_ink": f_measure(ink(g), ink_clean), "ms": ms})
    return pd.DataFrame(out).replace([np.inf], 99.0)


def plot(df: pd.DataFrame):
    agg = df.groupby(["density", "method"]).f_ink.mean().unstack()[METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for m in METHODS:
        axes[0].plot(100 * agg.index, agg[m], marker="o", label=LABELS[m])
    axes[0].set_xlabel("Mật độ muối tiêu (%)")
    axes[0].set_ylabel("F-measure nét chữ")
    axes[0].grid(alpha=0.3)
    psnr = df.groupby(["density", "method"]).psnr.mean().unstack()[METHODS]
    for m in METHODS:
        axes[1].plot(100 * psnr.index, psnr[m], marker="o", label=LABELS[m])
    axes[1].set_xlabel("Mật độ muối tiêu (%)")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def examples(data, config: dict):
    """Một vùng chữ phóng to: sạch | nhiễu 3% | trung vị 3×3 | trung vị 5×5 | thích nghi."""
    rng = np.random.default_rng(config["seed"] + 1)
    _, clean = data[0]
    h, w = clean.shape[:2]
    crop = (slice(int(0.30 * h), int(0.30 * h) + 90), slice(int(0.08 * w), int(0.08 * w) + 220))
    noisy = add_salt_pepper(clean, 0.03, rng, config["jpeg_quality"])
    panels = [("Sạch", clean), ("Nhiễu 3%", noisy), ("Trung vị 3×3", denoise.median(noisy, 3)),
              ("Trung vị 5×5", denoise.median(noisy, 5)), ("Trung vị thích nghi", denoise.adaptive_median(noisy))]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 2.2))
    for ax, (t, im) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(im[crop], cv2.COLOR_BGR2RGB), interpolation="nearest")
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
    summary = df.groupby(["density", "method"]).agg(psnr=("psnr", "mean"), ssim=("ssim", "mean"), f_ink=("f_ink", "mean"),
                                                    ms=("ms", "mean"), detected_pct=("detected_ratio", lambda x: 100 * x.mean()))
    summary = summary.round(4).reset_index()
    save_table(summary, EXP_ID, "summary")
    print(summary.pivot(index="method", columns="density", values="f_ink").reindex(METHODS).round(3).to_string())
    print(summary.pivot(index="method", columns="density", values="psnr").reindex(METHODS).round(2).to_string())
    print(summary.groupby("density").detected_pct.first().round(3).to_string())
    print(summary[summary.density == 0.03].set_index("method").ms.reindex(METHODS).round(1).to_string())
    print("Hình:", save_figure(plot(df), EXP_ID), save_figure(examples(data, CONFIG), EXP_ID, "examples"))


if __name__ == "__main__":
    main()
