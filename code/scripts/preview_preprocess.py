"""Xem trước tiền xử lý trên ảnh thật: ảnh gốc + 4 góc tìm được | ảnh gửi bước ② | sau CLAHE.

Lấy ngẫu nhiên (seed cố định) mỗi bộ N ảnh gốc khác nhau (bỏ bản tăng cường .rf. trùng ảnh gốc).

Chạy (từ thư mục code/):  python scripts/preview_preprocess.py --per-set 5 --sets 2 3
Kết quả: results/<--out>/<bộ>_<ảnh>.jpg + summary.csv
"""
import argparse
import random

import cv2
import numpy as np
import pandas as pd

from common.experiment import results_dir
from common.io import read_image, write_image
from common.paths import DATASETS, REPO_DIR
from common.seed import set_seed
from preprocessing.enhance import apply_clahe
from preprocessing.pipeline import preprocess

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PANEL_H = 600
COLORS = {"contour": (0, 200, 0), "hough": (255, 128, 0), "minrect": (0, 165, 255), "full_image": (0, 0, 255)}


def pick_images(set_id: int, n: int, rng: random.Random) -> list:
    files = sorted(p for p in DATASETS[set_id].rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    by_source = {}
    for p in files:
        by_source.setdefault(p.name.split(".rf.")[0], p)   # một ảnh / ảnh gốc
    return rng.sample(sorted(by_source.values()), min(n, len(by_source)))


def fit_height(img: np.ndarray, h: int = PANEL_H) -> np.ndarray:
    img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (max(1, int(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((28, img.shape[1], 3), np.uint8)      # dải tiêu đề nằm trên ảnh, không che nội dung
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def process(path, out) -> tuple:
    """Chạy preprocess() đầy đủ: khung trái = ảnh gốc + 4 góc, giữa = ảnh gửi ② (rectified_path), phải = CLAHE."""
    img = read_image(path)
    res = preprocess(img, f"_{path.stem.split('.rf.')[0]}", out_dir=out / "_rect", doc_type="other")
    rect = read_image(REPO_DIR / res.rectified_path)
    q, method, corners = res.quality, res.corner_method, np.asarray(res.corners_src)

    overlay = img.copy()
    thick = max(2, max(img.shape[:2]) // 300)
    cv2.polylines(overlay, [corners.astype(np.int32).reshape(-1, 1, 2)], True, COLORS[method], thick)
    for (x, y), name in zip(corners.astype(int), ("TL", "TR", "BR", "BL")):
        cv2.circle(overlay, (x, y), thick * 3, COLORS[method], -1)
    panel = np.hstack([
        label(fit_height(overlay), f"goc: {method} | blur={q.blur_var:.0f} glare={q.glare_ratio:.1%}"),
        label(fit_height(rect), f"gui buoc 2: {rect.shape[1]}x{rect.shape[0]} | nghieng={res.skew_deg:+.2f}"),
        label(fit_height(apply_clahe(rect)), "CLAHE (chi de phan loai)"),
    ])
    row = {"file": str(path.relative_to(DATASETS[1].parent)), "corner_method": method,
           "blur_var": round(q.blur_var, 1), "is_blurry": q.is_blurry,
           "glare_ratio": round(q.glare_ratio, 4), "has_glare": q.has_glare,
           "skew_deg": round(res.skew_deg, 2), "rect_w": rect.shape[1], "rect_h": rect.shape[0]}
    return panel, row


def main():
    parser = argparse.ArgumentParser(description="Xem trước tiền xử lý trên ảnh thật")
    parser.add_argument("--sets", type=int, nargs="+", default=[2])
    parser.add_argument("--per-set", type=int, default=5)
    parser.add_argument("--out", default="preview_0111")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    out = results_dir(args.out)
    rows = []
    for set_id in args.sets:
        for path in pick_images(set_id, args.per_set, rng):
            panel, row = process(path, out)
            name = f"bo{set_id}_{path.name.split('.rf.')[0]}.jpg"
            write_image(out / name, panel, jpeg_quality=90)
            rows.append({"preview": name, **row})
            print(f"{name:<45} {row['corner_method']:<10} blur={row['blur_var']:<8} glare={row['glare_ratio']}")
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    print("Xong:", out)


if __name__ == "__main__":
    main()
