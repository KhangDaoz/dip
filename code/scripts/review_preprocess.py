"""Kiểm tra diện rộng preprocess() trên ảnh thật, tự gắn cờ ảnh đáng ngờ, in trang ảnh thu nhỏ để xem bằng mắt.

Cờ:
    nghieng      |góc đo lại trên ảnh kết quả| > SKEW_MAX độ (chỉnh nghiêng chưa đủ)
    chu_doc      tỉ lệ ký tự có láng giềng ngang < VERTICAL_FLAG (trang còn xoay 90°)
    mat_chu      tỉ lệ ký tự ảnh gốc còn nằm trong ảnh kết quả (chiếu qua H) < KEEP_MIN (cắt mất nội dung)
    loa          quality.has_glare
    con_nhieu    tỉ lệ điểm nhiễu xung trên ảnh kết quả > NOISE_RATIO_MIN

Chạy (từ code/):  python scripts/review_preprocess.py --sets 2 --per-set 0
    per-set 0 = lấy mọi file; N = N ảnh gốc ngẫu nhiên (mỗi ảnh gốc 1 bản tăng cường ngẫu nhiên)
Kết quả: results/<--out>/summary.csv, sheet_XX.jpg (trang ảnh thu nhỏ, ảnh có cờ xếp trước), panels/<ảnh>.jpg
"""
import argparse
import random

import cv2
import numpy as np
import pandas as pd

from common.experiment import results_dir
from common.io import read_image, to_gray, write_image
from common.paths import DATASETS, REPO_DIR
from common.seed import set_seed
from preprocessing.corners import char_centroids
from preprocessing.denoise import NOISE_RATIO_MIN, denoise, noise_ratio
from preprocessing.deskew import deskew_projection, horizontal_neighbor_ratio
from preprocessing.pipeline import preprocess
from preprocessing.quality import resize_long_side

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SKEW_MAX, VERTICAL_FLAG, KEEP_MIN = 1.0, 0.45, 0.95
COLORS = {"contour": (0, 200, 0), "hough": (255, 128, 0), "minrect": (0, 165, 255), "full_image": (0, 0, 255)}
CELL_H, CELL_W, COLS, ROWS = 460, 380, 3, 3


def pick(set_id: int, per_set: int, rng: random.Random) -> list:
    files = sorted(p for p in DATASETS[set_id].rglob("*") if p.suffix.lower() in IMAGE_EXTS and "images" in p.parts)
    if per_set == 0:
        return files
    by = {}
    for p in files:
        by.setdefault(p.name.split(".rf.")[0], []).append(p)
    return [rng.choice(by[s]) for s in rng.sample(sorted(by), min(per_set, len(by)))]


def chars_kept(img, H, out_shape) -> float:
    """Tỉ lệ ký tự của ảnh gốc (đã khử nhiễu: cụm nhiễu không phải chữ) nằm trong ảnh kết quả khi chiếu qua H.
    Đo trực tiếp phần nội dung bị cắt, không phụ thuộc tỉ lệ phóng của ảnh kết quả."""
    small, scale = resize_long_side(denoise(img)[0], 1000)
    pts = char_centroids(to_gray(small)) / scale
    if len(pts) == 0:
        return 1.0
    q = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, float).T
    q = q[:, :2] / q[:, 2:3]
    h, w = out_shape[:2]
    return float(((q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)).mean())


def fit(img, h, w=None):
    """Thu vào khung h×w giữ tỉ lệ, phần thừa tô xám — mọi ô cùng cỡ nên trang thu nhỏ gọn."""
    img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    w = w or CELL_W
    s = min(h / img.shape[0], w / img.shape[1])
    small = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    box = np.full((h, w, 3), 200, np.uint8)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    box[y:y + small.shape[0], x:x + small.shape[1]] = small
    return box


def bar(img, text, color=(0, 0, 0)):
    b = np.full((26, img.shape[1], 3), color, np.uint8)
    cv2.putText(b, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([b, img])


def review(path, out):
    img = read_image(path)
    res = preprocess(img, f"_{path.stem[:60]}", out_dir=out / "_rect", doc_type="other")
    rect = read_image(REPO_DIR / res.rectified_path)
    residual = deskew_projection(rect)[1]
    horiz = horizontal_neighbor_ratio(rect)
    keep = chars_kept(img, res.homography, rect.shape)
    noise_in, noise_out = noise_ratio(img), noise_ratio(rect)
    flags = [f for f, on in (("nghieng", abs(residual) > SKEW_MAX), ("chu_doc", horiz < VERTICAL_FLAG),
                             ("mat_chu", keep < KEEP_MIN), ("loa", res.quality.has_glare),
                             ("con_nhieu", noise_out > NOISE_RATIO_MIN)) if on]
    row = {"file": str(path.relative_to(DATASETS[1].parent)), "corner_method": res.corner_method,
           "skew_deg": round(res.skew_deg, 2), "residual_skew": round(residual, 2), "horiz_ratio": round(horiz, 2),
           "char_keep": round(keep, 2), "noise_in_pct": round(100 * noise_in, 3), "noise_out_pct": round(100 * noise_out, 3),
           "glare_pct": round(100 * res.quality.glare_ratio, 2), "blur_var": round(res.quality.blur_var, 1),
           "flags": " ".join(flags)}

    overlay = img.copy()
    c = np.asarray(res.corners_src).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [c], True, COLORS[res.corner_method], max(2, max(img.shape[:2]) // 300))
    left = bar(fit(overlay, CELL_H), f"goc: {res.corner_method}  nhieu {100 * noise_in:.2f}%")
    right = bar(fit(rect, CELL_H), f"ra: nghieng {res.skew_deg:+.1f} con {residual:+.1f}  giu chu {keep:.0%}")
    panel = np.hstack([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right])
    title = (path.name.split(".rf.")[0][:48] + "  " + (" ".join(flags) or "OK"))
    panel = bar(panel, title, (0, 0, 170) if flags else (0, 110, 0))
    return row, panel


def contact_sheets(panels, out):
    w = max(p.shape[1] for p in panels)
    cells = [cv2.copyMakeBorder(p, 4, 4, 4, 4 + w - p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in panels]
    per = COLS * ROWS
    blank = np.full_like(cells[0], 255)
    for k in range(0, len(cells), per):
        chunk = cells[k:k + per] + [blank] * (per - len(cells[k:k + per]))
        sheet = np.vstack([np.hstack(chunk[r * COLS:(r + 1) * COLS]) for r in range(ROWS)])
        write_image(out / f"sheet_{k // per + 1:02d}.jpg", sheet, jpeg_quality=85)


def main():
    ap = argparse.ArgumentParser(description="Kiểm tra diện rộng preprocess() trên ảnh thật")
    ap.add_argument("--sets", type=int, nargs="+", default=[2])
    ap.add_argument("--per-set", type=int, nargs="+", default=[0, 150])
    ap.add_argument("--out", default="review_0211")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    out = results_dir(args.out)
    rows, panels = [], []
    for set_id, n in zip(args.sets, args.per_set):
        for path in pick(set_id, n, rng):
            try:
                row, panel = review(path, out)
            except Exception as e:  # một ảnh lỗi không chặn cả đợt
                row, panel = {"file": str(path), "flags": f"LOI:{type(e).__name__}:{e}"}, None
            row["set"] = set_id
            rows.append(row)
            if panel is not None:
                name = f"bo{set_id}_{path.stem[:70]}.jpg"
                write_image(out / "panels" / name, panel, jpeg_quality=88)
                panels.append((bool(row["flags"]), row["flags"], name, panel))
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    panels.sort(key=lambda t: (not t[0], t[1], t[2]))
    contact_sheets([p for *_, p in panels], out)
    flagged = df["flags"].fillna("").str.len() > 0
    print(f"{len(df)} ảnh, {flagged.sum()} có cờ")
    print(df["flags"].fillna("").str.split().explode().value_counts().to_string())
    print(df.groupby("set").agg(noise_in=("noise_in_pct", "mean"), noise_out=("noise_out_pct", "mean"),
                                residual=("residual_skew", lambda x: x.abs().mean())).round(3).to_string())
    print("Xong:", out)


if __name__ == "__main__":
    main()
