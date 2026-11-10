"""T8 — Ablation toàn pipeline: tắt từng bước thì CER cuối thay đổi thế nào? Người chạy: Trường (cả nhóm cung cấp cách tắt).
Kế hoạch: KeHoach.md mục 10, 6.2 (T8).

Dữ liệu: 450 mẫu test bộ 5 (30 ảnh gốc × 15 biến thể; bộ 4 không có chữ chuẩn nên không đo CER).
Chữ chuẩn: mọi dòng nhãn VAIPE (boxes_clean) có tâm nằm trong vùng thuốc (medicine_region_clean), theo thứ tự đọc.
Đầu ra của hệ = chữ OCR của các KHOẢN THUỐC do ④ tìm tự động (Tesseract 5 vie, --psm 6), nối từ trên xuống.
Cấu hình (mục 10):
  full        ① nắn + chỉnh nghiêng + khử lóa → ② tách mực + xóa dấu / bút → khử bóng → Sauvola → ④ tách khoản → OCR ảnh xám sạch
  no_geom     − nắn phối cảnh + chỉnh nghiêng (Hiếu)
  no_glare    − khử lóa (Hiếu; lọc trung vị và CLAHE của ① chỉ phục vụ dò góc / phân loại, không nằm trên đường OCR)
  no_shadow   − khử bóng (Trường)
  no_ink      − tách mực + xóa dấu / bút (Trường)
  ocr_binary  OCR ảnh nhị phân thay ảnh xám (Trường) — pipeline OCR ảnh xám; nhị phân hóa phục vụ tách vùng
  no_seg      − tách vùng: OCR cả trang ảnh xám sạch (--psm 3) (Khang, Thắng)
  raw         không xử lý gì: OCR cả trang ảnh chụp (mốc dưới)
Độ đo: CER toàn tập (tổng lỗi / tổng ký tự; phạt cả chữ thừa và sai thứ tự đọc) + precision / recall / F1 trên từ.
Kết quả: results/t8/{per_sample.csv, summary.csv, by_recipe.csv, config.json}

Chạy (từ code/): python experiments/t8_ablation.py   (cần Tesseract 5 + vie; ≈ 20 phút với 4 tiến trình)
"""
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _ocr_common import crop, reading_order, token_counts, token_prf
from cleaning import ink, inpaint, stamp
from cleaning.pipeline import PARAMS as CP, binarize_clean, shadow_free
from common.experiment import results_dir, save_config, save_table
from common.io import read_image
from common.paths import REPO_DIR, RESULTS_DIR
from common.seed import set_seed
from ocr import engines
from ocr.metrics import _norm, corpus_cer, edit_distance
from preprocessing import enhance, glare as G, quality
from preprocessing.pipeline import PARAMS as PP, preprocess, resize_long_side
from prescription_segmentation.pipeline import PARAMS as SP, segment
from synthetic import dataset as D

EXP_ID = "t8"
BASE = {"geom": True, "glare": True, "enhance": None, "ink": True, "shadow": True, "ocr_on": "gray", "seg": "items"}
CONFIGS = {
    "full": {},
    "no_geom": {"geom": False},
    "no_glare": {"glare": False},
    "no_shadow": {"shadow": False},
    "no_ink": {"ink": False},
    "ocr_binary": {"ocr_on": "binary"},
    "no_seg": {"seg": "page"},
    "raw": {"geom": False, "glare": False, "ink": False, "shadow": False, "seg": "raw"},
}
LABELS = {"full": "Đầy đủ", "no_geom": "− nắn phối cảnh + chỉnh nghiêng", "no_glare": "− khử lóa",
          "no_shadow": "− khử bóng", "no_ink": "− tách mực + xóa dấu / bút",
          "ocr_binary": "OCR ảnh nhị phân (thay ảnh xám)", "no_seg": "− tách vùng (OCR cả trang)",
          "raw": "Không xử lý (OCR ảnh chụp)", "clahe": "+ CLAHE trước ②", "he": "+ cân bằng lược đồ toàn cục trước ②"}
CONFIG = {"seed": 42, "split": "test", "lang": "vie", "pad": 3, "workers": 4, "base": BASE, "configs": CONFIGS,
          "segment_params": {k: v for k, v in SP.items() if k not in ("ocr", "ocr_lang")},
          "clahe": {"clip": PP["clahe_clip"], "tile": PP["clahe_tile"]}}


def gt_text(lab):
    reg = lab["medicine_region_clean"]
    reg = reg["bbox"] if isinstance(reg, dict) else reg
    rx, ry, rw, rh = reg
    inside = lambda b: rx <= b[0] + b[2] / 2 <= rx + rw and ry <= b[1] + b[3] / 2 <= ry + rh
    boxes = reading_order([b for b in lab["boxes_clean"] if b.get("text") and inside(b["bbox"])], key=lambda b: b["bbox"])
    return "\n".join(b["text"] for b in boxes)


def glare_only(img):
    """Khử lóa của ① (H4–H5) mà không nắn: dùng cho cấu hình − nắn."""
    small, _ = resize_long_side(img, PP["long_side"])
    m = quality.glare_mask(small, PP["tau_v"], PP["tau_s"])
    lo, hi = PP["glare_fix_range"]
    if lo <= quality.glare_ratio(m) <= hi:
        return G.remove_glare(img, cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST))
    return img


def apply_enhance(img, kind):
    if kind is None:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    lab[:, :, 0] = enhance.clahe(L, CONFIG["clahe"]["clip"], CONFIG["clahe"]["tile"]) if kind == "clahe" \
        else enhance.equalize_global(L)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def stage1(img, sid, geom, glare, cache, work):
    key = (geom, glare)
    if key not in cache:
        if geom:
            p = {} if glare else {"glare_fix_range": (2.0, 2.0)}
            pre = preprocess(img, f"{sid}_{int(glare)}", work, doc_type="prescription", params=p)
            cache[key] = read_image(REPO_DIR / pre.rectified_path) if not Path(pre.rectified_path).is_absolute() \
                else read_image(pre.rectified_path)
        else:
            cache[key] = glare_only(img) if glare else img
    return cache[key]


def remove_marks(img):
    """② R3–R6 cho đơn thuốc: mask đỏ → dấu (độ tròn) → xóa dấu + nét bút xanh bằng inpainting."""
    m = ink.ink_masks(img)
    st, _ = stamp.split_stamp_vs_text(m["red"])
    marks = np.maximum(st, m["blue"]) if CP["remove_blue"] else st
    return inpaint.remove_marks(img, cv2.dilate(marks, np.ones((3, 3), np.uint8)), CP["inpaint"])


def run_config(img, cfg, sid, cache, work):
    lang, pad = CONFIG["lang"], CONFIG["pad"]
    if cfg["seg"] == "raw":
        return engines.ocr_page(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), lang)[0], 0
    x = stage1(img, sid, cfg["geom"], cfg["glare"], cache, work)
    x = apply_enhance(x, cfg["enhance"])
    if cfg["ink"]:
        x = remove_marks(x)
    gray = shadow_free(x, CP) if cfg["shadow"] else cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    b, _ = binarize_clean(gray, CP)
    src = gray if cfg["ocr_on"] == "gray" else 255 - b
    if cfg["seg"] == "page":
        return engines.ocr_page(src, lang)[0], 0
    seg = segment(b, CONFIG["segment_params"])
    return "\n".join(engines.ocr_block(crop(src, it, pad), lang)[0] for it in seg["items"]), len(seg["items"])


def job(args):
    row, names = args
    lab = D.label(row)
    gt = gt_text(lab)
    img = D.image(row)
    work = Path(tempfile.mkdtemp(prefix="tmp_t8_", dir=RESULTS_DIR))
    cache, out = {}, []
    try:
        for name in names:
            cfg = {**BASE, **ALL_CONFIGS[name]}
            pred, n_items = run_config(img, cfg, row["sample_id"], cache, work)
            tp, npred, ngt = token_counts(pred, gt)
            out.append({"sample_id": row["sample_id"], "recipe": row["recipe_name"], "config": name,
                        "errors": edit_distance(_norm(pred), _norm(gt)), "chars": len(_norm(gt)),
                        "tp": tp, "n_pred": npred, "n_gt": ngt, "n_items": n_items, "gt": gt, "pred": pred})
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


ALL_CONFIGS = {**CONFIGS, "clahe": {"enhance": "clahe"}, "he": {"enhance": "he"}}


def summarize(df, names):
    rows = []
    for n in names:
        g = df[df["config"] == n]
        rows.append({"config": n, "label": LABELS[n], "samples": len(g),
                     "cer": corpus_cer(list(g["pred"].fillna("")), list(g["gt"].fillna(""))),
                     **token_prf(g["tp"].sum(), g["n_pred"].sum(), g["n_gt"].sum())})
    return pd.DataFrame(rows)


def run(exp_id, names, config_extra=None):
    set_seed(CONFIG["seed"])
    results_dir(exp_id)
    save_config({**CONFIG, "run_configs": names, **(config_extra or {})}, exp_id)
    if not engines.available("tesseract"):
        raise SystemExit("Cần Tesseract 5 + gói vie")
    m = D.manifest(CONFIG["split"])
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rows_in = [r for _, r in m.iterrows()][:limit]
    rows = []
    with ProcessPoolExecutor(CONFIG["workers"]) as ex:
        for i, r in enumerate(ex.map(job, [(r, names) for r in rows_in], chunksize=1)):
            rows.extend(r)
            if i % 50 == 0:
                print(f"{i}/{len(rows_in)}", flush=True)
    df = pd.DataFrame(rows)
    save_table(df, exp_id, "per_sample")
    s = summarize(df, names)
    save_table(s, exp_id, "summary")
    by = []
    for (rec, n), g in df.groupby(["recipe", "config"]):
        by.append({"recipe": rec, "config": n, "samples": len(g),
                   "cer": corpus_cer(list(g["pred"].fillna("")), list(g["gt"].fillna("")))})
    save_table(pd.DataFrame(by).pivot(index="recipe", columns="config", values="cer").reset_index(), exp_id, "by_recipe")
    print(s.round(4).to_string())
    return df


if __name__ == "__main__":
    run(EXP_ID, list(CONFIGS))
