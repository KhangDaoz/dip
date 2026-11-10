"""T2-CER — Sauvola hay Otsu cho CER thấp hơn trên ẢNH THẬT? Người làm: Trường. Kế hoạch: KeHoach.md mục 6.2 (T2-CER), 6.3, 6.4.

TIÊU CHÍ CHỐT TRƯỚC KHI CHẠY (ghi vào config.json, không đổi sau khi xem kết quả):
  Phương pháp nhị phân chính = phương pháp có CER toàn tập thấp nhất trên crop dòng VAIPE tập CHỌN THAM SỐ (valid).
  Nếu Sauvola kém phương pháp tốt nhất < 0,005 CER thì giữ Sauvola (bền với bóng — T3). Tập test VAIPE và bộ 2 chỉ để báo cáo.

Dữ liệu (ảnh thật):
  * VAIPE valid (chọn) / test (báo cáo): ảnh đơn thuốc scan; mỗi dòng có box + chữ chuẩn trong nhãn. Lấy ngẫu nhiên
    (seed 42) `n_lines` dòng có ≥ 3 ký tự mỗi tập.
  * Bộ 2 (phiếu XN chụp thật, tiếng Anh): khung cụm từ + chữ chuẩn đã soát của K0 (32 ảnh) — thay cho `dataset2_50`
    không có trên Drive nhóm. Chữ bộ 2 chỉ cao 4–11 px nên mọi crop phóng ×3 (nội suy bậc ba) như K5.
Quy trình: nhị phân CẢ TRANG (sau khử bóng Closing 81, hoặc không) bằng từng phương pháp → cắt theo box → OCR dòng
(Tesseract 5, --psm 7). Đối chiếu: OCR ảnh xám đã khử bóng (không nhị phân).
Độ đo: CER toàn tập (tổng lỗi / tổng ký tự), CER trung vị theo dòng.
Kết quả: results/t2cer/{lines.csv, summary.csv, config.json}

Chạy (từ code/): python experiments/t2cer_binarize_ocr.py
"""
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from _ocr_common import crop, ink_to_page
from cleaning import background, binarize, postprocess
from cleaning.pipeline import PARAMS as CP, estimate_char_height, min_area_for
from common.experiment import results_dir, save_config, save_table
from common.io import load_json, read_image
from common.paths import DATASETS, RESULTS_DIR
from common.seed import set_seed
from ocr import engines
from ocr.metrics import cer, corpus_cer

EXP_ID = "t2cer"
CONFIG = {
    "seed": 42, "n_lines": 600, "min_chars": 3, "pad": 3, "scale_bo2": 3, "workers": 4,
    "criterion": "CER toàn tập thấp nhất trên crop dòng VAIPE valid; nếu Sauvola kém phương pháp tốt nhất < 0,005 thì "
                 "giữ Sauvola (bền với bóng, T3). Chốt trước khi chạy.",
    "sauvola": {"window_factor": CP["sauvola_window_factor"], "k": CP["sauvola_k"]},
    "variants": [("gray", True)] + [(m, True) for m in binarize.METHODS] + [("sauvola", False), ("otsu", False)],
}


def page_binary(gray, method, shadow_removal):
    g = background.remove_shadow(gray, background.background_closing(gray, CP["background_kernel"])) \
        if shadow_removal else gray
    if method == "gray":
        return g, None
    h = estimate_char_height(g)
    if method in ("sauvola", "niblack", "wolf"):
        w = int(max(15, round(CONFIG["sauvola"]["window_factor"] * h))) | 1
        kw = {"window": w}
        if method == "sauvola":
            kw["k"] = CONFIG["sauvola"]["k"]
        b = binarize.binarize(g, method, **kw)
    else:
        b = binarize.binarize(g, method)
    return g, postprocess.remove_small(b, min_area_for(h, CP["min_area_factor"]))


def ocr_crop(c, lang, scale):
    if c.size == 0:
        return ""
    if scale > 1:
        c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return engines.ocr_line(c, lang)[0]


def job(args):
    """Một ảnh: mọi biến thể × mọi dòng đã chọn của ảnh đó."""
    path, lines, lang, scale, dataset, split = args
    gray = read_image(path, gray=True)
    out = []
    for method, sr in CONFIG["variants"]:
        g, b = page_binary(gray, method, sr)
        src = g if b is None else ink_to_page(b)
        for ln in lines:
            pred = ocr_crop(crop(src, ln["bbox"], CONFIG["pad"]), lang, scale)
            out.append({"dataset": dataset, "split": split, "image": path.stem, "line": ln["id"],
                        "method": method, "shadow_removal": sr, "gt": ln["text"], "pred": pred,
                        "cer": cer(pred, ln["text"])})
    return out


def vaipe_jobs(split, rng):
    d = DATASETS[1] / split
    cand = []
    for lp in sorted((d / "labels").glob("*.json")):
        for i, e in enumerate(load_json(lp)):
            if len((e.get("text") or "").strip()) >= CONFIG["min_chars"]:
                x0, y0, x1, y1 = e["box"]
                cand.append((lp.stem, {"id": i, "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                                       "text": e["text"]}))
    pick = [cand[i] for i in sorted(rng.choice(len(cand), min(CONFIG["n_lines"], len(cand)), replace=False))]
    by_img = {}
    for img_id, ln in pick:
        by_img.setdefault(img_id, []).append(ln)
    return [(d / "images" / f"{k}.png", v, "vie", 1, "vaipe", split) for k, v in by_img.items()]


def bo2_jobs(rng):
    gt = load_json(RESULTS_DIR / "k0" / "gt.json")
    cand = []
    for key, g in gt.items():
        for i, w in enumerate(g["words"]):
            if w.get("text") and len(w["text"].strip()) >= CONFIG["min_chars"] and not w.get("text_uncertain"):
                cand.append((key, {"id": i, "bbox": w["bbox"], "text": w["text"]}))
    pick = [cand[i] for i in sorted(rng.choice(len(cand), min(CONFIG["n_lines"], len(cand)), replace=False))]
    by_img = {}
    for key, ln in pick:
        by_img.setdefault(key, []).append(ln)
    return [(DATASETS[2] / gt[k]["image"], v, "eng", CONFIG["scale_bo2"], "bo2", "eval") for k, v in by_img.items()]


def main():
    rng = set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    if not engines.available("tesseract"):
        raise SystemExit("Cần Tesseract 5 + gói vie, eng")
    jobs = vaipe_jobs("valid", rng) + vaipe_jobs("test", rng) + bo2_jobs(rng)
    rows = []
    with ProcessPoolExecutor(CONFIG["workers"]) as ex:
        for r in ex.map(job, jobs, chunksize=2):
            rows.extend(r)
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID, "lines")

    summ = []
    for (ds, sp, m, sr), g in df.groupby(["dataset", "split", "method", "shadow_removal"]):
        summ.append({"dataset": ds, "split": sp, "method": m, "shadow_removal": sr, "n_lines": len(g),
                     "chars": int(sum(len(t) for t in g["gt"])), "cer": corpus_cer(list(g["pred"]), list(g["gt"])),
                     "cer_median": float(np.median(g["cer"]))})
    s = pd.DataFrame(summ).sort_values(["dataset", "split", "cer"])
    save_table(s, EXP_ID, "summary")
    print(s.round(4).to_string())

    v = s[(s["dataset"] == "vaipe") & (s["split"] == "valid") & (s["method"] != "gray")]
    best = v.iloc[0]
    sau = v[(v["method"] == "sauvola") & (v["shadow_removal"])].iloc[0]
    chosen = "sauvola" if sau["cer"] - best["cer"] < 0.005 else best["method"]
    print(f"\nTiêu chí (valid VAIPE): tốt nhất = {best['method']} (CER {best['cer']:.4f}); Sauvola {sau['cer']:.4f} "
          f"→ chọn: {chosen}")


if __name__ == "__main__":
    main()
