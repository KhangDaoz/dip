"""S3 — Tách khoản thuốc có giúp OCR không? Người làm: Thắng. Kế hoạch: KeHoach.md mục 8.3 (S3), 8.1 (S-B5).

Dữ liệu: 176 ảnh test VAIPE (GT khoản đã soát bằng mắt — results/s0/review_test.csv). Chữ chuẩn = mọi dòng nhãn có tâm
nằm trong vùng thuốc GT của S0 (khung bao box `drugname`, `quantity`, `usage`; gồm cả dòng "Ghi chú…" nằm xen giữa),
theo thứ tự đọc (gom hàng theo tâm y, trái → phải).
Đầu vào OCR: ảnh xám sau ② (khử bóng Closing 81); bbox lấy TỰ ĐỘNG bằng ④ (vùng thuốc S-B1 → dòng S-B2 → khoản S-B3/4)
với tham số cuối (prescription_segmentation.pipeline.PARAMS) — không dùng GT.
Phương án (Tesseract 5, gói vie):
  page    — OCR cả trang (--psm 3), đối chứng chính "không tách vùng".
  region  — OCR một lần vùng thuốc tự động (--psm 6).
  lines   — OCR từng dòng trong vùng thuốc (--psm 7).
  items   — OCR từng khoản thuốc (--psm 6) = pipeline.
  gt_lines — OCR từng box dòng GT (--psm 7): trần trên khi tách vùng hoàn hảo, chỉ còn lỗi của OCR.
Độ đo: CER toàn tập (tổng lỗi / tổng ký tự, phạt cả chữ thừa ngoài phần thuốc và sai thứ tự đọc); precision / recall / F1
trên từ (không phạt thứ tự).
Kết quả: results/s3/{per_image.csv, summary.csv, config.json}

Chạy (từ code/): python experiments/s3_ocr.py   (sau s0_gt_vaipe.py; cần Tesseract 5 + vie)
"""
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from _ocr_common import DRUG_LABELS, crop, reading_order, token_counts, token_prf, xyxy_to_xywh
from cleaning.pipeline import PARAMS as CP, binarize_clean, shadow_free
from common.experiment import results_dir, save_config, save_table
from common.io import load_json, read_image
from common.paths import DATASETS, RESULTS_DIR
from common.seed import set_seed
from ocr import engines
from ocr.metrics import corpus_cer, edit_distance, _norm
from prescription_segmentation.pipeline import PARAMS as SP, segment

EXP_ID = "s3"
CONFIG = {"seed": 42, "split": "test", "lang": "vie", "pad": 3, "workers": 4, "labels": list(DRUG_LABELS),
          "segment_params": {k: v for k, v in SP.items() if k not in ("ocr", "ocr_lang")}}
METHODS = ["page", "region", "lines", "items", "gt_lines"]
LABELS = {"page": "OCR cả trang", "region": "OCR vùng thuốc (1 lần)", "lines": "OCR từng dòng",
          "items": "OCR từng khoản (pipeline)", "gt_lines": "OCR từng dòng GT (trần trên)"}


region_gt = {}


def _init(regions):
    region_gt.update(regions)


def job(img_id):
    d = DATASETS[1] / CONFIG["split"]
    lab = load_json(d / "labels" / f"{img_id}.json")
    rx, ry, rw, rh = region_gt[img_id]
    inside = lambda b: rx <= b[0] + b[2] / 2 <= rx + rw and ry <= b[1] + b[3] / 2 <= ry + rh
    gt_boxes = reading_order([{"bbox": xyxy_to_xywh(e["box"]), "text": e["text"]} for e in lab
                              if inside(xyxy_to_xywh(e["box"]))], key=lambda e: e["bbox"])
    gt = "\n".join(e["text"] for e in gt_boxes)
    gray = shadow_free(read_image(d / "images" / f"{img_id}.png"), CP)
    b, _ = binarize_clean(gray, CP)
    seg = segment(b, CONFIG["segment_params"])
    lang, pad = CONFIG["lang"], CONFIG["pad"]
    pred = {
        "page": engines.ocr_page(gray, lang)[0],
        "region": engines.ocr_block(crop(gray, seg["region"], pad), lang)[0],
        "lines": "\n".join(engines.ocr_line(crop(gray, l, pad), lang)[0] for l in seg["lines"]),
        "items": "\n".join(engines.ocr_block(crop(gray, it, pad), lang)[0] for it in seg["items"]),
        "gt_lines": "\n".join(engines.ocr_line(crop(gray, e["bbox"], pad), lang)[0] for e in gt_boxes),
    }
    out = []
    for m in METHODS:
        tp, npred, ngt = token_counts(pred[m], gt)
        out.append({"image": img_id, "method": m, "gt": gt, "pred": pred[m],
                    "errors": edit_distance(_norm(pred[m]), _norm(gt)), "chars": len(_norm(gt)),
                    "tp": tp, "n_pred": npred, "n_gt": ngt, "n_items": len(seg["items"])})
    return out


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    if not engines.available("tesseract"):
        raise SystemExit("Cần Tesseract 5 + gói vie")
    gt = load_json(RESULTS_DIR / "s0" / f"gt_items_{CONFIG['split']}.json")
    ids = [k for k, g in gt.items() if g.get("review_ok")]
    print(f"{len(ids)} ảnh test đã soát (review_ok)")
    rows = []
    with ProcessPoolExecutor(CONFIG["workers"], initializer=_init, initargs=({k: gt[k]["region"] for k in ids},)) as ex:
        for r in ex.map(job, ids, chunksize=2):
            rows.extend(r)
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID, "per_image")
    summ = []
    for m in METHODS:
        g = df[df["method"] == m]
        summ.append({"method": m, "label": LABELS[m], "images": len(g),
                     "cer": corpus_cer(list(g["pred"]), list(g["gt"])),
                     **token_prf(g["tp"].sum(), g["n_pred"].sum(), g["n_gt"].sum())})
    s = pd.DataFrame(summ)
    save_table(s, EXP_ID, "summary")
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
