"""K5 — Dựng bảng (OCR theo ô) có giúp OCR so với OCR cả trang? Người làm: Khang. Kế hoạch: team/KeHoach.md mục 7.2.

Dữ liệu: GT chính thức K0 (results/k0/gt.json, 32 ảnh bộ 2). Text GT: LLM phiên âm từ ảnh gốc, người soát
(experiments/k0_text_gt.json); hộp text = null bị bỏ khỏi chuỗi đáp án.
Đầu vào OCR: ảnh xám sau ② (khử bóng, kéo giãn tương phản), cắt về VÙNG THÂN BẢNG (hình bao các khung GT + lề h) —
chung cho cả 3 phương án để không phạt OCR cả trang vì phần đầu phiếu. Mọi vùng cắt phóng ×3 (nội suy bậc ba, C2)
trước Tesseract 5 (eng) vì chữ bộ 2 chỉ cao 4–11 px (trung vị 5 px).
Phương án:
  page    — OCR cả vùng bảng một lần (Tesseract --psm 3), đối chứng.
  cells   — pipeline ③ đầy đủ, tự động: cụm từ (Dilation) → dòng (projection + xóa kẻ) → cột → ô; OCR từng ô (--psm 7).
  gt_box  — OCR từng khung cụm từ GT (--psm 7): trần trên khi tách ô hoàn hảo, chỉ còn lỗi của OCR.
Độ đo: F1 trên từ (tập từ có lặp, không phân biệt hoa thường — đo nhận dạng, KHÔNG phạt thứ tự đọc); CER toàn tập (tổng lỗi / tổng ký tự, ocr.metrics.corpus_cer) trên chuỗi thân bảng theo thứ tự đọc (dòng trên →
dưới, ô trái → phải); với cells và gt_box thêm: tỉ lệ dòng chỉ số đủ trường (mỗi trường GT có CER ≤ 0,2 với trường dự
đoán cùng vai trò ở dòng khớp) và tỉ lệ KẾT QUẢ số đọc đúng (parse_number bằng nhau) — thứ OCR cả trang không cho được
vì không biết số nào thuộc cột nào.
Kết quả: results/k5/ (table.csv, per_image.csv, rows.csv, config.json).

Chạy (từ code/): python experiments/k5_ocr.py   (sau k0_gt.py; cần Tesseract 5 + pytesseract)
"""
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from cleaning.pipeline import PARAMS as CLEAN_PARAMS, binarize_clean, shadow_free
from common.experiment import results_dir, save_config, save_table
from common.io import load_json, read_image
from common.paths import DATASETS, RESULTS_DIR
from common.seed import set_seed
from ocr import engines
from ocr.metrics import _norm, cer, corpus_cer, edit_distance
from table_extraction import flags
from table_extraction.pipeline import structure

EXP_ID = "k5"
CONFIG = {"seed": 42, "lang": "eng", "scale": 3, "pad": 2, "field_cer_max": 0.2,
          # tham số cấu trúc = tham số cuối của K1/K2 (chọn giống nhau ở cả 5 fold, results/k2/folds.csv)
          "structure": {"opening": 0, "word_a": 2.0, "word_v": 0.5, "row_se": 2.0, "smooth_window": 1,
                        "valley_ratio": 0.1, "col_gap": 0.5, "col_cross": 0.6}}
ROLES = ("parameter", "result", "unit", "range")


def up(img):
    s = CONFIG["scale"]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def crop(gray, box, pad):
    x, y, w, h = box
    H, W = gray.shape
    return gray[max(y - pad, 0):min(y + h + pad, H), max(x - pad, 0):min(x + w + pad, W)]


def ocr_box(gray, box):
    c = crop(gray, box, CONFIG["pad"])
    return engines.ocr_line(up(c), CONFIG["lang"])[0] if c.size else ""


def tokens(s):
    """Tập từ có lặp (không phân biệt hoa thường) — để đo nhận dạng mà KHÔNG phạt sai thứ tự đọc."""
    from collections import Counter
    return Counter(_norm(s).lower().split())


def gt_rows(words):
    rows = {}
    for w in words:
        rows.setdefault(w["row"], []).append(w)
    return [sorted(v, key=lambda w: w["bbox"][0]) for _, v in sorted(rows.items())]


def row_fields(ws, text_of):
    """{role: chuỗi} của một dòng; text_of(w) trả về chuỗi của cụm từ / ô."""
    out = {}
    for w in ws:
        t = text_of(w)
        if t:
            out[w["role"]] = (out.get(w["role"], "") + " " + t).strip()
    return out


def field_ok(pred: dict, gt: dict) -> bool:
    return all(cer(pred.get(r, ""), t) <= CONFIG["field_cer_max"] for r, t in gt.items())


def result_ok(pred: dict, gt: dict):
    if "result" not in gt or flags.parse_number(gt["result"]) is None:
        return None
    return flags.parse_number(pred.get("result", "")) == flags.parse_number(gt["result"])


def process(item):
    key, g = item
    img = read_image(DATASETS[2] / g["image"])
    gray = shadow_free(img, CLEAN_PARAMS)
    binary, _ = binarize_clean(gray, CLEAN_PARAMS)
    words = [w for w in g["words"] if w["text"] is not None]
    # vùng thân bảng = hình bao các khung GT + lề
    xs = [w["bbox"][0] for w in g["words"]]
    ys = [w["bbox"][1] for w in g["words"]]
    xe = [w["bbox"][0] + w["bbox"][2] for w in g["words"]]
    ye = [w["bbox"][1] + w["bbox"][3] for w in g["words"]]
    m = 6
    X0, Y0 = max(min(xs) - m, 0), max(min(ys) - m, 0)
    X1, Y1 = min(max(xe) + m, gray.shape[1]), min(max(ye) + m, gray.shape[0])
    g_crop, b_crop = gray[Y0:Y1, X0:X1], binary[Y0:Y1, X0:X1]

    rows_gt = gt_rows(words)
    gt_text = "\n".join(" ".join(w["text"] for w in r) for r in rows_gt)

    # page: OCR cả vùng bảng
    page_text = engines.ocr_page(up(g_crop), CONFIG["lang"])[0]

    # gt_box: OCR từng khung GT (tọa độ trong vùng cắt)
    def local(b):
        return [b[0] - X0, b[1] - Y0, b[2], b[3]]
    box_txt = {w["lid"]: ocr_box(g_crop, local(w["bbox"])) for w in words}
    gtbox_text = "\n".join(" ".join(box_txt[w["lid"]] for w in r) for r in rows_gt)

    # cells: pipeline ③ tự động trên vùng bảng
    s = structure(b_crop, CONFIG["structure"])
    cell_txt = {(c["row"], c["col"]): ocr_box(g_crop, c["bbox"]) for c in s["cells"]}
    by_row = {}
    for c in s["cells"]:
        by_row.setdefault(c["row"], []).append(c)
    cells_text = "\n".join(" ".join(cell_txt[(c["row"], c["col"])] for c in sorted(v, key=lambda c: c["col"]))
                           for _, v in sorted(by_row.items()))
    roles = flags.column_roles(s["cells"], cell_txt) if s["cells"] else {}

    # khớp dòng GT ↔ dòng dự đoán theo chồng lấn dọc lớn nhất
    pred_rows = []
    for r, v in sorted(by_row.items()):
        y0 = min(c["bbox"][1] for c in v) + Y0
        y1 = max(c["bbox"][1] + c["bbox"][3] for c in v) + Y0
        f = {}
        for c in sorted(v, key=lambda c: c["col"]):
            t = cell_txt[(c["row"], c["col"])]
            role = roles.get(c["col"], "unit")
            if t:
                f[role] = (f.get(role, "") + " " + t).strip()
        pred_rows.append((y0, y1, f))
    rec_rows = []
    for r in rows_gt:
        gt_f = row_fields(r, lambda w: w["text"])
        if "parameter" not in gt_f or "result" not in gt_f:
            continue
        ry0 = min(w["bbox"][1] for w in r)
        ry1 = max(w["bbox"][1] + w["bbox"][3] for w in r)
        ov = [max(0, min(ry1, b) - max(ry0, a)) for a, b, _ in pred_rows]
        pf = pred_rows[int(np.argmax(ov))][2] if ov and max(ov) > 0 else {}
        bf = row_fields(r, lambda w: box_txt[w["lid"]])
        rec_rows.append({"image": key, "n_fields": len(gt_f),
                         "cells_ok": field_ok(pf, gt_f), "gt_box_ok": field_ok(bf, gt_f),
                         "cells_result": result_ok(pf, gt_f), "gt_box_result": result_ok(bf, gt_f)})

    per = {"image": key, "n_chars": len(_norm(gt_text)), "n_rows_gt": len(rows_gt), "n_rows_pred": len(by_row),
           "gt": gt_text, "page": page_text, "cells": cells_text, "gt_box": gtbox_text}
    tg = tokens(gt_text)
    per["tok_gt"] = sum(tg.values())
    for m_ in ("page", "cells", "gt_box"):
        per[f"err_{m_}"] = edit_distance(_norm(per[m_]), _norm(gt_text))
        tp_ = tokens(per[m_])
        per[f"tok_tp_{m_}"] = sum((tp_ & tg).values())
        per[f"tok_pred_{m_}"] = sum(tp_.values())
    return per, rec_rows


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    if not engines.available():
        raise SystemExit("K5 cần Tesseract 5 + pytesseract (requirements-ocr.txt)")
    gt = load_json(RESULTS_DIR / "k0" / "gt.json")
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(process, gt.items()))
    per = pd.DataFrame([r[0] for r in res])
    rows = pd.DataFrame([x for r in res for x in r[1]])
    save_table(per, EXP_ID, "per_image")
    save_table(rows, EXP_ID, "rows")
    summ = []
    for m in ("page", "cells", "gt_box"):
        tp, npred, ngt = per[f"tok_tp_{m}"].sum(), per[f"tok_pred_{m}"].sum(), per["tok_gt"].sum()
        P, R = tp / npred if npred else 0, tp / ngt
        d = {"method": m, "cer": corpus_cer(per[m].tolist(), per["gt"].tolist()),
             "cer_median_image": float((per[f"err_{m}"] / per["n_chars"]).median()),
             "token_precision": P, "token_recall": R, "token_f1": 2 * P * R / (P + R) if P + R else 0}
        if m != "page":
            d["rows_all_fields_ok"] = rows[f"{m}_ok"].mean()
            r = rows[f"{m}_result"].dropna()
            d["result_number_ok"] = r.astype(bool).mean()
        summ.append(d)
    df = pd.DataFrame(summ)
    save_table(df, EXP_ID)
    print(f"{len(per)} ảnh · {int(per.n_chars.sum())} ký tự đáp án · {len(rows)} dòng chỉ số (có tên + kết quả)")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
