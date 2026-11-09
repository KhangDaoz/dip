"""D9 — Thời gian chạy từng bước của pipeline (CPU). Kế hoạch: KeHoach.md mục 5.2 (D9), 12.1 (mục 4.7 báo cáo).

Dữ liệu: 10 ảnh trong code/samples/, mỗi ảnh chạy 3 lần, lấy trung vị. Loại tài liệu cho trước (đo riêng bộ phân loại).
Bước: ① tiền xử lý (gồm phân loại) · ② làm sạch · ③ dựng bảng hoặc ④ tách khoản thuốc (không OCR).
Kết quả: results/d9/

Chạy (từ code/): python experiments/d9_runtime.py   (nên chạy khi máy rảnh)
"""
import tempfile
import time

import numpy as np
import pandas as pd

from cleaning.pipeline import clean
from common.experiment import results_dir, save_config, save_table
from common.io import read_image
from common.paths import REPO_DIR, SAMPLES_DIR
from common.seed import set_seed
from prescription_segmentation.pipeline import segment
from preprocessing.pipeline import preprocess
from table_extraction.pipeline import structure

EXP_ID = "d9"
CONFIG = {"seed": 42, "repeats": 3}


def doc_type_of(name):
    return "lab_report" if name.startswith("lab") else "prescription" if name.startswith("rx") else "other"


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in sorted(SAMPLES_DIR.glob("*")):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img = read_image(p)
            dt = doc_type_of(p.name)
            for _ in range(CONFIG["repeats"]):
                t = {}
                t0 = time.perf_counter()
                pre = preprocess(img, p.stem, tmp)                   # có phân loại
                t["1_preprocess_classify_ms"] = (time.perf_counter() - t0) * 1000
                if dt == "other":
                    rows.append({"image": p.name, "doc_type": dt, **t})
                    continue
                t0 = time.perf_counter()
                cl = clean(read_image(REPO_DIR / pre.rectified_path), dt, p.stem, tmp)
                t["2_clean_ms"] = (time.perf_counter() - t0) * 1000
                b = read_image(REPO_DIR / cl.binary_path, gray=True)
                t0 = time.perf_counter()
                (structure(b) if dt == "lab_report" else segment(b))
                t["3_4_structure_ms"] = (time.perf_counter() - t0) * 1000
                rows.append({"image": p.name, "doc_type": dt, "width": img.shape[1], "height": img.shape[0], **t})
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID)
    med = df.groupby(["image", "doc_type"]).median(numeric_only=True).reset_index()
    summ = med.groupby("doc_type").median(numeric_only=True)
    summ["total_ms"] = summ[[c for c in summ.columns if c.endswith("_ms")]].sum(axis=1)
    save_table(summ.reset_index(), EXP_ID, "summary")
    print(summ.round(0).to_string())


if __name__ == "__main__":
    main()
