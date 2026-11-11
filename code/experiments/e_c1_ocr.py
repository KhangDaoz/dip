import argparse
import csv
import time

import numpy as np
import pandas as pd

from common.experiment import results_dir, save_config, save_table
from common.io import read_image
from common.paths import VAIPE_CROPS
from metrics.ocr import cer_by_label, corpus_cer, corpus_wer

EXP_ID = "e_c1"
LABELS = ("drugname", "quantity", "usage", "diagnose", "date", "other")


def make_engine(name: str):
    if name == "C0":
        from recognize.tesseract_engine import TesseractEngine
        return TesseractEngine("gray")
    if name == "C1":
        from recognize.tesseract_engine import TesseractEngine
        return TesseractEngine("binary")
    if name == "C2":
        from recognize.vietocr_engine import VietOCREngine
        return VietOCREngine("pretrained")
    if name == "C3":
        from recognize.vietocr_engine import VietOCREngine
        return VietOCREngine("finetuned")
    raise ValueError(name)


def load_crops(split: str, limit, seed: int):
    with open(VAIPE_CROPS / f"{split}_meta.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if limit and limit < len(rows):
        idx = np.sort(np.random.default_rng(seed).choice(len(rows), limit, replace=False))
        rows = [rows[i] for i in idx]
    crops = [read_image(VAIPE_CROPS / r["path"]) for r in rows]
    return rows, crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["C0", "C1", "C2", "C3"], choices=["C0", "C1", "C2", "C3"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="số vùng (lấy mẫu ngẫu nhiên); mặc định tất cả")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = results_dir(EXP_ID)
    save_config(vars(args), EXP_ID)

    rows, crops = load_crops(args.split, args.limit, args.seed)
    gts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    print(f"{len(crops)} vùng chữ ({args.split})")

    table = []
    for name in args.engines:
        try:
            engine = make_engine(name)
            engine.read_batch(crops[:4])
        except Exception as e:
            print(f"! bỏ qua {name}: {type(e).__name__}: {e}")
            continue
        t = time.perf_counter()
        preds = [p for p, _ in engine.read_batch(crops)]
        sec = (time.perf_counter() - t) / max(1, len(crops))
        by = cer_by_label(preds, gts, labels)
        table.append({"engine": engine.name, "CER": corpus_cer(preds, gts), "WER": corpus_wer(preds, gts),
                      **{f"CER_{l}": by.get(l) for l in LABELS}, "ms_per_crop": 1000 * sec, "n": len(gts)})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in table[-1].items()})
        pd.DataFrame({"path": [r["path"] for r in rows], "label": labels, "gt": gts, "pred": preds}).to_csv(
            out / f"preds_{name}.csv", index=False, encoding="utf-8-sig")
    if not table:
        raise SystemExit("Không engine nào chạy được.")
    df = pd.DataFrame(table)
    save_table(df, EXP_ID, "table")
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
