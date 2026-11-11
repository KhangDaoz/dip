"""E-C1 (+ E-C2 tùy chọn) — so bộ OCR trên vùng cắt theo box CHUẨN của test (plan §5.3). Người làm: Khang.

    python experiments/e_c1_ocr.py                          # C0, C1, C2, C3
    python experiments/e_c1_ocr.py --engines C2 C3 --limit 500
    python experiments/e_c1_ocr.py --c2                     # thêm E-C2: đầu vào gray/otsu/clahe cho C0, C2
Độ đo: CER, WER toàn tập + CER riêng drugname, quantity, usage; giây/vùng.
"""
import argparse
import time

import _v2  # noqa: F401
import pandas as pd

from common.io import read_image, save_json
from metrics.ocr import cer_by_label, corpus_cer, corpus_wer
from recognize import get_engine
from recognize.crop import crop_box, prepare
from vaipe.dataset import list_images


def load_crops(split, limit):
    crops, texts, labels = [], [], []
    for img in list_images(split):
        im = read_image(img.image_path)
        for a in img.annotations:
            if not a.text.strip():
                continue
            crops.append(crop_box(im, a.box, pad=2))
            texts.append(a.text.strip())
            labels.append(a.label)
            if limit and len(crops) >= limit:
                return crops, texts, labels
    return crops, texts, labels


def evaluate_engine(name, engine, crops, gts, labels, mode=None):
    import cv2
    scale = getattr(engine, "scale", 1.0)
    inp = []
    for c in crops:
        if scale != 1.0:
            c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        inp.append(prepare(c, mode) if mode else c)
    if hasattr(engine, "mode") and mode:   # Tesseract tự prepare → tắt để không làm 2 lần
        engine.mode = "color"
    if hasattr(engine, "scale"):
        engine.scale = 1.0
    t = time.perf_counter()
    preds = [p for p, _ in engine.read_batch(inp)]
    dt = (time.perf_counter() - t) / max(1, len(inp))
    by = cer_by_label(preds, gts, labels)
    return {"engine": name, "input": mode or "default", "CER": corpus_cer(preds, gts), "WER": corpus_wer(preds, gts),
            "CER_drugname": by.get("drugname"), "CER_quantity": by.get("quantity"), "CER_usage": by.get("usage"),
            "s_per_crop": dt, "n": len(gts)}, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["C0", "C1", "C2", "C3"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="số vùng tối đa (Tesseract chậm)")
    ap.add_argument("--c2", action="store_true", help="chạy thêm E-C2")
    args = ap.parse_args()
    out = _v2.out_dir("e_c1")
    crops, gts, labels = load_crops(args.split, args.limit)
    print(f"{len(crops)} vùng chữ ({args.split})")
    rows = []
    for e in args.engines:
        try:
            r, preds = evaluate_engine(e, get_engine(e), crops, gts, labels)
        except Exception as ex:
            print(f"! {e}: {type(ex).__name__}: {ex}")
            continue
        rows.append(r)
        print(r)
        pd.DataFrame({"label": labels, "gt": gts, "pred": preds}).to_csv(out / f"preds_{e}.csv", index=False,
                                                                           encoding="utf-8-sig")
    if args.c2:
        for e in ("C0", "C2"):
            for mode in ("gray", "otsu", "clahe"):
                try:
                    r, _ = evaluate_engine(e, get_engine(e), crops, gts, labels, mode=mode)
                    r["exp"] = "E-C2"
                    rows.append(r)
                    print(r)
                except Exception as ex:
                    print(f"! E-C2 {e}/{mode}: {ex}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "table.csv", index=False, encoding="utf-8-sig")
    save_json(out / "config.json", {**vars(args), "commit": _v2.git_hash()})
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
