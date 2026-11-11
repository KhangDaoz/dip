"""E-X1 — ablation: thay từng khối của pipeline DL bằng đáp án chuẩn để biết lỗi đến từ đâu (plan §5.5). Người làm: Hiếu.

Dòng: (B chuẩn, C chuẩn) · (B chuẩn, C dự đoán) · (B dự đoán, C chuẩn) · (B dự đoán, C dự đoán = pipeline dl).
    python experiments/e_x1_ablation.py --ocr C3     # hoặc C2 nếu chưa có C3
"""
import argparse

import _v2  # noqa: F401
import pandas as pd

from common.io import save_json
from metrics.items import bootstrap_ci, evaluate
from pipeline import Pipeline
from vaipe.dataset import list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", default="C3")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    gt = _v2.load_items_gt("test")
    images = [img for img in list_images("test") if img.image_id in gt][:args.limit]
    gt = {img.image_id: gt[img.image_id] for img in images}
    out = _v2.out_dir("e_x1")
    rows = []
    for det, ocr in (("gt", "gt"), ("gt", args.ocr), ("yolo", "gt"), ("yolo", args.ocr)):
        name = f"B={'chuẩn' if det == 'gt' else 'YOLO'}, C={'chuẩn' if ocr == 'gt' else ocr}"
        print("==", name)
        res = _v2.run_cached(Pipeline(detector=det, ocr=ocr), images, out / f"{det}_{ocr}", use_gt=True)
        agg, counts = evaluate(gt, _v2.items_of(res))
        f1, lo, hi = bootstrap_ci(counts, "item_f1")
        rows.append({"config": name, "item_F1": f1, "CI95_lo": lo, "CI95_hi": hi, "name_F1": agg["name_f1"],
                     "qty_acc": agg["qty_acc"], "exact_rate": agg["exact_rate"]})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(out / "table.csv", index=False, encoding="utf-8-sig")
    save_json(out / "config.json", {**vars(args), "commit": _v2.git_hash()})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
