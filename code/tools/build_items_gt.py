"""Sinh đáp án khoản thuốc items_gt.json từ nhãn VAIPE bằng chính luật ghép D2 (plan §2.3).

    python tools/build_items_gt.py --split test        # → data/1_vaipe/test/items_gt.json
    python tools/build_items_gt.py --split all --review 10   # + in 10 ảnh test ngẫu nhiên để soát bằng mắt

Kiểm tra tự động: số khoản = số box drugname; cảnh báo khoản thiếu quantity hoặc không có usage.
Kết quả soát tay ghi vào results/gt_review.csv (image_id, ok, ghi_chu).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assemble.group import assemble  # noqa: E402
from common.io import save_json  # noqa: E402
from common.paths import RESULTS_DIR, vaipe_split  # noqa: E402
from vaipe.dataset import list_images  # noqa: E402


def build(split: str) -> tuple:
    out, warns = [], []
    for img in list_images(split):
        d = assemble(img.annotations)
        n_drug = sum(a.label == "drugname" for a in img.annotations)
        if len(d["items"]) != n_drug:
            warns.append(f"{img.image_id}: {len(d['items'])} khoản ≠ {n_drug} drugname")
        for it in d["items"]:
            if it.quantity is None:
                warns.append(f"{img.image_id}: '{it.name}' thiếu quantity")
            if not it.usage:
                warns.append(f"{img.image_id}: '{it.name}' không có usage")
        out.append({"image_id": img.image_id,
                    "items": [{"name": it.name, "quantity": it.quantity, "unit": it.unit, "usage": it.usage}
                              for it in d["items"]],
                    "diagnose": d["diagnose"], "date": d["date"]})
    return out, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    ap.add_argument("--review", type=int, default=0, help="in N ảnh test ngẫu nhiên (seed 42) để soát")
    args = ap.parse_args()
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    for s in splits:
        gt, warns = build(s)
        path = save_json(vaipe_split(s) / "items_gt.json", gt)
        n_items = sum(len(g["items"]) for g in gt)
        print(f"[items_gt] {s}: {len(gt)} ảnh, {n_items} khoản → {path} ({len(warns)} cảnh báo)")
        for w in warns[:20]:
            print("   !", w)
        if len(warns) > 20:
            print(f"   ... còn {len(warns) - 20} cảnh báo")
        if args.review and s == "test":
            pick = random.Random(42).sample(gt, min(args.review, len(gt)))
            review = RESULTS_DIR / "gt_review.csv"
            review.parent.mkdir(parents=True, exist_ok=True)
            if not review.exists():
                review.write_text("image_id,ok,ghi_chu\n" + "".join(f"{g['image_id']},,\n" for g in pick),
                                  encoding="utf-8-sig")
            for g in pick:
                print(f"\n== {g['image_id']} | chẩn đoán: {g['diagnose']} | ngày: {g['date']}")
                for i, it in enumerate(g["items"], 1):
                    print(f"   {i}. {it['name']} | {it['quantity']} {it['unit']} | {' / '.join(it['usage'])}")
            print(f"\nĐiền kết quả soát vào {review}")


if __name__ == "__main__":
    main()
