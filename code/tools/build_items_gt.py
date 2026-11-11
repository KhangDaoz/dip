from __future__ import annotations

import argparse
import json
from pathlib import Path

from assemble.group import first_date, merge_diagnose, reading_order
from assemble.rules import clean_name, clean_usage, parse_quantity
from common.contracts import TextBox
from common.paths import RESULTS_DIR, vaipe_labels, vaipe_split
from vaipe.dataset import read_label_json


def _xywh_to_box(line: dict) -> TextBox:
    x, y, w, h = line["bbox"]
    return TextBox(box=[x, y, x + w, y + h], label=line["label"], score=1.0, text=line["text"])


def convert_image(v1: dict, label_path: Path) -> dict:
    items = []
    for it in v1["items"]:
        lines = [_xywh_to_box(l) for l in it["lines"]]
        drug = [b for b in lines if b.label == "drugname"]
        qty = [b for b in lines if b.label == "quantity"]
        usage = reading_order([b for b in lines if b.label == "usage"])
        q, unit = parse_quantity(qty[0].text) if qty else (None, None)
        items.append({"name": clean_name(drug[0].text if drug else it["name"]),
                      "quantity": q, "unit": unit, "usage": [clean_usage(u.text) for u in usage]})
    anns = read_label_json(label_path) if label_path.exists() else []
    boxes = [TextBox(box=a.box, label=a.label, score=1.0, text=a.text) for a in anns]
    first_y = min((b.box[1] for b in boxes if b.label == "drugname"), default=float("inf"))
    n_drug = sum(b.label == "drugname" for b in boxes)
    return {"items": items, "diagnose": merge_diagnose(boxes, first_y), "date": first_date(boxes),
            "reviewed": bool(v1.get("reviewed")) and bool(v1.get("review_ok")),
            "_n_drug_labels": n_drug}


def main():
    ap = argparse.ArgumentParser(description="Dựng items_gt.json từ đáp án v1")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--src", type=Path, default=None, help="mặc định results/s0/gt_items_<split>.json")
    args = ap.parse_args()
    src = args.src or RESULTS_DIR / "s0" / f"gt_items_{args.split}.json"
    if not src.exists():
        raise SystemExit(f"Không thấy {src} — lấy file đáp án v1 từ Drive nhóm (results/ không có trong git)")
    v1 = json.loads(src.read_text(encoding="utf-8"))

    out, bad = {}, []
    for image_id, g in sorted(v1.items()):
        rec = convert_image(g, vaipe_labels(args.split) / f"{image_id}.json")
        if rec.pop("_n_drug_labels") != len(rec["items"]):
            bad.append(image_id)
        if any(it["quantity"] is None for it in rec["items"]):
            bad.append(image_id)
        out[image_id] = rec

    dst = vaipe_split(args.split) / "items_gt.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    n_items = sum(len(v["items"]) for v in out.values())
    n_rev = sum(v["reviewed"] for v in out.values())
    print(f"[build_items_gt] {args.split}: {len(out)} ảnh, {n_items} khoản, {n_rev} ảnh đã soát → {dst}")
    print(f"[build_items_gt] vi phạm bất biến (số khoản ≠ số drugname, hoặc thiếu SL): {sorted(set(bad)) or 0}")
    print(f"[build_items_gt] có diagnose: {sum(bool(v['diagnose']) for v in out.values())}, "
          f"có date: {sum(bool(v['date']) for v in out.values())} — nhớ soát tay 10 ảnh")


if __name__ == "__main__":
    main()
