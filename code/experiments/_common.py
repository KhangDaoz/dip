"""Hàm dùng chung cho E-X1, E-A1: chấm F1 khoản thuốc ĐÚNG như E-D2 (assemble.scoring của khối D)."""
from __future__ import annotations

import json
from pathlib import Path

from assemble.group import assemble
from assemble.load import load_khang
from assemble.scoring import bootstrap_ci, date_accuracy, evaluate_d
from common.paths import vaipe_split


def load_gt(split: str = "test") -> dict:
    path = vaipe_split(split) / "items_gt.json"
    if not path.exists():
        raise SystemExit(f"Thiếu {path} — chạy: python tools/build_items_gt.py --split {split}")
    return json.loads(path.read_text(encoding="utf-8"))


def score(name: str, results: dict, gt: dict) -> dict:
    """results = {image_id: (items, diagnose, date)} → một dòng bảng, cùng cột với E-D2."""
    gt_items = {k: v["items"] for k, v in gt.items()}
    preds = {k: results.get(k, ([], None, None))[0] for k in gt}
    agg, counts = evaluate_d(gt_items, preds, use_lexicon=True)
    _, lo, hi = bootstrap_ci(counts, "item_f1")
    nolex, _ = evaluate_d(gt_items, preds, use_lexicon=False)
    row = {"config": name, "f1_item": round(agg["item_f1"], 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
           "f1_item_nolex": round(nolex["item_f1"], 4), "f1_name": round(agg["name_f1"], 4),
           "qty_acc": round(agg["qty_acc"], 4), "exact_rate": round(agg["exact_rate"], 4),
           "date_acc": round(date_accuracy({k: results.get(k, (None, None, None))[2] for k in gt},
                                           {k: v["date"] for k, v in gt.items()}), 4)}
    print(f"  {name:<28} F1={row['f1_item']} [{row['ci_low']}, {row['ci_high']}]  tên={row['f1_name']}  "
          f"SL={row['qty_acc']}  ngày={row['date_acc']}")
    return row


def assemble_handoff(path: Path, lexicon, use_gt_text: bool = False) -> dict:
    """File bàn giao B+C → {image_id: (items, diagnose, date)} qua khối D."""
    return {k: assemble(v, lexicon=lexicon) for k, v in load_khang(path, use_gt_text=use_gt_text).items()}
