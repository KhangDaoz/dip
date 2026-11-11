from __future__ import annotations

import json
import subprocess

import pandas as pd

from assemble.group import assemble
from common.contracts import TextBox
from common.experiment import save_config, save_table
from common.paths import REPO_DIR, vaipe_labels, vaipe_split
from assemble.scoring import evaluate_d
from vaipe.dataset import read_label_json

EXP = "E-D1"
SPLIT = "test"


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()
    except Exception:
        return "unknown"


def main():
    gt = json.loads((vaipe_split(SPLIT) / "items_gt.json").read_text(encoding="utf-8"))
    gt_items = {k: v["items"] for k, v in gt.items()}
    rows, bad = [], []
    for mode in ("labels", "text_only"):
        preds = {}
        for image_id in gt:
            anns = read_label_json(vaipe_labels(SPLIT) / f"{image_id}.json")
            boxes = [TextBox(box=a.box, label=a.label if mode == "labels" else None, score=1.0, text=a.text)
                     for a in anns]
            items, _, _ = assemble(boxes, lexicon=None, has_labels=(mode == "labels"))
            preds[image_id] = [{"name": it.name, "quantity": it.quantity} for it in items]
        agg, counts = evaluate_d(gt_items, preds, use_lexicon=False)
        n_bad = 0
        for k, c in zip(sorted(gt_items), counts):
            if c["fp"] or c["fn"]:
                n_bad += 1
                bad.append({"mode": mode, "image_id": k, "fp": c["fp"], "fn": c["fn"],
                            "pred": json.dumps(preds[k], ensure_ascii=False),
                            "gt": json.dumps([{"name": g["name"], "quantity": g["quantity"]} for g in gt_items[k]],
                                             ensure_ascii=False)})
        rows.append({"mode": mode, "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"],
                     "f1": round(agg["item_f1"], 4), "n_bad_images": n_bad})
        print(f"[{EXP}] {mode:<10} TP={agg['tp']} FP={agg['fp']} FN={agg['fn']} F1={agg['item_f1']:.4f} ảnh sai={n_bad}")

    save_table(pd.DataFrame(rows), EXP)
    save_table(pd.DataFrame(bad, columns=["mode", "image_id", "fp", "fn", "pred", "gt"]), EXP, "bad_cases")
    save_config({"experiment": EXP, "split": SPLIT, "n_images": len(gt), "commit": commit_hash(),
                 "lexicon": None, "conf_yolo": 0.5, "cer_thr": 0.1,
                 "input": f"data/1_vaipe/{SPLIT}/labels", "gt": f"data/1_vaipe/{SPLIT}/items_gt.json"}, EXP)


if __name__ == "__main__":
    main()
