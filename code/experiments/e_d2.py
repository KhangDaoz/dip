from __future__ import annotations

import argparse
import json
import subprocess

import pandas as pd

from assemble.group import assemble
from assemble.lexicon import MARGIN, TAU_HIGH, TAU_LOW, Lexicon
from assemble.load import load_khang
from assemble import vlm as V
from common.experiment import results_dir, save_config, save_table
from common.paths import REPO_DIR, vaipe_images, vaipe_split
from assemble.scoring import bootstrap_ci, date_accuracy, evaluate_d

EXP = "E-D2"
SPLIT = "test"


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()
    except Exception:
        return "unknown"


def score_row(name: str, results: dict, gt: dict, extra: dict | None = None):
    gt_items = {k: v["items"] for k, v in gt.items()}
    preds = {k: v[0] for k, v in results.items()}
    agg, counts = evaluate_d(gt_items, preds, use_lexicon=True)
    _, lo, hi = bootstrap_ci(counts, "item_f1")
    nolex, _ = evaluate_d(gt_items, preds, use_lexicon=False)
    row = {"pipeline": name, "f1_item": round(agg["item_f1"], 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
           "f1_item_nolex": round(nolex["item_f1"], 4), "f1_name": round(agg["name_f1"], 4),
           "qty_acc": round(agg["qty_acc"], 4), "exact_rate": round(agg["exact_rate"], 4),
           "date_acc": round(date_accuracy({k: v[2] for k, v in results.items()},
                                           {k: v["date"] for k, v in gt.items()}), 4),
           "n_json_fail": "", "n_api_error": ""}
    row.update(extra or {})
    print(f"[{EXP}] {name:<14} F1={row['f1_item']} [{row['ci_low']}, {row['ci_high']}] "
          f"không từ điển={row['f1_item_nolex']} F1 tên={row['f1_name']}")
    return row


def dump_preds(name: str, results: dict):
    path = results_dir(EXP) / "preds" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, (items, diag, date) in results.items():
            f.write(json.dumps({"image_id": k, "items": [it.to_dict() for it in items],
                                "diagnose": diag, "date": date}, ensure_ascii=False) + "\n")


def run_handoff(name: str, path: str, gt: dict, lex: Lexicon, use_gt_text: bool = False):
    data = load_khang(path, use_gt_text=use_gt_text)
    missing = [k for k in gt if k not in data]
    if missing:
        print(f"[{EXP}] ⚠️ {name}: thiếu {len(missing)} ảnh trong file (tính là đơn rỗng)")
    results = {k: assemble(data.get(k, []), lexicon=lex) for k in gt}
    dump_preds(name, results)
    return score_row(name, results, gt)


def run_vlm(gt: dict, lex: Lexicon, workers: int):
    cache = results_dir(EXP) / "vlm_raw"
    (results_dir(EXP) / "prompt.txt").write_text(V.PROMPT, encoding="utf-8")
    jobs = [(k, str(vaipe_images(SPLIT) / f"{k}.png")) for k in gt]
    recs = V.run_many(jobs, cache, workers=workers)
    api_errors = sum(bool(r.get("error")) for r in recs.values())
    fails = sum(r["parsed"] is None and not r.get("error") for r in recs.values())
    if api_errors:
        print(f"[{EXP}] ⚠️ VLM lỗi API ở {api_errors} ảnh (không cache) — chạy lại lệnh để gọi tiếp")
    results = {k: V.vlm_to_items(recs[k]["parsed"], lex) for k in gt}
    dump_preds("vlm", results)
    model = next((r.get("model") for r in recs.values() if r.get("model")), None)
    row = score_row("vlm", results, gt, {"n_json_fail": fails, "n_api_error": api_errors})
    return row, model


def main():
    ap = argparse.ArgumentParser(description="E-D2: bảng chính khối D")
    ap.add_argument("--variant", action="append", default=[], help="TÊN=ĐƯỜNG_DẪN file bàn giao")
    ap.add_argument("--check", default=None, help="TÊN biến thể dùng cho dòng kiểm 'text chuẩn'")
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    gt = json.loads((vaipe_split(SPLIT) / "items_gt.json").read_text(encoding="utf-8"))
    lex = Lexicon.from_split("train")
    variants = dict(v.split("=", 1) for v in args.variant)
    rows = [run_handoff(n, p, gt, lex) for n, p in variants.items()]
    if args.check:
        rows.append(run_handoff(f"{args.check}_gt_text", variants[args.check], gt, lex, use_gt_text=True))
    model = None
    if args.vlm:
        row, model = run_vlm(gt, lex, args.workers)
        rows.append(row)

    save_table(pd.DataFrame(rows), EXP)
    save_config({"experiment": EXP, "split": SPLIT, "n_images": len(gt), "commit": commit_hash(),
                 "variants": variants, "check": args.check, "vlm_model": model,
                 "lexicon": {"source": "train", "size": len(lex), "tau_high": TAU_HIGH, "tau_low": TAU_LOW,
                             "margin": MARGIN},
                 "conf_yolo": 0.5, "dedupe_iou": 0.7, "cer_thr": 0.1, "bootstrap": {"n": 1000, "seed": 42}}, EXP)


if __name__ == "__main__":
    main()
