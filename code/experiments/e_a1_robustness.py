"""E-A1 — tiền xử lý cổ điển (khối A) có giúp pipeline chính khi ảnh xấu đi? (plan §5.1, E-X2)

Pipeline chính (YOLO + C3 + D) trên ảnh test suy giảm L0 / L2 / L4, có / không khối A. Chấm F1 khoản thuốc như E-D2.
Ảnh suy giảm lệch hình học so với nhãn → chỉ đo đầu-cuối (không cần box).
L0 + none trùng dòng dl_c3 của E-D2 → dùng lại results/handoff/dl_c3.jsonl nếu có.
Mỗi tổ hợp ~3,5 s/ảnh trên RTX 3050 (VietOCR là phần chậm) → 176 ảnh ≈ 10 phút / tổ hợp.

    python experiments/e_a1_robustness.py                                   # L0 L2 L4 × none full
    python experiments/e_a1_robustness.py --levels L0 L4 --enhance none full # cắt theo plan §8.4
Kết quả: results/e_a1/{table.csv, robustness.png, config.json}; cache results/e_a1/<mức>_<A>.jsonl
"""
import argparse
import json
import time

import pandas as pd

from _common import assemble_handoff, load_gt, score
from assemble.lexicon import Lexicon
from common.experiment import results_dir, save_config, save_figure, save_table
from common.io import read_image
from common.paths import RESULTS_DIR, VAIPE_DEGRADED
from pipeline import Pipeline, handoff_record

EXP_ID = "e_a1"


def run_combo(level, enh, ids, path):
    """Chạy A → B → C trên ảnh mức `level`, lưu bàn giao (cache, chạy tiếp được). Trả về giây/ảnh."""
    done = set()
    if path.exists():
        done = {json.loads(l)["image_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [i for i in ids if i not in done]
    if not todo:
        return None
    pipe, t0 = Pipeline.preset("dl_c3", enhance=enh, lexicon=None), time.perf_counter()
    with open(path, "a", encoding="utf-8") as f:
        for k, iid in enumerate(todo, 1):
            _, boxes, _ = pipe.read(read_image(VAIPE_DEGRADED / level / f"{iid}.png"))
            f.write(json.dumps(handoff_record(iid, boxes), ensure_ascii=False) + "\n")
            if k % 25 == 0:
                print(f"  {k}/{len(todo)} ({(time.perf_counter() - t0) / k:.1f}s/ảnh)")
    return (time.perf_counter() - t0) / len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["L0", "L2", "L4"])
    ap.add_argument("--enhance", nargs="+", default=["none", "full"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = results_dir(EXP_ID)
    gt = load_gt("test")
    ids = sorted(gt)[:args.limit]
    gt = {k: gt[k] for k in ids}
    lex = Lexicon.from_split("train")
    rows = []
    for lv in args.levels:
        if not (VAIPE_DEGRADED / lv).exists():
            raise SystemExit(f"Thiếu {VAIPE_DEGRADED / lv} — chạy: python tools/make_degraded.py")
        for enh in args.enhance:
            path = out / f"{lv}_{enh}.jsonl"
            reuse = RESULTS_DIR / "handoff" / "dl_c3.jsonl"
            if lv == "L0" and enh == "none" and reuse.exists() and not path.exists():
                path = reuse                                  # ảnh L0 = ảnh gốc → trùng dl_c3 của E-D2
                print(f"[{lv}/{enh}] dùng lại {reuse}")
            else:
                print(f"[{lv}/{enh}]")
                run_combo(lv, enh, ids, path)
            row = score(f"{lv} · A={enh}", {k: v for k, v in assemble_handoff(path, lex).items() if k in gt}, gt)
            rows.append({"level": lv, "enhance": enh, **row})
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID)
    save_config(vars(args), EXP_ID)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for enh, g in df.groupby("enhance"):
        ax.errorbar(g["level"], g["f1_item"], yerr=[g["f1_item"] - g["ci_low"], g["ci_high"] - g["f1_item"]],
                    marker="o", capsize=3, label={"none": "không khối A", "full": "khối A đầy đủ"}.get(enh, enh))
    ax.set_xlabel("Mức suy giảm")
    ax.set_ylabel("F1 khoản thuốc (CI 95 %)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, EXP_ID, "robustness")


if __name__ == "__main__":
    main()
