"""E-A1 / E-X2 — độ bền: pipeline DL trên ảnh suy giảm L0/L2/L4, có / không tiền xử lý khối A (plan §5.1). Người làm: Hiếu.

Cần sinh ảnh trước: python tools/make_degraded.py
    python experiments/e_a1_robustness.py --ocr C3 --enhance none light full
"""
import argparse

import _v2  # noqa: F401
import pandas as pd

from common.io import save_json
from common.paths import VAIPE_DEGRADED
from metrics.items import bootstrap_ci, evaluate
from pipeline import Pipeline
from vaipe.dataset import list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", default="C3")
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 2, 4])
    ap.add_argument("--enhance", nargs="+", default=["none", "light", "full"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    gt = _v2.load_items_gt("test")
    images = [img for img in list_images("test") if img.image_id in gt][:args.limit]
    gt = {img.image_id: gt[img.image_id] for img in images}
    out = _v2.out_dir("e_a1")
    rows = []
    for lv in args.levels:
        src = VAIPE_DEGRADED / f"L{lv}"
        if not src.exists():
            print(f"! thiếu {src} — chạy tools/make_degraded.py")
            continue
        for en in args.enhance:
            print(f"== L{lv} · enhance={en}")
            pipe = Pipeline(enhance=en, detector="yolo", ocr=args.ocr)
            res = _v2.run_cached(pipe, images, out / f"L{lv}_{en}", image_dir=src)
            agg, counts = evaluate(gt, _v2.items_of(res))
            f1, lo, hi = bootstrap_ci(counts, "item_f1")
            rows.append({"level": f"L{lv}", "enhance": en, "item_F1": f1, "CI95_lo": lo, "CI95_hi": hi,
                         "name_F1": agg["name_f1"], "qty_acc": agg["qty_acc"], **_v2.mean_timing(res)})
            print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(out / "table.csv", index=False, encoding="utf-8-sig")
    save_json(out / "config.json", {**vars(args), "commit": _v2.git_hash()})
    print(df.to_string(index=False))
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for en, g in df.groupby("enhance"):
            ax.errorbar(g["level"], g["item_F1"], yerr=[g["item_F1"] - g["CI95_lo"], g["CI95_hi"] - g["item_F1"]],
                        marker="o", capsize=3, label=en)
        ax.set_xlabel("Mức suy giảm")
        ax.set_ylabel("F1 khoản thuốc")
        ax.set_ylim(0, 1)
        ax.legend(title="Khối A")
        fig.tight_layout()
        fig.savefig(out / "robustness.png", dpi=200)
    except Exception as e:
        print("bỏ qua vẽ hình:", e)


if __name__ == "__main__":
    main()
