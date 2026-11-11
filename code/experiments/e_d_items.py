"""E-D1 / E-D2 — F1 khoản thuốc đầu-cuối (plan §5.4). Người làm: Thắng. **Bảng chính của báo cáo.**

E-D1: luật D2 trên box + chữ chuẩn (B chuẩn, C chuẩn) so với items_gt → kỳ vọng ≈ 1 (kiểm tra luật).
E-D2: (i) classical · (ii) hybrid · (iii) dl (C3, chính) · (iv) dl_c2 · (v) vlm — bootstrap 1000 lần CI 95 %.

    python experiments/e_d_items.py                                   # đủ 5 pipeline trên 176 test
    python experiments/e_d_items.py --pipelines dl dl_c2 --limit 20   # thử nhanh
Dự đoán được cache ở results/e_d2/<pipeline>/ — chạy lại chỉ tính phần thiếu. Xóa thư mục để chạy lại từ đầu.
"""
import argparse

import _v2  # noqa: F401
import pandas as pd

from common.io import save_json
from metrics.items import bootstrap_ci, evaluate
from pipeline import Pipeline
from vaipe.dataset import list_images

ALL = ["classical", "hybrid", "dl", "dl_c2", "vlm"]


def run_vlm(images, cache, workers):
    from assemble.lexicon import default_lexicon
    from assemble.vlm import PROMPT, extract_many
    (cache.parent / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    res = extract_many([img.image_path for img in images], workers=workers, out_dir=cache, lexicon=default_lexicon())
    return {r.image_id: r for r in res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8, help="số luồng VLM")
    ap.add_argument("--no-lexicon", action="store_true")
    args = ap.parse_args()
    gt = _v2.load_items_gt(args.split)
    images = [img for img in list_images(args.split) if img.image_id in gt][:args.limit]
    gt = {img.image_id: gt[img.image_id] for img in images}

    # E-D1
    out1 = _v2.out_dir("e_d1")
    d1 = _v2.run_cached(Pipeline(detector="gt", ocr="gt", lexicon=None), images, out1 / "pred", use_gt=True)
    agg, counts = evaluate(gt, _v2.items_of(d1), use_lexicon=False)
    bad = [i for i, c in zip(sorted(gt), counts) if not c["exact"]]
    save_json(out1 / "result.json", {**agg, "wrong_images": bad})
    print(f"E-D1 F1 khoản = {agg['item_f1']:.4f}, đơn sai: {len(bad)} {bad[:10]}")

    # E-D2
    out = _v2.out_dir("e_d2")
    rows = []
    for name in args.pipelines:
        print(f"\n== {name}")
        cache = out / name
        try:
            if name == "vlm":
                res = run_vlm(images, cache, args.workers)
            else:
                pipe = Pipeline.preset(name, lexicon=None if args.no_lexicon else "default")
                res = _v2.run_cached(pipe, images, cache)
        except Exception as e:
            print(f"! bỏ qua {name}: {type(e).__name__}: {e}")
            continue
        agg, counts = evaluate(gt, _v2.items_of(res), use_lexicon=not args.no_lexicon)
        f1, lo, hi = bootstrap_ci(counts, "item_f1", n=1000)
        rows.append({"pipeline": name, "item_F1": f1, "CI95_lo": lo, "CI95_hi": hi, "item_P": agg["item_p"],
                     "item_R": agg["item_r"], "name_F1": agg["name_f1"], "qty_acc": agg["qty_acc"],
                     "exact_rate": agg["exact_rate"], **_v2.mean_timing(res), "n": agg["n_images"]})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(out / "table.csv", index=False, encoding="utf-8-sig")
    save_json(out / "config.json", {**vars(args), "commit": _v2.git_hash()})
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
