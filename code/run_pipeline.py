import argparse
import time
from pathlib import Path

from common.io import read_image, save_json, write_image
from common.paths import OUT_DIR
from pipeline import PRESETS, Pipeline, draw_overlay

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="Trích xuất thông tin đơn thuốc")
    ap.add_argument("input", type=Path, help="ảnh hoặc thư mục ảnh")
    ap.add_argument("--preset", choices=list(PRESETS), default="dl_c3")
    ap.add_argument("--enhance", default="none", help="none | geom | light | full")
    ap.add_argument("--no-lexicon", action="store_true", help="tắt so khớp từ điển tên thuốc")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    paths = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"Không có ảnh nào trong {args.input}")
    pipe = Pipeline.preset(args.preset, enhance=args.enhance, **({"lexicon": None} if args.no_lexicon else {}))
    print("Cấu hình:", pipe.config())

    n_ok, t0 = 0, time.perf_counter()
    for p in paths:
        try:
            img = read_image(p)
            res = pipe(img, p.stem)
            save_json(args.out / f"{p.stem}.json", res.to_dict())
            write_image(args.out / f"{p.stem}_overlay.png", draw_overlay(img, res))
            n_ok += 1
            if not args.quiet:
                print(f"\n{p.stem}  ({len(res.items)} khoản; " + ", ".join(f"{k} {v:.2f}s" for k, v in res.timing.items()) + ")")
                for i, it in enumerate(res.items, 1):
                    print(f"  {i}. {it.lexicon_match or it.name} | SL: {it.quantity} {it.unit or ''} | {' / '.join(it.usage)}")
                print(f"  Chẩn đoán: {res.diagnose}\n  Ngày: {res.date}")
        except Exception as e:
            print(f"\n{p.stem}  LỖI: {type(e).__name__}: {e}")
    print(f"\nXong {n_ok}/{len(paths)} ảnh trong {time.perf_counter() - t0:.1f}s → {args.out}")
    return 0 if n_ok == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
