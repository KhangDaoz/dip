"""Chạy pipeline v2 đầu-cuối trên một ảnh hoặc thư mục (plan §4).

    python run_pipeline.py samples/                          # pipeline chính: YOLO + VietOCR C3 + luật
    python run_pipeline.py samples/ --preset dl_c2           # VietOCR pretrained (khi chưa có C3)
    python run_pipeline.py samples/ --preset classical       # đối chứng cổ điển (không cần trọng số)
    python run_pipeline.py anh.png --enhance full            # bật tiền xử lý A1–A5
    python run_pipeline.py samples/ --vlm                    # đối chứng VLM (cần VLM_API_KEY)

Ghi out/<image_id>.json (PrescriptionResult) + out/<image_id>_overlay.png.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from common.io import read_image, save_json, write_image  # noqa: E402
from common.paths import OUT_DIR  # noqa: E402
from pipeline import PRESET_PIPELINES, Pipeline, draw_overlay  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_inputs(p: Path) -> list:
    if p.is_file():
        return [p]
    return sorted(x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS)


def print_result(res):
    print(f"\n{res.image_id}  ({len(res.items)} khoản, " +
          ", ".join(f"{k} {v:.2f}s" for k, v in res.timing.items()) + ")")
    for i, it in enumerate(res.items, 1):
        name = it.lexicon_match or it.name
        print(f"  {i}. {name} | SL: {it.quantity} {it.unit or ''} | {' / '.join(it.usage)}")
    print(f"  Chẩn đoán: {res.diagnose}\n  Ngày: {res.date}")


def main():
    ap = argparse.ArgumentParser(description="Trích xuất đơn thuốc VAIPE")
    ap.add_argument("input", type=Path, help="ảnh hoặc thư mục ảnh")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--preset", choices=list(PRESET_PIPELINES), default="dl")
    ap.add_argument("--detector", choices=["yolo", "classical"])
    ap.add_argument("--ocr", choices=["C0", "C1", "C2", "C3"])
    ap.add_argument("--enhance", default="none", help="none | geom | light | full")
    ap.add_argument("--no-lexicon", action="store_true", help="tắt so khớp từ điển D3")
    ap.add_argument("--vlm", action="store_true", help="chạy đối chứng VLM thay pipeline")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    paths = list_inputs(args.input)
    if not paths:
        raise SystemExit(f"Không có ảnh nào trong {args.input}")

    if args.vlm:
        from assemble.vlm import extract
        run = lambda img, iid: extract(img, iid)[0]   # noqa: E731
    else:
        kw = {k: v for k, v in (("detector", args.detector), ("ocr", args.ocr)) if v}
        pipe = Pipeline.preset(args.preset, enhance=args.enhance, lexicon=None if args.no_lexicon else "default", **kw)
        if not args.quiet:
            print("Cấu hình:", pipe.config())
        run = pipe

    n_ok = 0
    t0 = time.perf_counter()
    for p in paths:
        try:
            img = read_image(p)
            res = run(img, p.stem)
            save_json(args.out / f"{p.stem}.json", res.to_dict())
            write_image(args.out / f"{p.stem}_overlay.png", draw_overlay(img, res))
            n_ok += 1
            if not args.quiet:
                print_result(res)
        except Exception as e:   # một ảnh lỗi không chặn ảnh khác
            print(f"\n{p.stem}  LỖI: {type(e).__name__}: {e}")
    print(f"\nXong {n_ok}/{len(paths)} ảnh trong {time.perf_counter() - t0:.1f}s → {args.out}")
    return 0 if n_ok == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
