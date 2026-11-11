"""Hàm dùng chung cho thí nghiệm v2 (E-*). Chạy mọi script từ code/: python experiments/e_xx.py"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
for p in (CODE / "src", CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.contracts import PrescriptionResult  # noqa: E402
from common.io import load_json, read_image, save_json  # noqa: E402
from common.paths import RESULTS_DIR, vaipe_split  # noqa: E402


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=CODE, text=True).strip()
    except Exception:
        return "unknown"


def load_items_gt(split: str = "test") -> dict:
    """{image_id: list khoản GT}. Cần chạy tools/build_items_gt.py trước."""
    path = vaipe_split(split) / "items_gt.json"
    if not path.exists():
        raise SystemExit(f"Thiếu {path} — chạy: python tools/build_items_gt.py --split {split}")
    return {d["image_id"]: d["items"] for d in load_json(path)}


def run_cached(pipe, images: list, cache_dir: Path, image_dir: Path | None = None, use_gt: bool = False,
               limit: int | None = None) -> dict:
    """Chạy `pipe` trên list VAIPEImage, lưu/đọc cache <cache_dir>/<id>.json. Trả về {image_id: PrescriptionResult}.

    image_dir: đọc ảnh từ thư mục khác (ảnh suy giảm) thay vì img.image_path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    imgs = images[:limit] if limit else images
    t0 = time.perf_counter()
    for k, img in enumerate(imgs, 1):
        f = cache_dir / f"{img.image_id}.json"
        if f.exists():
            out[img.image_id] = PrescriptionResult.from_dict(load_json(f))
            continue
        path = (image_dir / img.image_path.name) if image_dir else img.image_path
        try:
            res = pipe(read_image(path), img.image_id, gt=img.annotations if use_gt else None)
        except Exception as e:
            print(f"   ! {img.image_id}: {type(e).__name__}: {e}")
            res = PrescriptionResult(image_id=img.image_id)
        save_json(f, res.to_dict())
        out[img.image_id] = res
        if k % 20 == 0:
            print(f"   {k}/{len(imgs)}  ({(time.perf_counter() - t0) / k:.2f}s/ảnh)")
    return out


def items_of(results: dict) -> dict:
    return {k: v.items for k, v in results.items()}


def mean_timing(results: dict) -> dict:
    keys = {k for r in results.values() for k in r.timing}
    n = max(1, len(results))
    return {f"t_{k}": sum(r.timing.get(k, 0) for r in results.values()) / n for k in sorted(keys)}


def out_dir(exp: str) -> Path:
    d = RESULTS_DIR / exp
    d.mkdir(parents=True, exist_ok=True)
    return d
