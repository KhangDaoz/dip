from __future__ import annotations

import argparse
import json
import time

from common.io import read_image
from common.paths import RESULTS_DIR
from pipeline import PRESETS, Pipeline, handoff_record
from vaipe.dataset import list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", nargs="+", default=["dl_c3"], choices=list(PRESETS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out_dir = RESULTS_DIR / "handoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(args.split)[:args.limit]

    for name in args.preset:
        path = out_dir / f"{name}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["image_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
        todo = [im for im in images if im.image_id not in done]
        print(f"[{name}] {len(done)} ảnh đã có, còn {len(todo)}")
        pipe = Pipeline.preset(name, lexicon=None)
        times = {"B": 0.0, "C": 0.0}
        t0 = time.perf_counter()
        with open(path, "a", encoding="utf-8") as f:
            for k, im in enumerate(todo, 1):
                _, boxes, timing = pipe.read(read_image(im.image_path))
                for key in times:
                    times[key] += timing[key]
                f.write(json.dumps(handoff_record(im.image_id, boxes, gt=im.annotations), ensure_ascii=False) + "\n")
                f.flush()
                if k % 20 == 0:
                    print(f"  {k}/{len(todo)} ({(time.perf_counter() - t0) / k:.2f}s/ảnh)")
        if todo:
            (out_dir / f"{name}_timing.json").write_text(json.dumps(
                {"n": len(todo), **{f"s_per_img_{k}": v / len(todo) for k, v in times.items()}}, indent=2),
                encoding="utf-8")
        print(f"[{name}] xong → {path}")


if __name__ == "__main__":
    main()
