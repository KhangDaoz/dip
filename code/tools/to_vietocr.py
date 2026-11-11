from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from common.boxes import clip_box
from common.contracts import BOX_LABELS
from common.paths import CODE_DIR, VAIPE_CROPS
from vaipe.dataset import clean_text, list_images

SPLITS = ("train", "valid", "test")
VIETOCR_CONFIG = CODE_DIR / "configs" / "vietocr_vgg_transformer.yml"


def load_vocab(config_path: Path = VIETOCR_CONFIG) -> set:
    with open(config_path, encoding="utf-8") as f:
        return set(yaml.safe_load(f)["vocab"])


def crop_box(img_bgr: np.ndarray, box, pad: int = 2) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = clip_box(box, w, h)
    return img_bgr[max(y0 - pad, 0):min(y1 + pad, h), max(x0 - pad, 0):min(x1 + pad, w)]


def convert_split(split: str, out: Path, vocab: set, classes: set, pad: int) -> dict:
    (out / "img" / split).mkdir(parents=True)
    lines, meta = [], []
    per_class, skip, bad_chars = Counter(), Counter(), Counter()
    images = list_images(split)
    for img in images:
        bgr = cv2.imread(str(img.image_path))
        if bgr is None:
            raise SystemExit(f"Không đọc được {img.image_path}")
        h, w = bgr.shape[:2]
        for k, a in enumerate(img.annotations):
            if a.label not in classes:
                continue
            text = clean_text(a.text)
            x0, y0, x1, y1 = clip_box(a.box, w, h)
            if not text:
                skip["empty_text"] += 1
                continue
            if x1 - x0 < 2 or y1 - y0 < 2:
                skip["degenerate_box"] += 1
                continue
            oov = [c for c in text if c not in vocab]
            if oov:
                bad_chars.update(oov)
                skip["char_not_in_vocab"] += 1
                continue
            rel = f"img/{split}/{img.image_id}_{k:03d}.png"
            cv2.imwrite(str(out / rel), crop_box(bgr, a.box, pad))
            lines.append(f"{rel}\t{text}")
            meta.append(f"{rel}\t{a.label}\t{text}")
            per_class[a.label] += 1
    lines.sort()
    meta.sort()
    (out / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"{split}_meta.tsv").write_text("path\tlabel\ttext\n" + "\n".join(meta) + "\n",
                                           encoding="utf-8")
    print(f"[to_vietocr] {split:5s}: {len(images)} ảnh, {len(lines)} crops "
          f"({', '.join(f'{c}={per_class[c]}' for c in BOX_LABELS if c in classes)}); "
          f"bỏ {dict(skip)}")
    return {"images": len(images), "crops": len(lines),
            "crops_per_class": {c: per_class[c] for c in BOX_LABELS if c in classes},
            "skipped": dict(skip), "chars_not_in_vocab": dict(bad_chars)}


def main():
    ap = argparse.ArgumentParser(description="Cắt crops VietOCR từ nhãn VAIPE")
    ap.add_argument("--classes", nargs="+", default=list(BOX_LABELS), choices=BOX_LABELS)
    ap.add_argument("--pad", type=int, default=2, help="đệm pixel quanh box")
    ap.add_argument("--out", type=Path, default=VAIPE_CROPS)
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    vocab, classes = load_vocab(), set(args.classes)
    stats = {s: convert_split(s, out, vocab, classes, args.pad) for s in SPLITS}
    (out / "stats.json").write_text(
        json.dumps({"classes": args.classes, "pad": args.pad, **stats}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    print(f"[to_vietocr] xong: {out}")


if __name__ == "__main__":
    main()
