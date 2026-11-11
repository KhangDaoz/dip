"""Chuyển nhãn VAIPE → bộ dữ liệu YOLO cho khối B (plan §5.2), 6 lớp.

Cùng cách với script đã dùng để train `weights/yolo_vaipe.pt` trên Kaggle
(UIT4meow_VAIPE/tools/prepare_yolo_pres.py): box kẹp vào ảnh, bỏ box suy biến (≤ 1 px).

Đầu ra (mặc định data/1_vaipe/yolo):
    images/{train,valid,test}/<id>.png   symlink tới ảnh gốc (copy nếu không symlink được)
    labels/{train,valid,test}/<id>.txt   "class cx cy w h" chuẩn hóa, mỗi box một dòng
    data.yaml                            đường dẫn tuyệt đối, cho ultralytics
    stats.json                           số ảnh, số box theo lớp, số box bị bỏ

Chạy:
    python tools/to_yolo.py
    python tools/to_yolo.py --out /tmp/yolo
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2

from common.boxes import clip_box, to_yolo
from common.contracts import BOX_LABELS, LABEL_TO_ID
from common.paths import VAIPE_YOLO
from vaipe.dataset import list_images

SPLITS = ("train", "valid", "test")


def link_or_copy(src: Path, dst: Path):
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_split(split: str, out_root: Path) -> dict:
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)
    per_class, n_skip = Counter(), 0
    images = list_images(split)
    for img in images:
        h, w = cv2.imread(str(img.image_path), cv2.IMREAD_UNCHANGED).shape[:2]
        lines = []
        for a in img.annotations:
            x0, y0, x1, y1 = clip_box(a.box, w, h)
            if x1 - x0 <= 1 or y1 - y0 <= 1:
                n_skip += 1
                continue
            cx, cy, bw, bh = to_yolo([x0, y0, x1, y1], w, h)
            lines.append(f"{LABEL_TO_ID[a.label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            per_class[a.label] += 1
        link_or_copy(img.image_path.resolve(), img_out / img.image_path.name)
        (lbl_out / f"{img.image_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[to_yolo] {split:5s}: {len(images)} ảnh, {sum(per_class.values())} box "
          f"({', '.join(f'{c}={per_class[c]}' for c in BOX_LABELS)}); bỏ {n_skip} box suy biến")
    return {"images": len(images), "boxes": sum(per_class.values()),
            "boxes_per_class": {c: per_class[c] for c in BOX_LABELS},
            "skipped_degenerate_boxes": n_skip}


def main():
    ap = argparse.ArgumentParser(description="Chuyển nhãn VAIPE sang định dạng YOLO")
    ap.add_argument("--out", type=Path, default=VAIPE_YOLO)
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    stats = {s: convert_split(s, out) for s in SPLITS}
    if not stats["train"]["images"]:
        raise SystemExit("Không thấy ảnh train — giải nén VAIPE vào data/1_vaipe trước.")
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(BOX_LABELS))
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/valid\ntest: images/test\n"
        f"names:\n{names}", encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"[to_yolo] xong: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
