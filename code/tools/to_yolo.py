"""Chuyển nhãn VAIPE → định dạng YOLO (plan §5.2).

Mỗi ảnh → 1 file .txt trong <out>/labels/, mỗi dòng: `class_id cx cy w h` chuẩn hóa 0–1.
Sinh `data.yaml` cho ultralytics.

Chạy:
    python tools/to_yolo.py                  # sinh cả train/valid/test
    python tools/to_yolo.py --split train    # chỉ một split
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common.contracts import BOX_LABELS, LABEL_TO_ID
from common.boxes import to_yolo
from common.paths import VAIPE_YOLO
from vaipe.dataset import list_images


def convert_split(split: str, out_root: Path):
    """Sinh <out_root>/labels/<id>.txt + <out_root>/images/<id>.png (copy/symlink)."""
    img_out = out_root / "images"
    lbl_out = out_root / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    images = list_images(split)
    n = 0
    for img in images:
        # copy ảnh (symlink không dùng được trên Windows + Kaggle cần file thật)
        dst_img = img_out / img.image_path.name
        if not dst_img.exists():
            import shutil
            shutil.copy(img.image_path, dst_img)
        # đọc kích thước
        h, w = cv2.imread(str(img.image_path)).shape[:2]
        # ghi nhãn
        lines = []
        for a in img.annotations:
            cid = LABEL_TO_ID[a.label]
            cx, cy, bw, bh = to_yolo(a.box, w, h)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_out / f"{img.image_id}.txt").write_text("\n".join(lines), encoding="utf-8")
        n += 1
    print(f"[to_yolo] {split}: {n} ảnh → {out_root}")
    return n


def write_data_yaml(out_root: Path, train_dir: Path, valid_dir: Path):
    """Sinh data.yaml cho ultralytics."""
    yaml = f"""path: {out_root.as_posix()}
train: {train_dir.name}/images
val: {valid_dir.name}/images
test: test/images

names:
  0: drugname
  1: quantity
  2: usage
  3: diagnose
  4: date
  5: other
"""
    (out_root / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"[to_yolo] data.yaml → {out_root / 'data.yaml'}")


def main():
    ap = argparse.ArgumentParser(description="Chuyển nhãn VAIPE sang YOLO format")
    ap.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    ap.add_argument("--out", type=Path, default=None, help="thư mục xuất (mặc định VAIPE_YOLO)")
    args = ap.parse_args()
    out_root = args.out or VAIPE_YOLO
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    dirs = {}
    for s in splits:
        dirs[s] = out_root / s
        convert_split(s, dirs[s])
    if "train" in dirs and "valid" in dirs:
        write_data_yaml(out_root, dirs["train"], dirs["valid"])
    else:
        # always write data.yaml if both train+valid dirs exist on disk
        tr = out_root / "train"; va = out_root / "valid"
        if tr.exists() and va.exists():
            write_data_yaml(out_root, tr, va)


if __name__ == "__main__":
    main()
