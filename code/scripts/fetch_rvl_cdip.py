"""Lấy bộ 4 — tài liệu "khác" (lớp `other`) từ RVL-CDIP. Phụ trách: Hiếu.

Nguồn: HuggingFace `jordyvl/rvl_cdip_100_examples_per_class` (16 lớp × 100 ảnh, đã chia sẵn
50 train / 25 validation / 25 test mỗi lớp, ~180 MB) — tập con của RVL-CDIP (Harley et al., 2015).
License RVL-CDIP: chỉ dùng cho nghiên cứu (ảnh gốc thuộc IIT-CDIP / Legacy Tobacco Documents Library).

Kết quả (cùng cấu trúc với bộ 1–3):
    data/4_other_rvl_cdip/
    ├── train/images/   valid/images/   test/images/   ← 8 lớp TRAIN_CLASSES, nhãn doc_type = other
    ├── openset/images/                                ← OPENSET_CLASSES, KHÔNG dùng khi train (KeHoach_Hieu.md mục 3.5)
    ├── labels.csv                                     ← file, split, rvl_class, doc_type
    └── README.md

Chạy (từ thư mục code/):  python scripts/fetch_rvl_cdip.py
Cần: pip install pyarrow
"""
import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common.io import ensure_dir, write_image
from common.paths import DATA_DIR, DATASETS

REPO = "jordyvl/rvl_cdip_100_examples_per_class"
PARQUETS = {
    "train": "data/train-00000-of-00001-81f1d229db782541.parquet",
    "valid": "data/validation-00000-of-00001-00031909a6e73300.parquet",
    "test": "data/test-00000-of-00001-d5e0db6590d27073.parquet",
}
LABEL_NAMES = ["letter", "form", "email", "handwritten", "advertisement", "scientific report",
               "scientific publication", "specification", "file folder", "news article", "budget",
               "invoice", "presentation", "questionnaire", "resume", "memo"]
# KeHoach_Hieu.md mục 2: các lớp giống "giấy tờ" người dùng hay chụp nhầm
TRAIN_CLASSES = ["form", "invoice", "letter", "memo", "email", "news article", "scientific report",
                 "questionnaire"]
# KeHoach_Hieu.md mục 3.5: loại lạ không y tế, chỉ dùng để đánh giá open-set
OPENSET_CLASSES = ["budget", "resume"]


def download(rel_path: str, cache_dir: Path) -> Path:
    dst = cache_dir / Path(rel_path).name
    if not dst.exists():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel_path}"
        print("Tải", url)
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst


def slug(name: str) -> str:
    return name.replace(" ", "_")


def fetch(out_dir: Path, cache_dir: Path) -> pd.DataFrame:
    rows = []
    for split, rel in PARQUETS.items():
        df = pd.read_parquet(download(rel, cache_dir))
        for i, rec in enumerate(df.itertuples(index=False)):
            cls = LABEL_NAMES[int(rec.label)]
            if cls in TRAIN_CLASSES:
                target_split = split
            elif cls in OPENSET_CLASSES:
                target_split = "openset"
            else:
                continue
            img = np.array(Image.open(io.BytesIO(rec.image["bytes"])).convert("L"))
            name = f"rvl_{split}_{slug(cls)}_{i:04d}.png"
            write_image(out_dir / target_split / "images" / name, img)
            rows.append({"file": f"{target_split}/images/{name}", "split": target_split,
                         "rvl_class": cls, "doc_type": "other"})
    labels = pd.DataFrame(rows)
    labels.to_csv(out_dir / "labels.csv", index=False, encoding="utf-8")
    return labels


def write_readme(out_dir: Path, labels: pd.DataFrame) -> None:
    table = labels.pivot_table(index="rvl_class", columns="split", values="file", aggfunc="count",
                               fill_value=0)
    table = table.reindex(columns=[c for c in ["train", "valid", "test", "openset"] if c in table.columns])
    lines = ["| Lớp RVL-CDIP | " + " | ".join(table.columns) + " |",
             "| :--- |" + " ---: |" * len(table.columns)]
    for cls, r in table.iterrows():
        lines.append(f"| {cls} | " + " | ".join(str(v) for v in r.values) + " |")
    total = " | ".join(str(v) for v in table.sum().values)
    lines.append(f"| **Tổng** | {total} |")
    (out_dir / "README.md").write_text(f"""# Bộ 4 — Tài liệu khác (RVL-CDIP)

Lớp `other` cho phân loại 3 lớp (DIP D10). Sinh bằng `code/scripts/fetch_rvl_cdip.py`.

* **Nguồn:** HuggingFace [`{REPO}`](https://huggingface.co/datasets/{REPO}), tập con 100 ảnh / lớp của
  RVL-CDIP (Harley, Ufkes, Derpanis — ICDAR 2015). Giữ nguyên cách chia train / validation / test của nguồn.
* **License:** RVL-CDIP chỉ dùng cho **nghiên cứu**; ảnh gốc từ IIT-CDIP / Legacy Tobacco Documents Library.
* **Ảnh:** xám, PNG, độ phân giải gốc của RVL-CDIP (cạnh dài ~1000 px, ảnh scan).
* **`openset/`:** lớp {", ".join(OPENSET_CLASSES)} — **không dùng khi train**, chỉ để đánh giá tài liệu lạ (KeHoach_Hieu.md mục 3.5).
* **`labels.csv`:** `file`, `split`, `rvl_class`, `doc_type`.

{chr(10).join(lines)}

**Lưu ý shortcut:** ảnh RVL-CDIP là ảnh scan xám, độ phân giải thấp — khác hẳn bộ 1 (ảnh phẳng màu) và bộ 2, 3
(ảnh chụp 640×640). Bắt buộc chuẩn hóa ảnh trước khi trích đặc trưng (KeHoach_Hieu.md mục 3.5).
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Lấy bộ 4 (RVL-CDIP) cho lớp other")
    parser.add_argument("--out", type=Path, default=DATASETS[4])
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "_cache" / "rvl_cdip")
    args = parser.parse_args()
    labels = fetch(ensure_dir(args.out), ensure_dir(args.cache))
    write_readme(args.out, labels)
    print(labels.groupby("split").size().to_string())
    print("Xong:", args.out)


if __name__ == "__main__":
    main()
