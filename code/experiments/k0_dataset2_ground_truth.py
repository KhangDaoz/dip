"""Derive phrase-level row/column ground truth from the Roboflow Dataset 2 labels (57 original images).

The source annotations are semantic text spans (for example, a complete test
parameter phrase), not guaranteed one-box-per-token labels. Rows are inferred
from y-centres; columns are retained as semantic roles from the source labels.

Run from ``code/`` after installing the package:
    python experiments/k0_dataset2_ground_truth.py

Outputs go under ``results/khang_dataset2_ground_truth/`` (git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from common.paths import DATASETS, RESULTS_DIR


ROLE_MAP = {
    "Report Type": "report_type",
    "parameter": "parameter",
    "value": "result",
    "unit": "unit",
    "reference value": "range",
}
TABLE_ROLE_ORDER = ("parameter", "result", "unit", "range")


def load_class_names(dataset_dir: Path) -> list[str]:
    with (dataset_dir / "data.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    names = config.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"Không tìm thấy danh sách names trong {dataset_dir / 'data.yaml'}")
    unknown = set(names) - ROLE_MAP.keys()
    if unknown:
        raise ValueError(f"Nhãn chưa có ánh xạ: {sorted(unknown)}")
    return names


def read_boxes(label_path: Path, image_size: tuple[int, int], class_names: list[str]) -> list[dict]:
    width, height = image_size
    boxes = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_no}: cần 5 giá trị YOLO, nhận {len(parts)}")
        class_id = int(parts[0])
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"{label_path}:{line_no}: class id ngoài phạm vi: {class_id}")
        cx, cy, bw, bh = map(float, parts[1:])
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
            raise ValueError(f"{label_path}:{line_no}: tọa độ YOLO ngoài [0, 1]")
        x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
        x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
        label = class_names[class_id]
        boxes.append({
            "class_id": class_id,
            "label": label,
            "role": ROLE_MAP[label],
            "bbox_xyxy_px": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "bbox_xyxy_norm": [round(cx - bw / 2, 6), round(cy - bh / 2, 6),
                               round(cx + bw / 2, 6), round(cy + bh / 2, 6)],
            "cx_norm": cx,
            "cy_norm": cy,
            "height_norm": bh,
        })
    return boxes


def group_rows(boxes: list[dict]) -> tuple[list[list[dict]], float]:
    """Group phrase boxes by y-centre using the documented 0.5×median-height rule.

    ``report_type`` boxes describe the report/document header, so they are
    returned separately and do not create analyte rows.
    """
    table_boxes = [b for b in boxes if b["role"] != "report_type"]
    if not table_boxes:
        return [], 0.0
    tolerance = 0.5 * statistics.median(b["height_norm"] for b in table_boxes)
    ordered = sorted(table_boxes, key=lambda b: (b["cy_norm"], b["cx_norm"]))
    clusters: list[list[dict]] = []
    for box in ordered:
        if not clusters:
            clusters.append([box])
            continue
        center = statistics.median(item["cy_norm"] for item in clusters[-1])
        if abs(box["cy_norm"] - center) < tolerance:
            clusters[-1].append(box)
        else:
            clusters.append([box])
    clusters.sort(key=lambda row: statistics.median(b["cy_norm"] for b in row))
    return clusters, tolerance


def row_record(row_id: int, boxes: list[dict], image_size: tuple[int, int]) -> dict:
    width, height = image_size
    ordered = sorted(boxes, key=lambda b: b["cx_norm"])
    role_boxes: dict[str, list[dict]] = {role: [] for role in TABLE_ROLE_ORDER}
    for box in ordered:
        role_boxes[box["role"]].append({
            "label": box["label"],
            "class_id": box["class_id"],
            "bbox_xyxy_px": box["bbox_xyxy_px"],
            "bbox_xyxy_norm": box["bbox_xyxy_norm"],
        })
    x1 = min(b["bbox_xyxy_px"][0] for b in ordered)
    y1 = min(b["bbox_xyxy_px"][1] for b in ordered)
    x2 = max(b["bbox_xyxy_px"][2] for b in ordered)
    y2 = max(b["bbox_xyxy_px"][3] for b in ordered)
    return {
        "row_id": row_id,
        "y_center_norm": round(statistics.median(b["cy_norm"] for b in ordered), 6),
        "bbox_xyxy_px": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "bbox_xyxy_norm": [round(x1 / width, 6), round(y1 / height, 6),
                           round(x2 / width, 6), round(y2 / height, 6)],
        "fields_by_role": role_boxes,
        "box_count": len(ordered),
    }


def build_image_record(image_path: Path, class_names: list[str]) -> dict:
    split_dir = image_path.parent.parent
    label_path = split_dir / "labels" / f"{image_path.stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"Thiếu nhãn cho {image_path}: {label_path}")
    with Image.open(image_path) as im:
        image_size = im.size
    boxes = read_boxes(label_path, image_size, class_names)
    rows, tolerance = group_rows(boxes)
    report_type = [
        {"bbox_xyxy_px": b["bbox_xyxy_px"], "bbox_xyxy_norm": b["bbox_xyxy_norm"]}
        for b in boxes if b["role"] == "report_type"
    ]
    return {
        "image": image_path.name,
        "split": split_dir.name,
        "source_group": image_path.name.partition(".rf.")[0],
        "image_size_px": list(image_size),
        "annotation_box_count": len(boxes),
        "empty_label_file": not boxes,
        "report_type_spans": report_type,
        "row_tolerance_norm": round(tolerance, 8),
        "rows": [row_record(i, row, image_size) for i, row in enumerate(rows)],
    }


def select_images(dataset_dir: Path, split: str, sample_size: int, seed: int) -> list[Path]:
    images_dir = dataset_dir / split / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(p for p in images_dir.iterdir() if p.is_file()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        groups[path.name.partition(".rf.")[0]].append(path)
    keys = sorted(groups)
    rng = random.Random(seed)
    if sample_size > 0 and sample_size < len(keys):
        keys = sorted(rng.sample(keys, sample_size))
    return [sorted(groups[key])[0] for key in keys]


def save_review_images(records: list[dict], dataset_dir: Path, output_dir: Path) -> Path:
    """Save one full-resolution annotated image per sample and a Markdown index."""
    role_colors = {
        "report_type": (220, 0, 220), "parameter": (255, 130, 0),
        "range": (0, 170, 255), "unit": (0, 180, 0), "result": (230, 30, 30),
    }
    class_names = load_class_names(dataset_dir)
    image_dir = output_dir / "review_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "# Review ground truth bộ 2",
        "",
        "Mở từng ảnh ở độ phân giải gốc. Box màu biểu thị role; số ở lề trái là row_id do script suy ra.",
        "",
        "| Màu | Role |",
        "|---|---|",
    ]
    for role, color in role_colors.items():
        index_lines.append(f"| RGB{color} | `{role}` |")
    index_lines.extend(["", "| # | Ảnh | Box | Dòng suy ra |", "|---:|---|---:|---:|"])
    header_h, gutter_w = 42, 38
    for i, record in enumerate(records, 1):
        image_path = dataset_dir / record["split"] / "images" / record["image"]
        with Image.open(image_path) as source:
            original = source.convert("RGB")
        original_w, original_h = original.size
        annotated = Image.new("RGB", (original_w + gutter_w, original_h + header_h), "white")
        annotated.paste(original, (gutter_w, header_h))
        draw = ImageDraw.Draw(annotated)
        for role_i, (role, color) in enumerate(role_colors.items()):
            x = gutter_w + 8 + role_i * max(95, (original_w - 16) // len(role_colors))
            draw.rectangle((x, 12, x + 10, 22), fill=color)
            draw.text((x + 14, 10), role, fill=(0, 0, 0))
        label_path = dataset_dir / record["split"] / "labels" / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, cx, cy, bw, bh = line.split()
            role = ROLE_MAP[class_names[int(class_id)]]
            cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
            x1, y1 = (cx - bw / 2) * original_w + gutter_w, (cy - bh / 2) * original_h + header_h
            x2, y2 = (cx + bw / 2) * original_w + gutter_w, (cy + bh / 2) * original_h + header_h
            draw.rectangle((x1, y1, x2, y2), outline=role_colors[role], width=2)
        # Put inferred row numbers in a separate gutter, leaving the source pixels unobscured.
        for row in record["rows"]:
            y = header_h + row["y_center_norm"] * original_h
            draw.text((3, y - 5), str(row["row_id"]), fill=(0, 0, 0), stroke_width=1, stroke_fill=(255, 255, 255))
            x1 = gutter_w + row["bbox_xyxy_px"][0]
            draw.line((gutter_w + 28, y, x1, y), fill=(90, 90, 90), width=1)
        filename = f"{i:02d}_{image_path.stem}_review.png"
        annotated.save(image_dir / filename)
        index_lines.append(
            f"| {i} | [`{record['image']}`](review_images/{filename}) | "
            f"{record['annotation_box_count']} | {len(record['rows'])} |"
        )
    index_path = output_dir / "review_index.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh ground truth dòng/cột phrase-level cho bộ 2")
    parser.add_argument("--dataset", type=Path, default=DATASETS[2])
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Số nhóm ảnh nguồn duy nhất; 0 = tất cả nhóm trong split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "khang_dataset2_ground_truth")
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error("--sample-size phải >= 0")

    dataset_dir = args.dataset.resolve()
    class_names = load_class_names(dataset_dir)
    images = select_images(dataset_dir, args.split, args.sample_size, args.seed)
    records = [build_image_record(path, class_names) for path in images]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": dataset_dir.name,
        "license": "CC BY 4.0",
        "split": args.split,
        "sample_size": len(records),
        "seed": args.seed,
        "label_granularity": "semantic phrase/span",
        "column_ground_truth": "semantic role from source class label",
        "row_rule": "cluster phrase-box y-centres; tolerance is strictly less than 0.5 times median non-report-type box height",
        "report_type_policy": "keep as report_type_spans metadata; exclude from analyte rows",
        "class_mapping": {name: ROLE_MAP[name] for name in class_names},
        "samples": records,
    }
    json_path = args.out_dir / "ground_truth.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=("image", "source_group", "split", "box_count", "row_count", "empty_label"))
        writer.writeheader()
        for record in records:
            writer.writerow({
                "image": record["image"], "source_group": record["source_group"],
                "split": record["split"], "box_count": record["annotation_box_count"],
                "row_count": len(record["rows"]), "empty_label": record["empty_label_file"],
            })
    index_path = save_review_images(records, dataset_dir, args.out_dir)
    print(f"Ảnh nguồn: {len(records)}; dòng suy ra: {sum(len(r['rows']) for r in records)}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"Index ảnh review: {index_path}")
    print(f"Ảnh riêng: {args.out_dir / 'review_images'}")


if __name__ == "__main__":
    main()
