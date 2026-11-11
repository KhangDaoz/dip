"""Đọc nhãn VAIPE — bộ đọc chung cho cả nhóm (plan §5).

Nhãn gốc là JSON: list[{"id", "text", "label", "box": [x0,y0,x1,y1], "mapping"?}].
Box xyxy pixel trên ảnh gốc. 6 lớp theo BOX_LABELS trong contracts.

Dùng cho:
* Khối B (YOLO): to_yolo.py chuyển sang .txt
* Khối C (OCR): to_vietocr.py cắt crops
* Khối D (luật): build_items_gt.py gom khoản
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from common.contracts import BOX_LABELS, LABEL_TO_ID, TextBox
from common.paths import vaipe_images, vaipe_labels


@dataclass
class VAIPEAnnotation:
    """Một vùng chữ VAIPE."""
    id: int
    text: str
    label: str           # một trong BOX_LABELS
    box: list            # [x0, y0, x1, y1] pixel
    mapping: Optional[int] = None    # mã thuốc nội bộ VAIPE (chỉ có cho drugname)


@dataclass
class VAIPEImage:
    """Một ảnh + nhãn."""
    image_id: str        # "VAIPE_P_TRAIN_0"
    split: str
    image_path: Path
    annotations: list    # list[VAIPEAnnotation]


def clean_text(text) -> str:
    """Unicode NFC + gộp khoảng trắng — cùng chuẩn hóa với độ đo CER và lúc fine-tune VietOCR."""
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def read_label_json(json_path: Path) -> list:
    """Đọc một file nhãn JSON → list[VAIPEAnnotation]. Bỏ qua vùng có label lạ."""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    out = []
    for r in raw:
        label = r.get("label")
        if label not in LABEL_TO_ID:
            continue
        box = r["box"]
        if len(box) != 4 or box[2] < box[0] or box[3] < box[1]:
            continue
        out.append(VAIPEAnnotation(
            id=int(r["id"]), text=r.get("text", ""), label=label,
            box=[int(v) for v in box], mapping=r.get("mapping"),
        ))
    return out


def list_images(split: str) -> list:
    """Liệt kê ảnh trong split, kèm nhãn nếu có. Trả về list[VAIPEImage]."""
    img_dir = vaipe_images(split)
    lbl_dir = vaipe_labels(split)
    if not img_dir.exists():
        return []
    out = []
    for img_path in sorted(img_dir.glob("*.png")):
        image_id = img_path.stem
        json_path = lbl_dir / f"{image_id}.json"
        anns = read_label_json(json_path) if json_path.exists() else []
        out.append(VAIPEImage(image_id=image_id, split=split,
                              image_path=img_path, annotations=anns))
    return out


def filter_by_label(images: list, label: str) -> list:
    """Lấy các annotation thuộc một lớp. Trả về list[(VAIPEImage, VAIPEAnnotation)]."""
    if label not in LABEL_TO_ID:
        raise ValueError(f"label phải thuộc {BOX_LABELS}")
    out = []
    for img in images:
        for a in img.annotations:
            if a.label == label:
                out.append((img, a))
    return out


def stats(images: list) -> dict:
    """Thống kê số vùng theo lớp. Dùng cho in log."""
    counts = {l: 0 for l in BOX_LABELS}
    n_imgs = len(images)
    for img in images:
        for a in img.annotations:
            counts[a.label] += 1
    return {"n_images": n_imgs, "by_label": counts}
