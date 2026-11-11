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
    id: int
    text: str
    label: str
    box: list
    mapping: Optional[int] = None


@dataclass
class VAIPEImage:
    image_id: str
    split: str
    image_path: Path
    annotations: list


def clean_text(text) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def read_label_json(json_path: Path) -> list:
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
    if label not in LABEL_TO_ID:
        raise ValueError(f"label phải thuộc {BOX_LABELS}")
    out = []
    for img in images:
        for a in img.annotations:
            if a.label == label:
                out.append((img, a))
    return out


def stats(images: list) -> dict:
    counts = {l: 0 for l in BOX_LABELS}
    n_imgs = len(images)
    for img in images:
        for a in img.annotations:
            counts[a.label] += 1
    return {"n_images": n_imgs, "by_label": counts}
