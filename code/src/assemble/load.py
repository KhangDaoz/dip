"""Đọc file bàn giao của khối B+C (Khang) → dict[image_id, list[TextBox]] (plan_Thang.md mục 2.2).

Định dạng chưa chốt nên đọc linh hoạt:
* mảng JSON, một object JSON, hoặc JSONL; chịu dấu "," thừa ở cuối;
* tên trường kiểu Khang (image, class, conf, xyxy, ocr) hoặc kiểu plan_v2 §4.5 (image_id, label, score, box, text).

Trường "gt" KHÔNG được dùng khi dự đoán (G4). Chỉ đọc khi use_gt_text=True (dòng kiểm "text chuẩn" của E-D2).
"""
from __future__ import annotations

import json
from pathlib import Path

from common.contracts import BOX_LABELS, TextBox


def _parse_records(raw: str) -> list[dict]:
    raw = raw.strip().lstrip("﻿")
    candidates = [raw, "[" + raw.rstrip().rstrip(",") + "]"]
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, list) else [data]
    out = []                                  # JSONL
    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if line:
            out.append(json.loads(line))
    return out


def _to_box(rec: dict, use_gt_text: bool) -> TextBox | None:
    xy = rec.get("xyxy", rec.get("box"))
    if not xy or len(xy) != 4:
        return None
    x0, y0, x1, y1 = (int(round(float(v))) for v in xy)
    if x1 < x0 or y1 < y0:
        raise ValueError(f"box phải là [x0, y0, x1, y1] với x1 >= x0, y1 >= y0: {xy}")
    label = rec.get("class", rec.get("label"))
    if label not in BOX_LABELS:
        label = None
    score = float(rec.get("conf", rec.get("score", 1.0)) or 0.0)
    text = rec.get("gt", "") if use_gt_text else rec.get("ocr", rec.get("text", ""))
    text_conf = float(rec.get("ocr_conf", rec.get("text_conf", 0.0)) or 0.0)
    return TextBox(box=[x0, y0, x1, y1], label=label, score=min(max(score, 0.0), 1.0),
                   text=text or "", text_conf=min(max(text_conf, 0.0), 1.0))


def load_khang(path, use_gt_text: bool = False) -> dict[str, list[TextBox]]:
    """Đọc file bàn giao. use_gt_text=True chỉ dùng cho dòng kiểm "text chuẩn"."""
    out: dict[str, list[TextBox]] = {}
    for rec in _parse_records(Path(path).read_text(encoding="utf-8")):
        image_id = Path(str(rec.get("image_id") or rec.get("image"))).stem
        boxes = [b for b in (_to_box(r, use_gt_text) for r in rec.get("boxes", [])) if b is not None]
        out[image_id] = boxes
    return out
