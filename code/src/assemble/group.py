from __future__ import annotations

import copy
import re
from typing import Optional

from common.boxes import iou
from common.contracts import Item, PrescriptionResult, TextBox

from .rules import (classify_text, clean_diagnose, clean_name, clean_usage, has_index_prefix,
                    parse_date, parse_quantity)

CONF_YOLO = 0.5
DEDUPE_IOU = 0.7


def center_y(b: TextBox) -> float:
    return (b.box[1] + b.box[3]) / 2


def height(b: TextBox) -> float:
    return max(1.0, b.box[3] - b.box[1])


def union_box(boxes: list[TextBox]) -> list[int]:
    return [min(b.box[0] for b in boxes), min(b.box[1] for b in boxes),
            max(b.box[2] for b in boxes), max(b.box[3] for b in boxes)]


def choose_labels(boxes: list[TextBox], has_labels: Optional[bool] = None,
                  conf_thr: float = CONF_YOLO) -> list[TextBox]:
    if has_labels is None:
        has_labels = any(b.label is not None for b in boxes)
    out = []
    for b in boxes:
        b = copy.copy(b)
        if not has_labels or b.label is None:
            b.label = classify_text(b.text)
        elif b.label == "other" or b.score < conf_thr:
            guess = classify_text(b.text)
            if guess != "other":
                b.label = guess
        out.append(b)
    return out


def dedupe(boxes: list[TextBox], iou_thr: float = DEDUPE_IOU) -> list[TextBox]:
    kept: list[TextBox] = []
    for b in sorted(boxes, key=lambda b: -b.score):
        if all(iou(b.box, k.box) <= iou_thr for k in kept):
            kept.append(b)
    return kept


_GLUED_QTY = re.compile(r"^(.*?\S)\s+((?:[Ss][Ll1Ii|]|5[LlIi|])\s*[:.;]\s*\d+.*)$")


def split_glued_quantity(boxes: list[TextBox]) -> list[TextBox]:
    out = []
    for b in boxes:
        m = _GLUED_QTY.match(b.text or "") if b.label == "drugname" else None
        if m and parse_quantity(m.group(2))[0] is not None:
            name_box, qty_box = copy.copy(b), copy.copy(b)
            name_box.text, qty_box.text, qty_box.label = m.group(1), m.group(2), "quantity"
            out += [name_box, qty_box]
        else:
            out.append(b)
    return out


def merge_wrapped_names(boxes: list[TextBox]) -> list[TextBox]:
    drugs = sorted([b for b in boxes if b.label == "drugname"], key=lambda b: b.box[1])
    others = [b for b in boxes if b.label != "drugname"]
    merged: list[TextBox] = []
    for d in drugs:
        prev = merged[-1] if merged else None
        if (prev is not None and not has_index_prefix(d.text)
                and -0.5 * height(prev) <= d.box[1] - prev.box[3] < 1.5 * height(prev)
                and abs(d.box[0] - prev.box[0]) < 2 * height(prev)):
            prev.text = f"{prev.text} {d.text}"
            prev.box = union_box([prev, d])
            continue
        merged.append(copy.copy(d))
    return merged + others


def reading_order(boxes: list[TextBox]) -> list[TextBox]:
    rows: list[list[TextBox]] = []
    for b in sorted(boxes, key=center_y):
        if rows and abs(center_y(b) - center_y(rows[-1][0])) < 0.5 * height(b):
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: b.box[0])]


def split_bands(boxes: list[TextBox]) -> list[dict]:
    drugs = sorted([b for b in boxes if b.label == "drugname"], key=lambda b: b.box[1])
    if not drugs:
        return []
    last_top = drugs[-1].box[1]
    date_tops = [b.box[1] for b in boxes if b.label == "date" and b.box[1] > last_top]
    tops = [d.box[1] for d in drugs] + [min(date_tops) if date_tops else float("inf")]
    bands = []
    for i, drug in enumerate(drugs):
        inside = [b for b in boxes if tops[i] <= center_y(b) < tops[i + 1]]
        qs = sorted([b for b in inside if b.label == "quantity"], key=center_y)
        bands.append({"drug": drug,
                      "quantity": qs[0] if qs else None,
                      "usage": reading_order([b for b in inside if b.label == "usage"])})
    return bands


def merge_diagnose(boxes: list[TextBox], first_drug_y: float) -> Optional[str]:
    diag = [b for b in boxes if b.label == "diagnose" and center_y(b) < first_drug_y]
    if not diag:
        return None
    text = " ".join(b.text for b in reading_order(diag))
    return clean_diagnose(text) or None


def first_date(boxes: list[TextBox]) -> Optional[str]:
    for b in sorted([b for b in boxes if b.label == "date"], key=center_y):
        d = parse_date(b.text)
        if d:
            return d
    return None


def assemble(boxes: list[TextBox], lexicon=None, has_labels: Optional[bool] = None,
             conf_thr: float = CONF_YOLO) -> tuple[list[Item], Optional[str], Optional[str]]:
    boxes = choose_labels(boxes, has_labels=has_labels, conf_thr=conf_thr)
    boxes = dedupe(boxes)
    boxes = split_glued_quantity(boxes)
    boxes = merge_wrapped_names(boxes)

    items = []
    for band in split_bands(boxes):
        drug, q = band["drug"], band["quantity"]
        qty, unit = parse_quantity(q.text) if q else (None, None)
        name = clean_name(drug.text)
        match = None
        if lexicon is not None:
            _, best, _, accepted = lexicon.match(name)
            match = best if accepted else None
        parts = [drug] + ([q] if q else []) + band["usage"]
        items.append(Item(name=name, name_raw=drug.text, quantity=qty, unit=unit,
                          usage=[clean_usage(u.text) for u in band["usage"]],
                          lexicon_match=match, box=union_box(parts)))

    drugs = [b for b in boxes if b.label == "drugname"]
    first_y = min((d.box[1] for d in drugs), default=float("inf"))
    return items, merge_diagnose(boxes, first_y), first_date(boxes)


def assemble_result(image_id: str, boxes: list[TextBox], lexicon=None, **kw) -> PrescriptionResult:
    items, diagnose, date = assemble(boxes, lexicon=lexicon, **kw)
    return PrescriptionResult(image_id=image_id, items=items, diagnose=diagnose, date=date, boxes=boxes)
