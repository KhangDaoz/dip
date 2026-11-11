"""D2 — ghép vùng chữ đã có nhãn + chữ thành khoản thuốc (plan §5.4).

Dùng cho cả pipeline lẫn sinh items_gt (tools/build_items_gt.py). Đầu vào: list đối tượng có .box (xyxy),
.label, .text — TextBox hoặc VAIPEAnnotation đều được.
"""
from __future__ import annotations

from common.contracts import Item

from .text_utils import norm, parse_date, parse_quantity, strip_diagnose_prefix, strip_drug_prefix


def _cy(b):
    return (b.box[1] + b.box[3]) / 2


def _cx(b):
    return (b.box[0] + b.box[2]) / 2


def _y_overlap(a, b) -> float:
    inter = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
    return max(0.0, inter) / max(1, min(a.box[3] - a.box[1], b.box[3] - b.box[1]))


def _reading_order(boxes: list) -> list:
    """Trên → dưới, trái → phải: box cùng dòng khi tâm y lệch < nửa chiều cao."""
    rows = []
    for b in sorted(boxes, key=_cy):
        h = b.box[3] - b.box[1]
        if rows and abs(_cy(b) - _cy(rows[-1][-1])) < 0.5 * max(h, 1):
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for r in rows for b in sorted(r, key=lambda x: x.box[0])]


def _union(boxes: list) -> list:
    return [min(b.box[0] for b in boxes), min(b.box[1] for b in boxes),
            max(b.box[2] for b in boxes), max(b.box[3] for b in boxes)]


def group_items(boxes: list) -> list:
    """Trả về list[Item] theo thứ tự trên → dưới."""
    drugs = sorted([b for b in boxes if b.label == "drugname"], key=lambda b: b.box[1])
    qtys = [b for b in boxes if b.label == "quantity"]
    usages = [b for b in boxes if b.label == "usage"]
    dates = [b for b in boxes if b.label == "date"]
    used_q = set()
    items = []
    for i, d in enumerate(drugs):
        top = d.box[1]
        if i + 1 < len(drugs):
            bottom = drugs[i + 1].box[1]
        else:
            below = [x.box[1] for x in dates if x.box[1] > d.box[3]]
            bottom = min(below) if below else float("inf")
        # quantity cùng hàng (chồng y nhiều nhất), nằm bên phải tên thuốc
        cands = [(j, _y_overlap(d, q)) for j, q in enumerate(qtys)
                 if j not in used_q and _cx(q) > _cx(d) and _y_overlap(d, q) > 0.3]
        q = None
        if cands:
            j = max(cands, key=lambda c: c[1])[0]
            used_q.add(j)
            q = qtys[j]
        us = _reading_order([u for u in usages if top <= _cy(u) < bottom])
        qty, unit = parse_quantity(q.text) if q is not None else (None, None)
        parts = [d] + ([q] if q is not None else []) + us
        items.append(Item(name=strip_drug_prefix(d.text), name_raw=norm(d.text), quantity=qty, unit=unit,
                          usage=[norm(u.text) for u in us], box=_union(parts)))
    return items


def extract_fields(boxes: list) -> dict:
    """Chẩn đoán (nối các box diagnose theo thứ tự đọc, bỏ tiền tố) và ngày (dd/mm/yyyy)."""
    diag = _reading_order([b for b in boxes if b.label == "diagnose"])
    diagnose = strip_diagnose_prefix(" ".join(norm(b.text) for b in diag)) if diag else None
    date = None
    for b in sorted([b for b in boxes if b.label == "date"], key=lambda b: -b.box[1]):
        date = parse_date(b.text)
        if date:
            break
    return {"diagnose": diagnose or None, "date": date}


def assemble(boxes: list, lexicon=None) -> dict:
    """D2 (+ D3 nếu có lexicon) → {"items", "diagnose", "date"}."""
    items = group_items(boxes)
    if lexicon is not None:
        for it in items:
            it.lexicon_match = lexicon.match(it.name)[0]
    return {"items": items, **extract_fields(boxes)}
