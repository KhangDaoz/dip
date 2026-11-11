from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

BOX_LABELS = ("drugname", "quantity", "usage", "diagnose", "date", "other")
LABEL_TO_ID = {l: i for i, l in enumerate(BOX_LABELS)}


def _check_box(box, name="box"):
    if len(box) != 4:
        raise ValueError(f"{name} phải có 4 giá trị [x0, y0, x1, y1], nhận được {box}")
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError(f"{name}: x1 < x0 hoặc y1 < y0, nhận được {box}")


class _Contract:
    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self):
        return self


@dataclass
class TextBox(_Contract):
    box: list
    label: Optional[str] = None
    score: float = 0.0
    text: str = ""
    text_conf: float = 0.0

    def validate(self):
        _check_box(self.box)
        if self.label is not None and self.label not in BOX_LABELS:
            raise ValueError(f"label phải thuộc {BOX_LABELS} hoặc None")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score phải trong [0,1], nhận {self.score}")
        if not 0.0 <= self.text_conf <= 1.0:
            raise ValueError(f"text_conf phải trong [0,1], nhận {self.text_conf}")
        return self


@dataclass
class Item(_Contract):
    name: str
    name_raw: str
    quantity: Optional[int] = None
    unit: Optional[str] = None
    usage: list = field(default_factory=list)
    lexicon_match: Optional[str] = None
    box: list = field(default_factory=list)

    def validate(self):
        if self.box:
            _check_box(self.box, "box khoản")
        if self.quantity is not None and self.quantity < 0:
            raise ValueError(f"quantity phải >= 0, nhận {self.quantity}")
        return self


@dataclass
class PrescriptionResult(_Contract):
    image_id: str
    items: list = field(default_factory=list)
    diagnose: Optional[str] = None
    date: Optional[str] = None
    boxes: list = field(default_factory=list)
    timing: dict = field(default_factory=dict)

    def validate(self):
        for it in self.items:
            it.validate()
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "PrescriptionResult":
        items = [Item(**it) for it in d.get("items", [])]
        boxes = [TextBox(**b) for b in d.get("boxes", [])]
        return cls(image_id=d["image_id"], items=items, diagnose=d.get("diagnose"),
                   date=d.get("date"), boxes=boxes, timing=d.get("timing", {})).validate()
