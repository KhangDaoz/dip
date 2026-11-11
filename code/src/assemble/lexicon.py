from __future__ import annotations

import re
from typing import Optional

from common.paths import vaipe_labels

from .rules import clean_name

TAU_HIGH, TAU_LOW, MARGIN = 90.0, 80.0, 5.0


def _digits(s: str) -> list[str]:
    return re.findall(r"\d+", s)


_DOSE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*(mg|mcg|g|ml|ui|iu|%)(?![a-z])", re.IGNORECASE)


def doses(s: str) -> set[tuple[str, str]]:
    return {(n.replace(",", "."), u.lower()) for n, u in _DOSE.findall(s or "")}


class Lexicon:
    def __init__(self, names):
        self.names = sorted({n for n in names if n})

    @classmethod
    def from_split(cls, split: str = "train") -> "Lexicon":
        from vaipe.dataset import read_label_json
        names = []
        for p in sorted(vaipe_labels(split).glob("*.json")):
            names += [clean_name(a.text) for a in read_label_json(p) if a.label == "drugname"]
        if not names:
            raise FileNotFoundError(f"Không có tên thuốc nào trong {vaipe_labels(split)}")
        return cls(names)

    def __len__(self):
        return len(self.names)

    def __contains__(self, name):
        return name in self.names

    def match(self, name: str, tau_high: float = TAU_HIGH, tau_low: float = TAU_LOW,
              margin: float = MARGIN) -> tuple[str, Optional[str], float, bool]:
        try:
            from rapidfuzz import fuzz, process
        except ImportError as e:
            raise ImportError("Cần rapidfuzz: pip install rapidfuzz") from e
        if not name or not self.names:
            return name, None, 0.0, False
        top = process.extract(name, self.names, scorer=fuzz.ratio, limit=2)
        c1, s1 = top[0][0], float(top[0][1])
        c2, s2 = (top[1][0], float(top[1][1])) if len(top) > 1 else (None, 0.0)
        accept = s1 >= tau_high or (s1 >= tau_low and s1 - s2 >= margin)
        if accept and c2 is not None and s1 - s2 < margin and _digits(c1) != _digits(c2):
            accept = _digits(c1) == _digits(name)
        if accept and s1 < tau_high and doses(name) - doses(c1):
            accept = False
        return (c1 if accept else name), c1, s1, accept
