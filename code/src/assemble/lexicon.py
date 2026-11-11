"""Khối D4 — đưa tên thuốc về tên chuẩn bằng từ điển tên thuốc của train (plan_Thang.md mục 4.4).

Quy tắc (tham số cố định, không chỉnh trên test):
* lấy 2 ứng viên giống nhất theo rapidfuzz.fuzz.ratio, điểm s1 >= s2;
* s1 >= TAU_HIGH → thay; TAU_LOW <= s1 < TAU_HIGH và s1 - s2 >= MARGIN → thay; còn lại giữ tên cũ;
* khóa liều: 2 ứng viên sát nhau (s1 - s2 < MARGIN) mà khác dãy số → chỉ thay khi dãy số trùng với tên OCR;
* chặn liều sạch: vùng TAU_LOW <= s1 < TAU_HIGH, nếu tên OCR có một liều đọc sạch ("250MG", "0,25g") mà ứng viên
  không có → giữ tên OCR. Lý do: thuốc ngoài từ điển "CEPHALEXIN 250MG 0,25g" bị kéo thành "CEPHALEXIN PMP 500 0,5g"
  (điểm 80, sai liều). Chi phí đo trên valid: tên đạt CER ≤ 0,1 ở CER OCR 0,10 giảm 0,973 → 0,965.
"""
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
    """Các liều đọc sạch trong tên: "CEPHALEXIN 250MG 0,25g" → {("250", "mg"), ("0.25", "g")}."""
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
        """→ (tên dùng, ứng viên tốt nhất, điểm, có thay không)."""
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
        if accept and s1 < tau_high and doses(name) - doses(c1):     # chặn liều sạch (vùng TAU_LOW–TAU_HIGH)
            accept = False
        return (c1 if accept else name), c1, s1, accept
