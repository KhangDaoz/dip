"""D3 — so khớp tên thuốc với từ điển tên trong train (rapidfuzz, plan §5.4)."""
from __future__ import annotations

from pathlib import Path

from common.io import load_json, save_json
from common.paths import CODE_DIR

from .text_utils import norm, strip_drug_prefix

LEXICON_PATH = CODE_DIR / "models" / "lexicon_vaipe.json"


class Lexicon:
    def __init__(self, names, tau: float = 90.0):
        self.names = sorted({norm(n) for n in names if norm(n)})
        self._upper = [n.upper() for n in self.names]
        self.tau = tau

    def __len__(self):
        return len(self.names)

    def __contains__(self, name):
        return norm(name).upper() in set(self._upper)

    def match(self, name: str, tau: float | None = None) -> tuple:
        """Trả về (tên chuẩn hoặc None, điểm 0–100). Không phân biệt hoa/thường."""
        from rapidfuzz import fuzz, process
        if not name or not self.names:
            return None, 0.0
        best = process.extractOne(norm(name).upper(), self._upper, scorer=fuzz.ratio)
        if best is None:
            return None, 0.0
        _, score, idx = best
        return (self.names[idx] if score >= (self.tau if tau is None else tau) else None), float(score)

    def oov_rate(self, names) -> float:
        """Tỉ lệ tên KHÔNG có nguyên văn trong từ điển (báo cáo cùng D3)."""
        names = [n for n in names if n]
        if not names:
            return 0.0
        vocab = set(self._upper)
        return sum(norm(n).upper() not in vocab for n in names) / len(names)

    @classmethod
    def from_split(cls, split: str = "train", **kw) -> "Lexicon":
        from vaipe.dataset import list_images
        names = [strip_drug_prefix(a.text) for img in list_images(split) for a in img.annotations
                 if a.label == "drugname"]
        return cls(names, **kw)

    def save(self, path: Path = LEXICON_PATH) -> Path:
        return save_json(path, {"tau": self.tau, "names": self.names})

    @classmethod
    def load(cls, path: Path = LEXICON_PATH, **kw) -> "Lexicon":
        d = load_json(path)
        return cls(d["names"], **{"tau": d.get("tau", 90.0), **kw})


def default_lexicon(**kw):
    """Từ điển mặc định: đọc file đã lưu; chưa có thì dựng từ train và lưu lại. Không có data → None."""
    if LEXICON_PATH.exists():
        return Lexicon.load(**kw)
    lex = Lexicon.from_split("train", **kw)
    if len(lex):
        lex.save()
        return lex
    return None
