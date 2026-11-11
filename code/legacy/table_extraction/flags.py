"""K7 vai trò cột + cờ bất thường — hậu xử lý văn bản, KHÔNG tính là đóng góp DIP (KeHoach.md mục 1.3, 7.1)."""
import re

_NUM = r"[-+]?\d+(?:[.,]\d+)?"
UNIT_WORDS = ("g/dl", "g/l", "mg/dl", "mmol", "%", "fl", "pg", "/ul", "/µl", "/mm", "cells", "lakh", "10^",
              "x10", "u/l", "iu", "mill", "thou", "cumm", "/l")


def parse_number(text: str):
    """"14,9" / "14.9" → 14.9; không đọc được → None."""
    if not text:
        return None
    m = re.search(_NUM, text.replace(" ", ""))
    if not m:
        return None
    s = m.group(0)
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_range(text: str):
    """"13.5 - 17.5" → [13.5, 17.5]; "< 5" → [None→0, 5]; không đọc được → None."""
    if not text:
        return None
    t = text.replace("–", "-").replace("—", "-").replace("đến", "-").replace("~", "-").replace("to", "-")
    nums = re.findall(_NUM, t.replace(" ", "").replace("--", "-"))
    vals = []
    for s in nums:
        s = s.lstrip("+")
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        try:
            vals.append(abs(float(s.replace(",", ""))))
        except ValueError:
            pass
    if len(vals) >= 2:
        lo, hi = sorted(vals[:2])
        return [lo, hi]
    if len(vals) == 1:
        if "<" in t:
            return [0.0, vals[0]]
        if ">" in t:
            return [vals[0], float("inf")]
    return None


def flag(result, value_range) -> str:
    """So kết quả với khoảng tham chiếu → "LOW" / "NORMAL" / "HIGH" / "UNKNOWN"."""
    if result is None or not value_range or value_range[0] is None:
        return "UNKNOWN"
    lo, hi = value_range
    if result < lo:
        return "LOW"
    if result > hi:
        return "HIGH"
    return "NORMAL"


def _is_range(t):
    return parse_range(t) is not None and len(re.findall(_NUM, t)) >= 2


def column_roles(cells: list, texts: dict) -> dict:
    """Luật thứ tự cột + kiểu chữ (số, đơn vị) → {col: "parameter" | "result" | "unit" | "range"}.

    texts: {(row, col): chuỗi OCR}. Cột trái nhất chủ yếu chữ = parameter; cột chủ yếu "a - b" = range;
    cột chủ yếu số đơn = result; cột chứa đơn vị = unit.
    """
    by_col = {}
    for c in cells:
        by_col.setdefault(c["col"], []).append(texts.get((c["row"], c["col"]), ""))
    stats = {}
    for col, ts in by_col.items():
        n = max(len(ts), 1)
        stats[col] = {"range": sum(_is_range(t) for t in ts) / n,
                      "num": sum(parse_number(t) is not None and not _is_range(t) for t in ts) / n,
                      "unit": sum(any(u in t.lower() for u in UNIT_WORDS) for t in ts) / n}
    roles, free = {}, sorted(by_col)
    if not free:
        return roles
    roles[free[0]] = "parameter"
    rest = free[1:]
    for role in ("range", "result", "unit"):
        key = {"range": "range", "result": "num", "unit": "unit"}[role]
        cand = [c for c in rest if c not in roles]
        if cand:
            best = max(cand, key=lambda c: stats[c][key])
            if stats[best][key] > 0 or role == "result":
                roles[best] = role
    for c in rest:
        roles.setdefault(c, "unit")
    return roles
