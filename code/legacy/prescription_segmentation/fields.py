"""S-B5 tách trường bằng regex — hậu xử lý văn bản, KHÔNG tính là đóng góp DIP (KeHoach.md mục 1.3)."""
import re
import unicodedata

_DOSE_WORDS = {"morning": r"s[aá]ng", "noon": r"tr[uư]a", "evening": r"(?:chi[eề]u|t[oố]i)"}


def _num(s):
    s = s.replace(",", ".")
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_item_fields(text: str) -> dict:
    """Văn bản OCR một khoản → {"name", "strength", "dose": {"morning", "noon", "evening"}, "days", "note"}."""
    t = unicodedata.normalize("NFC", text or "")
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    first = lines[0] if lines else ""
    name = re.sub(r"^\s*\d+\s*[).:-]\s*", "", first)                  # bỏ số thứ tự "1)" ở đầu
    name = re.split(r"\bSL\b|S[ốo] l[ưu]ợng", name, flags=re.I)[0].strip()
    m = re.search(r"(\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|µg|iu|%)(?:\s*[+/]\s*\d+(?:[.,]\d+)?\s*(?:mg|g|ml)?)*)", t, re.I)
    strength = m.group(1).replace(" ", "") if m else None
    low = t.lower()
    dose = {}
    for k, pat in _DOSE_WORDS.items():
        dm = re.search(pat + r"\s*:?\s*(\d+(?:[.,/]\d+)?)", low)
        dose[k] = _num(dm.group(1)) if dm else 0
        dose[k] = dose[k] or 0
    dm = re.search(r"\b(\d+(?:[.,/]\d+)?)\s*-\s*(\d+(?:[.,/]\d+)?)\s*-\s*(\d+(?:[.,/]\d+)?)\b", low)
    if dm and not any(dose.values()):
        dose = {"morning": _num(dm.group(1)) or 0, "noon": _num(dm.group(2)) or 0, "evening": _num(dm.group(3)) or 0}
    days_m = re.search(r"(\d+)\s*(?:ng[aà]y|days?)", low)
    note_m = re.search(r"(tr[uư][oớ]c ăn|sau ăn|khi đói|after food|before food)", low)
    return {"name": name, "strength": strength, "dose": dose,
            "days": int(days_m.group(1)) if days_m else None, "note": note_m.group(1) if note_m else None}
