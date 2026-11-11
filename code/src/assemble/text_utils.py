"""Tiện ích chuỗi dùng chung khối D."""
from __future__ import annotations

import re
import unicodedata


def norm(s: str) -> str:
    """NFC + gộp khoảng trắng."""
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt (đ → d) và về chữ thường — để luật regex chịu được OCR sai dấu."""
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D").lower()


DRUG_PREFIX = re.compile(r"^\s*\d+\s*[\).:]\s*")


def strip_drug_prefix(s: str) -> str:
    """'1) RENAPRIL 5MG' → 'RENAPRIL 5MG'."""
    return norm(DRUG_PREFIX.sub("", s or ""))


_QTY = re.compile(r"(?:sl|so luong)\s*[:.;]?\s*(\d+)", re.I)
_NUM = re.compile(r"(\d+)")


def parse_quantity(s: str) -> tuple:
    """'SL: 28 Viên' → (28, 'Viên'). Không đọc được số → (None, None)."""
    raw = norm(s)
    m = _QTY.search(strip_accents(raw)) or _NUM.search(raw)
    if m is None:
        return None, None
    num = m.group(1)
    u = re.search(r"(?<!\d)" + num + r"\s*([^\d\s]+)", raw)
    return int(num), (u.group(1) if u else None)


_DATE_SLASH = re.compile(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})")
_DATE_WORDS = re.compile(r"ngay\s*(\d{1,2})\s*thang\s*(\d{1,2})\s*nam\s*(\d{4})")


def parse_date(s: str):
    """'Ngày 03/07/2019' hoặc 'Ngày 03 tháng 07 năm 2019' → '03/07/2019'. Không đọc được → None."""
    t = strip_accents(s)
    m = _DATE_SLASH.search(t) or _DATE_WORDS.search(t)
    if not m:
        return None
    d, mo, y = m.groups()
    return f"{int(d):02d}/{int(mo):02d}/{y}"


_DIAG_PREFIX = re.compile(r"^\s*ch[a-z]*n\s*do[a-z]*n\s*[:.]?\s*")


def strip_diagnose_prefix(s: str) -> str:
    """'Chẩn đoán: I10 - Tăng huyết áp' → 'I10 - Tăng huyết áp'."""
    raw = norm(s)
    m = _DIAG_PREFIX.match(strip_accents(raw))
    return raw[m.end():].strip() if m else raw
