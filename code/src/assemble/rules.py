"""Khối D — luật trên chuỗi: nhận diện lớp, làm sạch tên, đọc số lượng, cách dùng, ngày (plan_Thang.md mục 4.1, 4.3).

Mọi hàm ở đây chỉ làm việc trên chuỗi, không cần tọa độ.
Nguyên tắc: chỉ sửa lỗi OCR mà luật chắc chắn sửa đúng; không sửa được thì giữ nguyên hoặc trả None.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Tiền tố số thứ tự thuốc "1)", chịu lỗi OCR ")"→"]", "}", "l", "."
INDEX_PREFIX = re.compile(r"^\s*\d{1,2}\s*[)\]}l.]\s*")
# Tiền tố giả mà OCR hay bịa trước box không phải thuốc ("2) Chản đoán...", "1) Ngảy ...")
FAKE_PREFIX = re.compile(r"^\s*\d{1,2}\s*[)\]]\s*")

# Bảng nhận diện lớp bằng text, kiểm theo thứ tự (date, diagnose trước drugname để tiền tố giả không lừa được).
_RULES = (
    ("date", re.compile(r"^(\d{1,2}\s*[)\]]\s*)?ngay\s*\d")),
    ("diagnose", re.compile(r"^(\d{1,2}\s*[)\]]\s*)?ch[a-z]{0,2}n\s*d")),
    # "SL" chịu lỗi OCR: S→5 hoặc L→1/I/|, nhưng KHÔNG cả hai cùng lúc ("51..." là số, ví dụ mã BN, địa chỉ)
    ("quantity", re.compile(r"^(s[l1i|]|5[li|])\s*[:.;]?\s*\d")),
    ("drugname", re.compile(r"^\d{1,2}\s*[)\]}l.]\s*\S")),
    ("usage", re.compile(r"^(sang|trua|chieu|toi|uong)")),
)

_QTY = re.compile(r"^(?:s[l1i|]|5[li|])\s*[:.;]?\s*([0-9oli]+)\s*(\S+)?")
_DIGIT_LIKE = str.maketrans({"o": "0", "O": "0", "l": "1", "I": "1", "i": "1", "g": "9"})

# Từ chung trong cách dùng: dạng bỏ dấu → dạng chuẩn
_USAGE_WORDS = {"sang": "Sáng", "trua": "Trưa", "chieu": "Chiều", "toi": "Tối", "vien": "Viên", "uong": "Uống"}


def strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt, đổi đ→d, viết thường, bỏ khoảng trắng hai đầu."""
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D").lower().strip()


def collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def classify_text(text: str) -> str:
    """Đoán lớp của một vùng chữ chỉ từ text (mục 4.1). Không khớp mẫu nào → "other"."""
    t = strip_accents(text)
    for label, rx in _RULES:
        if not rx.match(t):
            continue
        if label == "drugname" and (len(t) <= 6 or re.match(r"^\d+\s*/", t)):
            continue   # "1/1" (số trang) hoặc chuỗi quá ngắn
        if label == "usage" and not re.search(r"\d", t):
            continue
        return label
    return "other"


def clean_name(text: str) -> str:
    """Tên thuốc: NFC, bỏ tiền tố "1)", gộp khoảng trắng. KHÔNG đổi hoa thường."""
    t = unicodedata.normalize("NFC", text or "")
    return collapse_spaces(INDEX_PREFIX.sub("", t))


def has_index_prefix(text: str) -> bool:
    return bool(INDEX_PREFIX.match(strip_accents(text)))


def _fix_digit_cluster(cluster: str) -> Optional[str]:
    """Đổi O/l/I/g→số trong một cụm giống số. Cụm phải có ít nhất 1 chữ số thật."""
    if not any(c.isdigit() for c in cluster):
        return None
    return cluster.translate(_DIGIT_LIKE)


def normalize_unit(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    t = strip_accents(token).strip(".,;:")
    if t.startswith("vien"):
        return "Viên"
    return token.strip(".,;:") or None


def parse_quantity(text: str) -> tuple[Optional[int], Optional[str]]:
    """"SL: 20 Viên" → (20, "Viên"). Chịu "SL.", "SL;", "5L:", "2O"→20. Không đọc được → (None, None)."""
    m = _QTY.match(strip_accents(text))
    if not m:
        return None, None
    num = _fix_digit_cluster(m.group(1))
    if num is None or not num.isdigit():
        return None, None
    return int(num), normalize_unit(m.group(2))


def clean_usage(text: str) -> str:
    """"Sảng 1 Viền" → "Sáng 1 Viên". Chỉ sửa các từ chung trong _USAGE_WORDS, giữ nguyên phần còn lại."""
    out = []
    for tok in collapse_spaces(unicodedata.normalize("NFC", text or "")).split(" "):
        m = re.match(r"^(\w+)(\W*)$", tok)
        if m and strip_accents(m.group(1)) in _USAGE_WORDS:
            tok = _USAGE_WORDS[strip_accents(m.group(1))] + m.group(2)
        out.append(tok)
    return " ".join(out)


def strip_fake_prefix(text: str) -> str:
    return FAKE_PREFIX.sub("", text or "")


_DIAG_PREFIX = re.compile(r"^\s*ch\w{0,3}n\s*\S{1,6}\s*:\s*", re.IGNORECASE)


def clean_diagnose(text: str) -> str:
    """Bỏ tiền tố giả "N)" và nhãn "Chẩn đoán:" (chỉ bỏ khi có dấu ":", để không cắt nhầm chữ rác)."""
    t = collapse_spaces(unicodedata.normalize("NFC", strip_fake_prefix(text)))
    return _DIAG_PREFIX.sub("", t).strip()


_DATE_NUM = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_DATE_WORD = re.compile(r"(\d{1,2})\s*thang\s*(\d{1,2})\s*nam\s*(\d{4})")


def _valid_date(d: int, m: int, y: int) -> Optional[str]:
    if 1 <= d <= 31 and 1 <= m <= 12 and 2000 <= y <= 2030:
        return f"{d:02d}/{m:02d}/{y}"
    return None


def parse_date(text: str) -> Optional[str]:
    """"Ngày 03/07/2019" hoặc "Ngày 03 tháng 07 năm 2019" (hoặc "03/07/2019" từ VLM) → "03/07/2019".

    Sửa O/l/I/g→số CHỈ trong cụm giống số (không đổi "ngay" thành "n9ay"). Ngày/tháng không hợp lệ → None.
    """
    t = strip_accents(strip_fake_prefix(text))
    t = re.sub(r"\s*[/.\-]\s*", "/", t)
    t = re.sub(r"[0-9olig]+", lambda m: _fix_digit_cluster(m.group()) or m.group(), t)
    for rx in (_DATE_NUM, _DATE_WORD):
        m = rx.search(t)
        if m:
            return _valid_date(*map(int, m.groups()))
    return None
