"""Đọc số, khoảng tham chiếu và gắn cờ bất thường (luật).

Phần phụ trách: Mắt xích ③ của pipeline (Khang) — Phiếu xét nghiệm.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple, Union

try:
    from common.contracts import LAB_FLAGS
except ImportError:
    try:
        from ..common.contracts import LAB_FLAGS
    except ImportError:
        LAB_FLAGS = ("LOW", "NORMAL", "HIGH", "UNKNOWN")


def parse_number(text: Union[str, float, int, None]) -> Optional[float]:
    """"14.9" / "14,9" / "1.2e3" → float; không phải số → None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    if not isinstance(text, str):
        return None

    s = text.strip()
    if not s:
        return None

    # Chuẩn hóa dấu phân cách thập phân
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # Dạng Việt/Âu: 1.000,50 -> 1000.50
            s = s.replace(".", "").replace(",", ".")
        else:
            # Dạng Anh/Mỹ: 1,000.50 -> 1000.50
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")

    # Thử parse trực tiếp số hoàn chỉnh
    try:
        return float(s)
    except ValueError:
        pass

    # Thử trích xuất pattern số nếu có ký hiệu phụ (ví dụ "14.9*", "14.9 (H)")
    # Loại trừ trường hợp chuỗi chứa nhiều ký tự chữ cái (như "abc", "neg")
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s)
    if match:
        # Nếu chuỗi không chứa quá nhiều chữ cái định tính
        alpha_count = sum(c.isalpha() for c in s)
        digits_count = sum(c.isdigit() for c in s)
        if digits_count > 0 and alpha_count <= 2:
            try:
                return float(match.group(0))
            except ValueError:
                pass

    return None


def parse_range(text: Union[str, None]) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """"13.5 - 17.5" / "< 5" / "> 40" → (thấp, cao) với None cho phía không giới hạn; không đọc được → None."""
    if text is None or not isinstance(text, str):
        return None

    s = text.strip()
    if not s:
        return None

    # Chuẩn hóa dấu phẩy thập phân giữa các chữ số: "13,5 - 17,5" -> "13.5 - 17.5"
    s_norm = re.sub(r"(\d+),(\d+)", r"\1.\2", s)

    # 1. Trường hợp một phía nhỏ hơn / nhỏ hơn hoặc bằng: "< 5", "<= 5", "≤ 5"
    match_lt = re.search(r"^[<≤]=?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", s_norm)
    if match_lt:
        try:
            high_val = float(match_lt.group(1))
            return (None, high_val)
        except ValueError:
            pass

    # 2. Trường hợp một phía lớn hơn / lớn hơn hoặc bằng: "> 40", ">= 40", "≥ 40"
    match_gt = re.search(r"^[>≥]=?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", s_norm)
    if match_gt:
        try:
            low_val = float(match_gt.group(1))
            return (low_val, None)
        except ValueError:
            pass

    # 3. Trường hợp hai đầu: "13.5 - 17.5", "13.5-17.5", "13.5 ~ 17.5", "13.5 đến 17.5", "13.5 to 17.5"
    # Dùng non-greedy hoặc phân tách rõ ràng dấu gạch ngang/từ nối
    pattern_two_sided = (
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"\s*(?:[-–—~]|đến|to)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    )
    match_two = re.search(pattern_two_sided, s_norm, re.IGNORECASE)
    if match_two:
        try:
            low_val = float(match_two.group(1))
            high_val = float(match_two.group(2))
            return (low_val, high_val)
        except ValueError:
            pass

    return None


def flag(
    result: Union[float, int, str, None],
    ref_range: Union[Tuple[Optional[float], Optional[float]], list, None],
) -> str:
    """So kết quả với khoảng → một giá trị trong common.contracts.LAB_FLAGS.

    Quy tắc:
    * Nếu result is None hoặc ref_range is None → "UNKNOWN".
    * Nếu ref_range có dạng (low, high):
        - low is not None and result < low → "LOW"
        - high is not None and result > high → "HIGH"
        - Còn lại → "NORMAL"
    """
    if result is None or ref_range is None:
        return "UNKNOWN"

    try:
        low = ref_range[0]
        high = ref_range[1]
    except (TypeError, IndexError, KeyError):
        return "UNKNOWN"

    # Nếu cả hai biên đều None
    if low is None and high is None:
        return "UNKNOWN"

    # Nếu result là chuỗi, thử parse số
    if isinstance(result, str):
        val = parse_number(result)
        if val is None:
            return "UNKNOWN"
    else:
        try:
            val = float(result)
        except (ValueError, TypeError):
            return "UNKNOWN"

    # So sánh với biên dưới
    if low is not None:
        try:
            if val < float(low):
                return "LOW"
        except (ValueError, TypeError):
            pass

    # So sánh với biên trên
    if high is not None:
        try:
            if val > float(high):
                return "HIGH"
        except (ValueError, TypeError):
            pass

    return "NORMAL"
