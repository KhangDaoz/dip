"""③ Dựng bảng phiếu xét nghiệm (Khang)."""
import cv2
import pytest

from common.contracts import LAB_FLAGS
from common.testing import todo
from table_extraction import flags, geometry, rows


@pytest.mark.parametrize("text, expected", [("14.9", 14.9), ("14,9", 14.9), ("abc", None), ("", None)])
@todo
def test_parse_number(text, expected):
    assert flags.parse_number(text) == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize("text, expected", [
    ("13.5 - 17.5", (13.5, 17.5)), ("13.5-17.5", (13.5, 17.5)), ("< 5", (None, 5.0)), ("> 40", (40.0, None)),
])
@todo
def test_parse_range(text, expected):
    assert flags.parse_range(text) == expected


@pytest.mark.parametrize("result, ref, expected", [
    (12.0, (13.5, 17.5), "LOW"), (15.0, (13.5, 17.5), "NORMAL"), (18.0, (13.5, 17.5), "HIGH"),
    (None, (13.5, 17.5), "UNKNOWN"), (15.0, None, "UNKNOWN"), (3.0, (None, 5.0), "NORMAL"),
])
@todo
def test_flag(result, ref, expected):
    out = flags.flag(result, ref)
    assert out in LAB_FLAGS and out == expected


@todo
def test_char_height_matches_font(text_mask):
    (_, h), _ = cv2.getTextSize("Hg", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    assert 0.4 * h < geometry.char_height(text_mask) < 1.5 * h


@todo
def test_rows_fixed_groups_three_rows():
    words = [[10, 100, 40, 20], [60, 102, 50, 20], [10, 150, 40, 20], [70, 148, 30, 20], [10, 200, 40, 20]]
    assert sorted(map(sorted, rows.rows_fixed(words))) == [[0, 1], [2, 3], [4]]
