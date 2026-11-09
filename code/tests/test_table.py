"""③ Dựng bảng phiếu xét nghiệm."""
from common.testing import todo
from table_extraction import flags, geometry, rows


@todo
def test_char_height_reasonable(text_mask):
    assert 10 <= geometry.char_height(text_mask) <= 30


@todo
def test_rows_projection_count(text_mask):
    h = geometry.char_height(text_mask)
    words = geometry.words_from_binary(text_mask, h)
    assert len(rows.rows_projection(text_mask, words, h)) == 12


@todo
def test_flag_rules():
    assert flags.flag(18.0, [13.5, 17.5]) == "HIGH"
    assert flags.flag(14.9, [13.5, 17.5]) == "NORMAL"
    assert flags.flag(None, [13.5, 17.5]) == "UNKNOWN"
