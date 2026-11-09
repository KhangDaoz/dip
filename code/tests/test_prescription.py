"""④ Bóc tách đơn thuốc."""
from common.testing import todo
from prescription_segmentation import lines


@todo
def test_lines_projection_count(text_mask):
    h = lines.char_height(text_mask)
    assert len(lines.lines_projection(text_mask, h)) == 12
