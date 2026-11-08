"""④ Bóc tách đơn thuốc (Thắng)."""
import pytest

from common.testing import todo
from conftest import N_LINES
from prescription_segmentation import items, lines


@pytest.mark.parametrize("fn", ["lines_projection", "lines_components"])
@todo
def test_lines_count(text_mask, fn):
    found = getattr(lines, fn)(text_mask)
    assert len(found) == N_LINES
    ys = [b[1] for b in found]
    assert ys == sorted(ys), "dòng phải sắp từ trên xuống"


@todo
def test_group_items_fixed():
    line_boxes = [[0, 10 * i, 100, 8] for i in range(6)]
    assert len(items.group_items_fixed(line_boxes, lines_per_item=2)) == 3


def test_group_items_rules_by_ordinal():
    line_boxes = [[0, 10 * i, 100, 8] for i in range(6)]
    texts = ["1) Paracetamol 500mg", "Sang 1 vien", "2) Amoxicillin 250mg", "Sang 1 vien", "Toi 1 vien",
             "3. Vitamin C"]
    found = items.group_items_rules(line_boxes, texts)
    assert [b[1] for b in found] == [0, 20, 50]


def _two_line_items(n_items=3, gap_in=4, gap_out=30):
    # mỗi khoản: dòng tên thuốc cao 24 px + dòng liều thụt lề cao 20 px (tỉ lệ như đơn VAIPE)
    line_boxes, y = [], 0
    for _ in range(n_items):
        line_boxes += [[0, y, 200, 24], [30, y + 24 + gap_in, 150, 20]]
        y += 24 + gap_in + 20 + gap_out
    return line_boxes


def test_group_items_rules_geometry():
    found = items.group_items_rules(_two_line_items())
    assert len(found) == 3
    assert found[0] == [0, 0, 200, 48]


def test_group_items_dilation_se_factor():
    line_boxes = _two_line_items(gap_in=4, gap_out=12)
    # SE dọc 0,3 × 22 px ≈ 7 px: dính khoảng trống 4 px, không dính 12 px
    assert len(items.group_items_dilation(line_boxes, se_factor=0.3)) == 3
    # SE quá lớn → mọi dòng dính thành một khối
    assert len(items.group_items_dilation(line_boxes, se_factor=1.0)) == 1


def test_group_items_fixed_gap_and_per_line():
    line_boxes = _two_line_items(gap_in=4, gap_out=12)
    assert len(items.group_items_fixed_gap(line_boxes, gap_px=8)) == 3
    assert len(items.group_items_fixed(line_boxes, lines_per_item=1)) == 6


def test_lines_components_keeps_diacritics_on_line(text_mask):
    import cv2
    mask = text_mask.copy()
    for x in range(60, 400, 40):    # dấu thanh nhỏ phía trên dòng đầu (y ≈ 70)
        cv2.circle(mask, (x, 48), 2, 255, -1)
    assert len(lines.lines_components(mask)) == N_LINES


def test_find_medicine_region(text_mask):
    from prescription_segmentation.region import find_medicine_region
    box = find_medicine_region(text_mask, se_factor=3.0)   # fixture giãn dòng rộng (55 px, chữ ~17 px)
    assert box is not None
    x, y, w, h = box
    assert y < 70 and y + h > 70 + 55 * (N_LINES - 1) - 5
