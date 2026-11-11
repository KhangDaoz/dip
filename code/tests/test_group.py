"""D2 — ghép khoản thuốc (plan §9). Đơn tổng hợp biết trước đáp án + 3 ảnh train thật (skip nếu thiếu data)."""
import pytest

from assemble.group import assemble, group_items
from assemble.text_utils import parse_date, parse_quantity, strip_diagnose_prefix, strip_drug_prefix
from common.contracts import TextBox


def _b(box, label, text):
    return TextBox(box=box, label=label, text=text)


@pytest.fixture
def rx():
    return [
        _b([28, 272, 732, 294], "diagnose", "Chẩn đoán: I10 - Tăng huyết áp"),
        _b([46, 386, 294, 413], "drugname", "1) RENAPRIL 5MG 5mg"),
        _b([595, 387, 708, 409], "quantity", "SL: 28 Viên"),
        _b([289, 416, 376, 436], "usage", "Sáng 1 Viên"),
        _b([400, 416, 480, 436], "usage", "Tối 1 Viên"),
        _b([46, 446, 294, 473], "drugname", "2) Amlodipin 5mg"),
        _b([595, 447, 708, 469], "quantity", "SL: 14 Viên"),
        _b([289, 476, 376, 496], "usage", "Sáng 1 Viên"),
        _b([484, 633, 621, 652], "date", "Ngày 03/07/2019"),
        _b([484, 700, 621, 720], "usage", "Sáng 9 Viên"),   # dưới ngày → không thuộc khoản nào
    ]


def test_group_synthetic(rx):
    items = group_items(rx)
    assert [it.name for it in items] == ["RENAPRIL 5MG 5mg", "Amlodipin 5mg"]
    assert [(it.quantity, it.unit) for it in items] == [(28, "Viên"), (14, "Viên")]
    assert items[0].usage == ["Sáng 1 Viên", "Tối 1 Viên"]
    assert items[1].usage == ["Sáng 1 Viên"]
    assert items[0].box == [46, 386, 708, 436]


def test_fields(rx):
    d = assemble(rx)
    assert d["diagnose"] == "I10 - Tăng huyết áp"
    assert d["date"] == "03/07/2019"


def test_parsers():
    assert strip_drug_prefix("12) Panadol 500mg") == "Panadol 500mg"
    assert parse_quantity("SL: 28 Viên") == (28, "Viên")
    assert parse_quantity("SL:5 Gói") == (5, "Gói")
    assert parse_quantity("không số") == (None, None)
    assert parse_date("Ngày 3 tháng 7 năm 2019") == "03/07/2019"
    assert strip_diagnose_prefix("Chẩn đoán: J20 - Viêm phế quản") == "J20 - Viêm phế quản"


def test_group_real_train_images():
    from vaipe.dataset import list_images
    imgs = list_images("train")
    if not imgs:
        pytest.skip("chưa có data/1_vaipe")
    for img in imgs[:3]:
        n_drug = sum(a.label == "drugname" for a in img.annotations)
        assert len(group_items(img.annotations)) == n_drug
