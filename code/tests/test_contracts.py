import pytest

from common.contracts import TextBox, Item, PrescriptionResult, BOX_LABELS, LABEL_TO_ID


def test_label_to_id():
    assert LABEL_TO_ID["drugname"] == 0
    assert LABEL_TO_ID["other"] == 5
    assert len(BOX_LABELS) == 6


def test_textbox_valid():
    tb = TextBox(box=[10, 20, 30, 40], label="drugname", score=0.9)
    tb.validate()


def test_textbox_invalid_box():
    with pytest.raises(ValueError, match="x1 < x0"):
        TextBox(box=[30, 20, 10, 40], label="drugname", score=0.9).validate()


def test_textbox_invalid_label():
    with pytest.raises(ValueError, match="label phải thuộc"):
        TextBox(box=[10, 20, 30, 40], label="foo", score=0.9).validate()


def test_textbox_label_none_ok():
    TextBox(box=[10, 20, 30, 40], label=None, score=0.5).validate()


def test_item_default_quantity_none():
    it = Item(name="PARACETAMOL", name_raw="1) PARACETAMOL")
    it.validate()
    assert it.quantity is None
    assert it.usage == []


def test_prescription_result_round_trip():
    r = PrescriptionResult(
        image_id="VAIPE_X",
        items=[Item(name="PARACETAMOL", name_raw="1) PARACETAMOL", quantity=10, unit="Viên")],
        diagnose="Cảm lạnh",
        date="2026-11-11",
        boxes=[TextBox(box=[0, 0, 100, 30], label="drugname", score=0.95)],
        timing={"A": 0.05, "B": 0.10},
    )
    d = r.to_dict()
    assert d["image_id"] == "VAIPE_X"
    assert len(d["items"]) == 1
    assert len(d["boxes"]) == 1
    r2 = PrescriptionResult.from_dict(d)
    assert r2.image_id == "VAIPE_X"
    assert r2.items[0].name == "PARACETAMOL"
    assert r2.boxes[0].label == "drugname"
