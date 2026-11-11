"""D1 — luật phân lớp trên chữ (plan §9): 20+ chuỗi mẫu, có cả chuỗi OCR sai dấu."""
import pytest

from assemble.classify_rules import classify_boxes, classify_text
from common.contracts import TextBox

CASES = [
    ("1) RENAPRIL 5MG 5mg", "drugname"),
    ("2) Amlodipin 5mg", "drugname"),
    ("10) Metformin 850mg", "drugname"),
    ("3 ) Aspirin 81mg", "drugname"),
    ("SL: 28 Viên", "quantity"),
    ("SL : 10 Viên", "quantity"),
    ("sl: 20 vien", "quantity"),
    ("Sáng 1 Viên", "usage"),
    ("Chiều 1 Viên", "usage"),
    ("Tối 2 Viên", "usage"),
    ("Uống 1 viên sau ăn", "usage"),
    ("Ngày uống 2 lần, mỗi lần 1 viên", "usage"),
    ("Sang 1 Vien", "usage"),
    ("Chẩn đoán: I10 - Tăng huyết áp vô căn", "diagnose"),
    ("Chấn đoán: E11 - Đái tháo đường", "diagnose"),
    ("Chan doan: J20", "diagnose"),
    ("Ngày 03/07/2019", "date"),
    ("Ngày 3 tháng 7 năm 2019", "date"),
    ("Bác sĩ khám bệnh", "other"),
    ("Họ tên:", "other"),
    ("Lời dặn: tái khám", "other"),
    ("Sáng", "other"),          # usage phải có số
]


@pytest.mark.parametrize("text,label", CASES)
def test_classify_text(text, label):
    assert classify_text(text) == label


def test_diagnose_continuation_line():
    boxes = [TextBox([28, 272, 732, 294], text="Chẩn đoán: I10 - Tăng huyết áp"),
             TextBox([28, 296, 500, 318], text="E11 - Đái tháo đường type 2"),
             TextBox([28, 400, 300, 420], text="Ghi chú khác")]
    classify_boxes(boxes)
    assert [b.label for b in boxes] == ["diagnose", "diagnose", "other"]
