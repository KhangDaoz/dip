"""Test khối D (plan_Thang.md mục 8.3). Không cần dữ liệu VAIPE: dùng file mẫu của Khang + tọa độ ghi tay."""
import copy
import json
from pathlib import Path

import pytest

from assemble.group import assemble, choose_labels, dedupe, merge_wrapped_names, reading_order, split_bands
from assemble.load import load_khang
from assemble.rules import classify_text, clean_name, clean_usage, parse_date, parse_quantity
from assemble.vlm import parse_vlm_json, vlm_to_items
from common.contracts import TextBox
from metrics.items import bootstrap_ci, items_prf, match_items

SAMPLE = Path(__file__).parent / "data" / "khang_sample.json"


def tb(box, label, text, score=1.0):
    return TextBox(box=box, label=label, score=score, text=text)


# ---------- D1: nhận diện lớp bằng text ----------
@pytest.mark.parametrize("text,label", [
    ("SL: 20 Viên", "quantity"), ("SL. 60 ViÊn", "quantity"), ("SL; 30 ViÊn", "quantity"),
    ("5L: 20 Viên", "quantity"), ("Sl 10 viên", "quantity"),
    ("1) GLUCOFAST 850 850mg", "drugname"), ("1] PARACETAMOL 500mg", "drugname"), ("2. AMLODIPIN 5mg", "drugname"),
    ("Chẩn đoán: J02 - Viêm họng", "diagnose"), ("Chấn đoán: T10", "diagnose"), ("Chần đoán: I10", "diagnose"),
    ("2) Chản dưỡng Khôngphưỡc", "diagnose"),
    ("Ngày 03/07/2019", "date"), ("1) Ngảy 18/00 / 201g", "date"), ("Ngày 03 tháng 07 năm 2019", "date"),
    ("Sáng 1 Viên", "usage"), ("Sảng 1 Viền", "usage"), ("Uống: Sáng 2 Viên CHiều 2 Viên", "usage"),
    ("1/1", "other"), ("TOA THUỐC BHYT", "other"), ("Sáng", "other"),
    ("5123456", "other"), ("511 Nguyễn Trãi", "other"), ("51 tuổi", "other"),   # "51..." là số, không phải "SL"
])
def test_classify_text(text, label):
    assert classify_text(text) == label


def test_choose_labels_prefers_yolo():
    # OCR bịa "2)" trước chẩn đoán; YOLO chắc chắn là diagnose → giữ diagnose
    out = choose_labels([tb([0, 0, 10, 10], "diagnose", "2) Chản dưỡng", 0.83)])
    assert out[0].label == "diagnose"
    # YOLO gán other (hoặc không chắc) → hỏi text
    out = choose_labels([tb([0, 0, 10, 10], "other", "SL: 20 Viên", 0.9),
                         tb([0, 20, 10, 30], "usage", "SL: 30 Viên", 0.3)])
    assert [b.label for b in out] == ["quantity", "quantity"]
    # không có nhãn (cổ điển) → hoàn toàn bằng text
    out = choose_labels([TextBox(box=[0, 0, 10, 10], label=None, text="1) ABC 5mg")])
    assert out[0].label == "drugname"


# ---------- D3: chuẩn hóa ----------
def test_parse_quantity():
    assert parse_quantity("SL: 20 Viên") == (20, "Viên")
    assert parse_quantity("SL. 60 ViÊn") == (60, "Viên")
    assert parse_quantity("SL; 30 viên") == (30, "Viên")
    assert parse_quantity("SL: 2O Viên") == (20, "Viên")
    assert parse_quantity("Sáng 1 Viên") == (None, None)


def test_parse_date():
    assert parse_date("Ngày 03/07/2019") == "03/07/2019"
    assert parse_date("Ngày 3 tháng 7 năm 2019") == "03/07/2019"
    assert parse_date("1) Ngảy 18/09 / 201g") == "18/09/2019"
    assert parse_date("1) Ngảy 18/00 / 201g") is None          # tháng 00 không hợp lệ
    assert parse_date("Ngày O3/07/2O19") == "03/07/2019"
    assert parse_date("03/07/2019") == "03/07/2019"             # dạng VLM trả về


def test_clean_name_and_usage():
    assert clean_name("1) GLUCOFAST  850 850mg") == "GLUCOFAST 850 850mg"   # giữ hoa thường
    assert clean_usage("Sảng 1 Viền") == "Sáng 1 Viên"
    assert clean_usage("Uống: Sáng 2 Viên CHiều 2 Viên") == "Uống: Sáng 2 Viên Chiều 2 Viên"


# ---------- D2: dải ngang ----------
def test_bands_template1_and_reading_order():
    data = load_khang(SAMPLE)["VAIPE_P_TRAIN_1"]
    bands = split_bands(choose_labels(data))
    assert [b["quantity"].text for b in bands] == ["SL. 60 ViÊn", "SL; 30 ViÊn", "SL. 30 ViÊn"]
    assert [u.text for u in bands[0]["usage"]] == ["Sảng 1 Viền", "Chiều 1 ViÊn"]   # Sáng trước Chiều


def test_bands_template2_quantity_below():
    # VAIPE_P_TRAIN_1082: SL ở dòng riêng bên dưới tên thuốc
    boxes = [
        tb([37, 477, 565, 520], "drugname", "1) Amoxicilin (Fabamox 500) 500mg"),
        tb([282, 544, 750, 585], "usage", "Uống: Sáng 2 Viên CHiều 2 Viên"),
        tb([42, 550, 197, 577], "quantity", "SL: 20 Viên"),
        tb([35, 607, 585, 645], "drugname", "2) Paracetamol (Partamol Tab.) 500mg"),
        tb([282, 675, 750, 712], "usage", "Uống: Sáng 2 Viên Chiều 2 Viên"),
        tb([42, 677, 195, 707], "quantity", "SL: 20 Viên"),
        tb([37, 735, 740, 775], "drugname", "3) Methyl prednisolon (Methylprednisolon 4) 4mg"),
        tb([282, 802, 552, 842], "usage", "Uống: Sáng 4 Viên"),
        tb([42, 807, 195, 835], "quantity", "SL: 15 Viên"),
    ]
    items, _, _ = assemble(boxes)
    assert [it.quantity for it in items] == [20, 20, 15]
    assert [it.name.split()[0] for it in items] == ["Amoxicilin", "Paracetamol", "Methyl"]
    assert all(len(it.usage) == 1 for it in items)


def test_date_above_drugs_is_not_band_end():
    boxes = [tb([0, 10, 100, 30], "date", "Ngày 01/01/2020"),           # YOLO nhầm box phía trên thành date
             tb([0, 100, 300, 120], "drugname", "1) ABC 5mg"),
             tb([400, 101, 500, 119], "quantity", "SL: 10 Viên")]
    assert split_bands(boxes)[0]["quantity"] is not None


def test_dedupe_and_wrapped_names():
    a, b = tb([0, 0, 100, 20], "drugname", "1) ABC", 0.9), tb([1, 0, 100, 20], "drugname", "1) ABC", 0.5)
    assert len(dedupe([a, b])) == 1
    boxes = [tb([40, 100, 400, 120], "drugname", "1) THUỐC RẤT DÀI"),
             tb([40, 122, 300, 142], "drugname", "PHẦN XUỐNG DÒNG 5mg")]
    merged = [x for x in merge_wrapped_names(boxes) if x.label == "drugname"]
    assert len(merged) == 1 and merged[0].text.endswith("PHẦN XUỐNG DÒNG 5mg")
    overlap = [tb([40, 100, 400, 122], "drugname", "1) THUỐC DÀI"),          # dòng 2 chồng 2 px lên dòng 1
               tb([40, 120, 300, 140], "drugname", "PHẦN DƯỚI 5mg")]
    assert sum(x.label == "drugname" for x in merge_wrapped_names(overlap)) == 1
    two = [tb([40, 100, 400, 120], "drugname", "1) A 5mg"),                  # 2 thuốc thật liền nhau → không gộp
           tb([40, 122, 400, 142], "drugname", "2) B 10mg")]
    assert sum(x.label == "drugname" for x in merge_wrapped_names(two)) == 2


def test_reading_order_same_row():
    s, c = tb([288, 416, 376, 436], "usage", "Sáng"), tb([513, 414, 604, 432], "usage", "Chiều")
    assert [b.text for b in reading_order([c, s])] == ["Sáng", "Chiều"]


# ---------- đầu-cuối trên file mẫu của Khang ----------
def test_assemble_khang_sample():
    items, diag, date = assemble(load_khang(SAMPLE)["VAIPE_P_TRAIN_1"])
    assert [(it.name, it.quantity, it.unit) for it in items] == [
        ("GLUCOFAST 850 850mg", 60, "Viên"), ("DIAMICRON MR TAB 30MG 60'S 30mg", 30, "Viên"),
        ("ENALAPRIL 5mg", 30, "Viên")]
    assert items[0].usage == ["Sáng 1 Viên", "Chiều 1 Viên"]
    assert diag is not None and not diag.startswith("2)")
    assert date is None                                         # "18/00" không hợp lệ → None, không đoán


def test_prediction_never_reads_gt(tmp_path):
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    junk = copy.deepcopy(raw)
    for b in junk["boxes"]:
        b["gt"] = "RÁC RÁC 1) SL: 999"
    p = tmp_path / "junk.json"
    p.write_text(json.dumps(junk, ensure_ascii=False), encoding="utf-8")
    key = lambda path: [(it.name, it.quantity) for it in assemble(load_khang(path)["VAIPE_P_TRAIN_1"])[0]]
    assert key(SAMPLE) == key(p)


def test_load_khang_tolerates_trailing_comma(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(SAMPLE.read_text(encoding="utf-8").rstrip() + ",", encoding="utf-8")
    assert "VAIPE_P_TRAIN_1" in load_khang(p)


# ---------- D4: từ điển ----------
def test_lexicon_match():
    pytest.importorskip("rapidfuzz")
    from assemble.lexicon import Lexicon
    lex = Lexicon(["GLUCOFAST 850 850mg", "DIAMICRON MR TAB 30MG 60'S 30mg", "DIAMICRON MR TAB 60MG 30'S 60mg"])
    assert lex.match("GLUC0FAST 850 850mg")[0] == "GLUCOFAST 850 850mg"           # 1 lỗi OCR → kéo về
    assert lex.match("DIAMICRON MR TAB 30MG 60'S 30mg")[0] == "DIAMICRON MR TAB 30MG 60'S 30mg"
    assert lex.match("HOÀN TOÀN KHÁC")[3] is False                               # không khớp → giữ nguyên
    # thuốc ngoài từ điển, khác liều: không được kéo sang liều khác (ca thật trên test)
    lex2 = Lexicon(["CEPHALEXIN PMP 500 0,5g", "AMOXYCILIN 250MG 0,25g"])
    assert lex2.match("CEPHALEXIN 250MG 0,25g")[0] == "CEPHALEXIN 250MG 0,25g"
    assert lex2.match("CEPHALEXIN PMP 5OO 0,5g")[0] == "CEPHALEXIN PMP 500 0,5g"   # liều bị OCR làm hỏng → vẫn sửa


# ---------- VLM: parse + chuẩn hóa (không gọi API) ----------
def test_vlm_parse_and_normalize():
    content = 'Kết quả:\n```json\n{"items":[{"name":"1) ENALAPRIL 5mg","quantity":"30","unit":"viên",' \
              '"usage":["Sảng 1 Viên"]}],"diagnose":"Chẩn đoán: I10","date":"18/09/2019"}\n```'
    items, diag, date = vlm_to_items(parse_vlm_json(content))
    assert (items[0].name, items[0].quantity, items[0].unit) == ("ENALAPRIL 5mg", 30, "Viên")
    assert diag == "I10" and date == "18/09/2019"
    assert parse_vlm_json("không phải json") is None
    assert vlm_to_items(None) == ([], None, None)


# ---------- độ đo ----------
def test_items_metrics():
    gt = [{"name": "ABC 5mg", "quantity": 10}, {"name": "XYZ 10mg", "quantity": 20}]
    assert match_items(gt, gt) == (2, 0, 0)
    assert match_items(gt[:1], gt) == (1, 0, 1)                                    # thiếu 1
    assert match_items(gt + [{"name": "QQQ", "quantity": 1}], gt) == (2, 1, 0)     # thừa 1
    assert match_items([{"name": "ABC 5mg", "quantity": 11}, gt[1]], gt) == (1, 1, 1)   # sai SL
    assert match_items([{"name": "abc 5MG", "quantity": 10}], gt[:1]) == (1, 0, 0)    # không phân biệt hoa thường
    r = items_prf({"a": gt, "b": gt[:1]}, {"a": gt, "b": gt})
    assert r["tp"] == 3 and r["fn"] == 1 and r["exact_rate"] == 0.5
    lo, hi = bootstrap_ci(r["per_image"], n=200)
    assert 0 <= lo <= r["f1"] <= hi <= 1
