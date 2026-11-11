import pytest

from common.contracts import Item, TextBox
from metrics.detection import evaluate as det_eval
from metrics.items import bootstrap_ci, evaluate, image_counts
from metrics.ocr import cer, corpus_cer, corpus_wer, wer

GT = [{"name": "RENAPRIL 5MG 5mg", "quantity": 28}, {"name": "Amlodipin 5mg", "quantity": 14}]


def test_cer_wer():
    assert cer("abc", "abc") == 0
    assert cer("abd", "abc") == pytest.approx(1 / 3)
    assert wer("sáng 1 viên", "Sáng 1 Viên") == pytest.approx(2 / 3)
    assert corpus_cer(["ab", "cd"], ["ab", "ce"]) == pytest.approx(1 / 4)
    assert corpus_wer(["a b", "c"], ["a b", "d"]) == pytest.approx(1 / 3)


def test_items_perfect():
    pred = [Item(name="RENAPRIL 5MG 5mg", name_raw="", quantity=28), Item(name="Amlodipin 5mg", name_raw="", quantity=14)]
    agg, _ = evaluate({"a": GT}, {"a": pred})
    assert agg["item_f1"] == 1.0 and agg["exact_rate"] == 1.0


def test_items_missing_and_extra():
    agg, _ = evaluate({"a": GT}, {"a": [Item(name="RENAPRIL 5MG 5mg", name_raw="", quantity=28)]})
    assert agg["item_p"] == 1.0 and agg["item_r"] == 0.5 and agg["exact_rate"] == 0
    extra = GT + [{"name": "Thuốc thừa", "quantity": 1}]
    agg, _ = evaluate({"a": GT}, {"a": extra})
    assert agg["item_r"] == 1.0 and agg["item_p"] == pytest.approx(2 / 3)


def test_items_ocr_tolerance_and_quantity():
    c = image_counts(GT, [{"name": "RENAPRIL 5MG 5mq", "quantity": 28},
                          {"name": "Amlodipin 5mg", "quantity": 10}])
    assert c["tp"] == 1 and c["name_tp"] == 2 and c["qty_ok"] == 1


def test_lexicon_match_used():
    pred = [Item(name="RENAPRL", name_raw="", quantity=28, lexicon_match="RENAPRIL 5MG 5mg")]
    assert image_counts(GT[:1], pred)["tp"] == 1
    assert image_counts(GT[:1], pred, use_lexicon=False)["tp"] == 0


def test_bootstrap():
    _, counts = evaluate({str(i): GT for i in range(20)}, {str(i): GT if i % 2 else [] for i in range(20)})
    base, lo, hi = bootstrap_ci(counts, n=200)
    assert lo <= base <= hi


def test_detection():
    gt = [[TextBox([0, 0, 10, 10], "drugname"), TextBox([20, 0, 30, 10], "quantity")]]
    pred = [[TextBox([0, 0, 10, 10], "drugname"), TextBox([20, 0, 30, 10], "usage"), TextBox([50, 50, 60, 60], "other")]]
    r = det_eval(gt, pred)
    assert r["all"][0] == pytest.approx(2 / 3) and r["all"][1] == 1.0
    r = det_eval(gt, pred, by_class=True)
    assert r["drugname"] == (1.0, 1.0, 1.0) and r["quantity"][1] == 0.0
