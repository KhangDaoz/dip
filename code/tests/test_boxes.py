"""Test common.boxes — IoU, đổi định dạng, ghép tham lam (plan §9)."""
import numpy as np
import pytest

from common.boxes import (
    iou, xyxy_to_xywh, xywh_to_xyxy, to_yolo, from_yolo,
    iou_matrix, greedy_match, precision_recall_f1,
)


def test_iou_identical_boxes():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_half_overlap():
    # 2 box 10x10 giao 5x5 → inter=25, union=175, IoU=25/175
    assert iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_iou_no_overlap():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_xyxy_xywh_round_trip():
    box = [10, 20, 30, 40]
    assert xywh_to_xyxy(xyxy_to_xywh(box)).tolist() == box


def test_to_yolo_from_yolo_round_trip():
    box = [10, 20, 110, 70]
    yolo = to_yolo(box, 200, 100)
    back = from_yolo(yolo, 200, 100)
    assert back.tolist() == pytest.approx(box, abs=1.0)


def test_iou_matrix_shape():
    M = iou_matrix([[0, 0, 10, 10]], [[5, 5, 15, 15], [0, 0, 10, 10]])
    assert M.shape == (1, 2)
    assert M[0, 1] == pytest.approx(1.0)


def test_greedy_match_order():
    # GT[1] khớp Pred[1] IoU=1.0, GT[0] khớp Pred[0] IoU=0.143
    gt = [[0, 0, 10, 10], [100, 100, 110, 110]]
    pred = [[5, 5, 15, 15], [100, 100, 110, 110]]
    pairs = greedy_match(gt, pred, iou_threshold=0.1)
    # cặp IoU 1.0 phải khớp trước
    assert pairs[0] == (1, 1, 1.0)
    assert pairs[1][0] == 0 and pairs[1][2] < 0.2


def test_greedy_match_threshold():
    gt = [[0, 0, 10, 10], [100, 100, 110, 110]]
    pred = [[5, 5, 15, 15], [100, 100, 110, 110]]
    # ngưỡng 0.5 → chỉ 1 cặp (IoU 1.0)
    pairs = greedy_match(gt, pred, iou_threshold=0.5)
    assert len(pairs) == 1
    assert pairs[0] == (1, 1, 1.0)


def test_greedy_match_unique():
    # 1 GT, 2 pred IoU 1.0 + 0.5 → chỉ khớp 1
    pairs = greedy_match([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10]], 0.4)
    assert len(pairs) == 1
    assert pairs[0][2] == 1.0


def test_precision_recall_f1():
    # 3 TP, 1 FP, 0 FN → P=3/4, R=1, F1=2*0.75*1/(0.75+1)
    p, r, f1 = precision_recall_f1(3, 1, 0)
    assert p == pytest.approx(0.75)
    assert r == 1.0
    assert f1 == pytest.approx(2 * 0.75 * 1 / 1.75)


def test_prf1_zero():
    p, r, f1 = precision_recall_f1(0, 0, 1)
    assert (p, r, f1) == (0.0, 0.0, 0.0)
