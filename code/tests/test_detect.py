"""Khối B: độ đo phát hiện (ghép tham lam, cộng dồn) và bộ phát hiện cổ điển trên ảnh tổng hợp."""
import cv2
import numpy as np

from detect.classical import ClassicalDetector, remove_rules, sauvola
from metrics.detection import DetectionCounter, confusion_counts, iou, match_greedy, prf


def test_iou_basic():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-9


def test_match_greedy_one_to_one_and_order():
    gt = [[0, 0, 10, 10], [20, 0, 30, 10]]
    pred = [[0, 0, 10, 9], [1, 0, 11, 10], [100, 100, 110, 110]]   # hai dự đoán tranh GT 0
    pairs = match_greedy(gt, pred)
    assert len(pairs) == 1 and pairs[0][0] == 0 and pairs[0][1] == 0   # IoU 0.9 thắng 0.818


def test_match_class_aware():
    gt, pred = [[0, 0, 10, 10]], [[0, 0, 10, 10]]
    assert match_greedy(gt, pred, gt_labels=["drugname"], pred_labels=["other"]) == []
    assert len(match_greedy(gt, pred, gt_labels=["drugname"], pred_labels=["drugname"])) == 1


def test_counter_accumulates_over_images():
    c = DetectionCounter()
    c.add([[0, 0, 10, 10]], [[0, 0, 10, 10]])                      # TP
    c.add([[0, 0, 10, 10], [50, 50, 60, 60]], [[200, 200, 210, 210]])  # 2 FN, 1 FP
    r = c.result()
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 2)
    assert abs(r["f1"] - prf(1, 1, 2)[2]) < 1e-12


def test_per_class_and_confusion():
    c = DetectionCounter(class_aware=True)
    gt = [[0, 0, 10, 10], [20, 0, 30, 10]]
    pred = [[0, 0, 10, 10], [20, 0, 30, 10]]
    c.add(gt, pred, ["drugname", "quantity"], ["drugname", "other"])
    pc = c.per_class_result()
    assert pc["drugname"]["tp"] == 1 and pc["quantity"]["fn"] == 1 and pc["other"]["fp"] == 1
    cm = confusion_counts(gt, ["drugname", "quantity"], pred, ["drugname", "other"])
    assert cm[("drugname", "drugname")] == 1 and cm[("quantity", "other")] == 1


def test_sauvola_ignores_flat_gray_block():
    gray = np.full((100, 100), 230, np.uint8)
    gray[30:70, 30:70] = 215                       # khối xám che thông tin của VAIPE
    assert sauvola(gray).sum() == 0


def test_remove_rules_keeps_text():
    b = np.zeros((60, 200), np.uint8)
    b[30, 10:190] = 255                            # đường kẻ ngang
    b[10:22, 50:52] = 255                          # nét chữ đứng cao 12
    out = remove_rules(b, 36)
    assert out[30, 100] == 0 and out[15, 50] == 255


def test_classical_finds_text_lines(page):
    det = ClassicalDetector(a=0.5, b=1.5)
    boxes = det.detect(page)
    assert len(boxes) >= 12                        # 12 dòng chữ trong ảnh mẫu (có thể tách thành vài cụm)
    assert all(tb.label is None for tb in boxes)
    ys = sorted({tb.box[1] // 55 for tb in boxes})
    assert len(ys) >= 10                           # phủ gần hết các dòng
