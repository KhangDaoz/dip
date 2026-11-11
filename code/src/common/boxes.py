"""Hộp bao vùng chữ: tiện ích IoU, đổi định dạng, ghép cặp tham lam.

Quy ước toàn pipeline v2: box = [x0, y0, x1, y1] (xyxy), pixel, số nguyên, gốc trên-trái.
Đây LÀ quy ước nhãn VAIPE gốc (khác v1 dùng [x, y, w, h]).

YOLO dùng `cx cy w h` chuẩn hóa 0–1; hàm `to_yolo`/`from_yolo` chuyển đổi.
"""
from __future__ import annotations

import numpy as np


def xyxy_to_xywh(box) -> np.ndarray:
    """[x0, y0, x1, y1] → [x, y, w, h] (pixel, góc trên-trái + kích thước)."""
    box = np.asarray(box, dtype=float)
    return np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]])


def xywh_to_xyxy(box) -> np.ndarray:
    """[x, y, w, h] → [x0, y0, x1, y1]."""
    box = np.asarray(box, dtype=float)
    return np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])


def to_yolo(box, img_w: int, img_h: int) -> np.ndarray:
    """xyxy (pixel) → [cx, cy, w, h] chuẩn hóa 0–1 theo kích thước ảnh."""
    x0, y0, x1, y1 = box
    return np.array([(x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h,
                     (x1 - x0) / img_w, (y1 - y0) / img_h])


def from_yolo(yolo_box, img_w: int, img_h: int) -> np.ndarray:
    """[cx, cy, w, h] chuẩn hóa → xyxy pixel."""
    cx, cy, w, h = yolo_box
    return np.array([(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                     (cx + w / 2) * img_w, (cy + h / 2) * img_h])


def clip_box(box, img_w: int, img_h: int) -> list:
    """Kẹp box xyxy vào trong ảnh, sắp lại nếu đảo, làm tròn thành int."""
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((min(max(x0, 0), img_w), min(max(x1, 0), img_w)))
    y0, y1 = sorted((min(max(y0, 0), img_h), min(max(y1, 0), img_h)))
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def iou(a, b) -> float:
    """IoU hai hộp xyxy. Trả về 0 nếu không giao."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    """Ma trận IoU |A| × |B|, dùngBroadcast để nhanh."""
    A = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    B = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    x0 = np.maximum(A[:, None, 0], B[None, :, 0])
    y0 = np.maximum(A[:, None, 1], B[None, :, 1])
    x1 = np.minimum(A[:, None, 2], B[None, :, 2])
    y1 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(A[:, 2] - A[:, 0], 0, None) * np.clip(A[:, 3] - A[:, 1], 0, None)
    area_b = np.clip(B[:, 2] - B[:, 0], 0, None) * np.clip(B[:, 3] - B[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def greedy_match(gt_boxes, pred_boxes, iou_threshold: float = 0.5):
    """Ghép tham lam theo IoU giảm dần. Mỗi GT khớp tối đa 1 pred và ngược lại.

    Trả về list[(gt_idx, pred_idx, iou)] — các cặp ghép được (IoU >= ngưỡng).
    Dùng cho P/R/F1 detection theo C8 · S49.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    M = iou_matrix(gt_boxes, pred_boxes)
    pairs = []
    used_gt, used_pred = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-M.ravel()), M.shape))[0]
    for gi, pi in order:
        if gi in used_gt or pi in used_pred:
            continue
        v = M[gi, pi]
        if v < iou_threshold:
            break
        pairs.append((int(gi), int(pi), float(v)))
        used_gt.add(gi)
        used_pred.add(pi)
    return pairs


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple:
    """Precision / Recall / F1 từ TP/FP/FN. Trả về (P, R, F1), 0 nếu mẫu âm."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f1)
