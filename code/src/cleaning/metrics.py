"""Độ đo cho T1–T5 (KeHoach.md mục 6.2)."""
import cv2
import numpy as np


def psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    """PSNR = 20 log10(255 / e_rms), e_rms = sqrt(1/MN ΣΣ (f̂ − f)²) (➕ C6 · S39)."""
    e = np.sqrt(np.mean((pred.astype(np.float64) - ref.astype(np.float64)) ** 2))
    return float("inf") if e == 0 else float(20 * np.log10(255.0 / e))


def ssim(pred: np.ndarray, ref: np.ndarray) -> float:
    """SSIM (📘 độ đo chuẩn, QĐ · S6)."""
    from skimage.metrics import structural_similarity
    if pred.ndim == 3:
        pred, ref = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    return float(structural_similarity(pred, ref, data_range=255))


def edge_band(gt: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Dải viền quanh nét chữ: (A ⊕ B) − (A ⊖ B), B = ksize × ksize (mở rộng tách biên C7 · S40). Vùng = 255."""
    a = (gt > 0).astype(np.uint8)
    se = np.ones((ksize, ksize), np.uint8)
    return ((cv2.dilate(a, se) - cv2.erode(a, se)) > 0).astype(np.uint8) * 255


def _counts(pred, gt, ignore):
    p, g = pred > 0, gt > 0
    if ignore is not None:
        keep = ~(ignore > 0)
        p, g = p & keep, g & keep
    tp = np.logical_and(p, g).sum()
    return tp, p.sum(), g.sum()


def f_measure(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray = None) -> float:
    """F-measure mức pixel, mực = 255 (C8 · S49): P = TP/(TP+FP), R = TP/(TP+FN), F = 2PR/(P+R).
    ignore: pixel không chấm (dải viền, KeHoach.md mục 6.4)."""
    tp, np_, ng = _counts(pred, gt, ignore)
    if tp == 0:
        return 0.0
    p, r = tp / np_, tp / ng
    return float(2 * p * r / (p + r))


def pseudo_f_measure(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray = None) -> float:
    """pseudo-F theo chuẩn DIBCO: recall tính trên khung xương (skeleton) của nét chữ đáp án."""
    from skimage.morphology import skeletonize
    skel = skeletonize(gt > 0).astype(np.uint8) * 255
    tp, np_, _ = _counts(pred, gt, ignore)
    if tp == 0:
        return 0.0
    p = tp / np_
    tps, _, ns = _counts(pred, skel, ignore)
    pr = tps / max(ns, 1)
    return float(2 * p * pr / (p + pr)) if p + pr > 0 else 0.0


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else 1.0
