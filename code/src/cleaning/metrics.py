"""Binary document scores; public inputs are uint8, ink=255, paper=0.

p-FM is the skeleton-recall + ordinary precision variant from H-DIBCO 2010
(Pratikakis et al., DOI:10.1109/ICFHR.2010.118, section III.B), NOT the later
weighted pseudo-precision/recall metric. Skeletonization: skimage Zhang.
DRD: normalized reciprocal-distance 5x5 weights, center zero; denominator is
the number of nonuniform 8x8 GT blocks (including partial boundary blocks).
Only valid neighbors contribute at image boundaries; weights are not locally
renormalized. See Lu et al., DOI:10.1109/LSP.2003.821748.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _binary(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 2 or image.size == 0 or not np.all((image == 0) | (image == 255))):
        raise ValueError("Expected nonempty 2D uint8 binary image, ink=255/background=0")
    return image == 255


def nonuniform_blocks(ink):
    """Count actual, nonoverlapping 8x8 blocks; don't invent padded GT pixels."""
    height, width = ink.shape
    total = 0
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = ink[y:y+8, x:x+8]
            count = np.count_nonzero(block)
            total += 0 < count < block.size
    return int(total)


class BinaryReference:
    """Cache GT-only skeleton and DRD maps for repeated candidate scoring."""

    def __init__(self, ground_truth):
        self.ink = _binary(ground_truth)
        self.n_ink = int(np.count_nonzero(self.ink))
        self.skeleton = skeletonize(self.ink, method="zhang")
        self.n_skeleton = int(np.count_nonzero(self.skeleton))
        self.nubn = nonuniform_blocks(self.ink)
        y, x = np.mgrid[-2:3, -2:3]
        distance = np.hypot(x, y)
        weights = np.divide(1.0, distance, out=np.zeros((5, 5)), where=distance > 0)
        self.weights = weights / weights.sum()
        self.weighted_ink = cv2.filter2D(self.ink.astype(np.float64), cv2.CV_64F,
                                        self.weights, borderType=cv2.BORDER_CONSTANT)
        self.valid_weight = cv2.filter2D(np.ones(self.ink.shape, np.float64), cv2.CV_64F,
                                        self.weights, borderType=cv2.BORDER_CONSTANT)

    def score(self, prediction):
        predicted = _binary(prediction)
        if predicted.shape != self.ink.shape:
            raise ValueError("Prediction and ground truth must have the same shape/frame")
        tp = int(np.count_nonzero(predicted & self.ink))
        n_pred = int(np.count_nonzero(predicted))
        fp, fn = n_pred - tp, self.n_ink - tp
        precision = tp / n_pred if n_pred else float(self.n_ink == 0)
        recall = tp / self.n_ink if self.n_ink else 1.0
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 1.0
        pseudo_recall = (np.count_nonzero(predicted & self.skeleton) / self.n_skeleton
                         if self.n_skeleton else 1.0)
        pfm = (2 * precision * pseudo_recall / (precision + pseudo_recall)
               if precision + pseudo_recall else 0.0)
        errors = predicted != self.ink
        n_errors = fp + fn
        if n_errors == 0:
            drd = 0.0
        elif self.nubn == 0:
            # DRD normalization is undefined for uniform GT; never claim zero
            # distortion for a wrong prediction in this degenerate case.
            drd = float("nan")
        else:
            to_paper = errors & ~predicted
            to_ink = errors & predicted
            distortion = self.weighted_ink[to_paper].sum()
            distortion += (self.valid_weight[to_ink] - self.weighted_ink[to_ink]).sum()
            drd = float(max(distortion, 0.0) / self.nubn)
        return {"f1": float(f1), "pfm_skeleton": float(pfm), "precision": float(precision),
                "recall": float(recall), "pseudo_recall": float(pseudo_recall),
                "psnr_db": float("inf") if n_errors == 0 else float(-10*np.log10(n_errors / predicted.size)),
                "drd": drd, "tp": tp, "fp": fp, "fn": fn, "error_pixels": n_errors,
                "gt_ink_pixels": self.n_ink, "gt_skeleton_pixels": self.n_skeleton, "nubn": self.nubn}
