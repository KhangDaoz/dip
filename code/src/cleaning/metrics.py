"""Độ đo cho T1–T5 (KeHoach.md mục 6.2)."""
import numpy as np


def psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    """PSNR = 20 log10(255 / e_rms) (➕ C6 · S39)."""
    raise NotImplementedError("psnr")


def ssim(pred: np.ndarray, ref: np.ndarray) -> float:
    raise NotImplementedError("ssim")


def f_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """F-measure mức pixel, mực = 255 (C8 · S49)."""
    raise NotImplementedError("f_measure")


def pseudo_f_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """pseudo-F theo chuẩn DIBCO."""
    raise NotImplementedError("pseudo_f_measure")


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    raise NotImplementedError("mask_iou")
