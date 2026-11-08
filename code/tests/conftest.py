"""Ảnh tài liệu tổng hợp dùng chung cho test — biết trước ground truth, không cần tải dữ liệu.

Fixture:
    page        ảnh BGR "tờ giấy" thẳng: nền 225, N_LINES dòng chữ đen
    page_gray   ảnh xám của page
    text_mask   ground truth mực của page (mực = 255)
    photo       dict {"img", "corners" (TL,TR,BR,BL), "H"}: page dán lên nền bàn tối với phối cảnh đã biết
"""
import cv2
import numpy as np
import pytest

PAGE_H, PAGE_W = 800, 600
N_LINES = 12
PAPER = 225          # giấy trong ảnh chụp thật hiếm khi trắng 255
INK = 30
LINE_Y0, LINE_STEP = 70, 55


def make_page() -> np.ndarray:
    page = np.full((PAGE_H, PAGE_W, 3), PAPER, np.uint8)
    for i in range(N_LINES):
        y = LINE_Y0 + i * LINE_STEP
        cv2.putText(page, f"Paracetamol 500mg x{i + 1} sang toi", (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (INK, INK, INK), 2, cv2.LINE_AA)
    return page


@pytest.fixture
def page() -> np.ndarray:
    return make_page()


@pytest.fixture
def page_gray(page) -> np.ndarray:
    return cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def text_mask(page_gray) -> np.ndarray:
    return np.where(page_gray < (PAPER + INK) // 2, 255, 0).astype(np.uint8)


@pytest.fixture
def photo(page) -> dict:
    return make_photo(page)


def make_photo(page: np.ndarray) -> dict:
    canvas = np.full((1000, 1000, 3), 60, np.uint8)
    canvas = cv2.add(canvas, np.random.default_rng(0).integers(0, 15, canvas.shape, dtype=np.uint8))
    src = np.float32([[0, 0], [PAGE_W - 1, 0], [PAGE_W - 1, PAGE_H - 1], [0, PAGE_H - 1]])
    dst = np.float32([[180, 110], [790, 170], [840, 900], [130, 860]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, H, (1000, 1000), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.full(page.shape[:2], 255, np.uint8), H, (1000, 1000))
    canvas[mask > 0] = warped[mask > 0]
    return {"img": canvas, "corners": dst, "H": H}


def rotate_ccw(img: np.ndarray, angle_deg: float, border=PAPER) -> np.ndarray:
    """Xoay ngược chiều kim đồng hồ bằng OpenCV — độc lập với code đang được test."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    value = (border,) * 3 if img.ndim == 3 else border
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=value)


def f_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    tp = np.logical_and(p, g).sum()
    if tp == 0:
        return 0.0
    precision, recall = tp / p.sum(), tp / g.sum()
    return 2 * precision * recall / (precision + recall)
