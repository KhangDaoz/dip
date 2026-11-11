"""Pipeline (plan §9): khối A, B cổ điển, ablation B+C chuẩn chạy không cần trọng số; YOLO/VietOCR skip nếu thiếu."""
import sys
from pathlib import Path

import numpy as np
import pytest

from common.contracts import PrescriptionResult
from common.paths import SAMPLES_DIR, YOLO_WEIGHTS
from vaipe.dataset import VAIPEAnnotation

from conftest import make_page, rotate_ccw


def test_enhance_all_steps(page):
    from enhance.pipeline import enhance
    out, info = enhance(page, "full")
    assert out.dtype == np.uint8 and out.ndim == 3
    assert set(info["timing"]) == {"rectify", "deskew", "shadow", "denoise", "clahe"}


def test_deskew_recovers_angle():
    from enhance.deskew import estimate_skew
    est = estimate_skew(rotate_ccw(make_page(), 3.0))
    assert abs(est - 3.0) <= 0.5


def test_classical_detector_finds_lines(page):
    from detect.classical import detect_classical
    boxes = detect_classical(page)
    assert len(boxes) >= 12 and all(b.label is None for b in boxes)


def test_pipeline_gt_ablation():
    from pipeline import Pipeline
    gt = [VAIPEAnnotation(1, "1) RENAPRIL 5MG 5mg", "drugname", [46, 386, 294, 413]),
          VAIPEAnnotation(2, "SL: 28 Viên", "quantity", [595, 387, 708, 409]),
          VAIPEAnnotation(3, "Sáng 1 Viên", "usage", [289, 416, 376, 436]),
          VAIPEAnnotation(4, "Ngày 03/07/2019", "date", [484, 633, 621, 652])]
    res = Pipeline(detector="gt", ocr="gt", lexicon=None)(np.zeros((800, 760, 3), np.uint8), "x", gt=gt)
    assert isinstance(res, PrescriptionResult)
    assert res.items[0].name == "RENAPRIL 5MG 5mg" and res.items[0].quantity == 28
    assert res.date == "03/07/2019"


def test_run_pipeline_one_sample():
    if not YOLO_WEIGHTS.exists():
        pytest.skip("chưa có code/weights/yolo_vaipe.pt")
    pytest.importorskip("vietocr")
    import subprocess
    sample = sorted(SAMPLES_DIR.glob("rx_vaipe_*.png"))[0]
    r = subprocess.run([sys.executable, "run_pipeline.py", str(sample), "--preset", "dl_c2", "--quiet"],
                       cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
