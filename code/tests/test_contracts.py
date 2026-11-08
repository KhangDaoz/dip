"""Hợp đồng dữ liệu giữa các mắt xích — ví dụ lấy đúng từ kế hoạch của từng người."""
import json

import numpy as np
import pytest

from common.contracts import (CleaningResult, LabReportResult, PreprocessResult,
                              PrescriptionResult)

PREPROCESS_EXAMPLE = {
    "image_id": "rx_000123",
    "rectified_path": "out/rx_000123_rect.png",
    "homography": [[1.02, 0.03, -12.1], [0.01, 0.98, -8.4], [0.0, 0.0, 1.0]],
    "corners_src": [[34, 51], [980, 40], [1002, 1390], [20, 1401]],
    "corner_method": "contour",
    "skew_deg": -2.4,
    "doc_type": "prescription",
    "doc_confidence": 0.94,
    "quality": {"blur_var": 182.5, "glare_ratio": 0.012, "is_blurry": False, "has_glare": False,
                "glare_removed": False},
}

CLEANING_EXAMPLE = {
    "image_id": "rx_000123",
    "clean_gray_path": "out/rx_000123_clean.png",
    "binary_path": "out/rx_000123_bin.png",
    "masks": {"stamp": "out/rx_000123_stamp.png", "handwriting": "out/rx_000123_hw.png",
              "red_text": "out/rx_000123_red.png"},
    "components": [{"bbox": [120, 340, 38, 22], "label": "printed", "confidence": 0.97}],
    "params": {"background_kernel": 41, "sauvola_window": 31, "sauvola_k": 0.2},
}

LAB_EXAMPLE = {
    "image_id": "lab_000045",
    "report_type": "CBC",
    "rows": [{"bbox": [60, 410, 820, 28], "parameter": "HB", "result": 14.9, "unit": "g/dl",
              "range": [13.5, 17.5], "flag": "NORMAL", "loinc": "718-7", "confidence": 0.91}],
}

RX_EXAMPLE = {
    "image_id": "rx_000123",
    "patient": {"bbox": [24, 280, 420, 50], "text": "Soumya Ranganathan, 37 years"},
    "diagnosis": {"bbox": [24, 350, 170, 55], "text": "Affective disorder"},
    "items": [{"bbox": [24, 480, 380, 90], "name": "Vinlep 150 Tablet", "strength": "150 mg",
               "dose": {"morning": 0, "noon": 0, "evening": 1}, "days": 90, "note": "after food",
               "confidence": 0.88}],
}


@pytest.mark.parametrize("cls, example", [
    (PreprocessResult, PREPROCESS_EXAMPLE),
    (CleaningResult, CLEANING_EXAMPLE),
    (LabReportResult, LAB_EXAMPLE),
    (PrescriptionResult, RX_EXAMPLE),
])
def test_roundtrip_json(cls, example):
    obj = cls.from_dict(example)
    again = cls.from_dict(json.loads(json.dumps(obj.to_dict())))
    assert again.to_dict() == example


def test_preprocess_rejects_bad_doc_type():
    with pytest.raises(ValueError):
        PreprocessResult.from_dict({**PREPROCESS_EXAMPLE, "doc_type": "invoice"})


def test_preprocess_rejects_singular_homography():
    with pytest.raises(ValueError):
        PreprocessResult.from_dict({**PREPROCESS_EXAMPLE, "homography": [[0] * 3] * 3})


def test_component_rejects_unknown_label():
    bad = {**CLEANING_EXAMPLE, "components": [{"bbox": [0, 0, 1, 1], "label": "logo", "confidence": 1}]}
    with pytest.raises(ValueError):
        CleaningResult.from_dict(bad)


def test_lab_rejects_unknown_flag():
    bad = {**LAB_EXAMPLE, "rows": [{**LAB_EXAMPLE["rows"][0], "flag": "H"}]}
    with pytest.raises(ValueError):
        LabReportResult.from_dict(bad)


def test_to_source_inverts_homography():
    pre = PreprocessResult.from_dict(PREPROCESS_EXAMPLE)
    src = np.array([[100.0, 200.0], [500.0, 900.0]])
    H = np.asarray(pre.homography)
    homog = np.hstack([src, np.ones((2, 1))]) @ H.T
    rect = homog[:, :2] / homog[:, 2:3]
    assert np.allclose(pre.to_source(rect), src)
