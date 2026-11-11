import numpy as np
import pytest

from common.contracts import TextBox
from recognize.tesseract_engine import BORDER, SCALE, available, prepare


def _line(text="SL: 28 Vien"):
    import cv2
    img = np.full((40, 260, 3), 235, np.uint8)
    cv2.putText(img, text, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return img


@pytest.mark.parametrize("mode", ["gray", "binary"])
def test_prepare_shape_and_polarity(mode):
    crop = _line()
    out = prepare(crop, mode)
    assert out.ndim == 2 and out.dtype == np.uint8
    assert out.shape == (int(40 * SCALE) + 2 * BORDER, int(260 * SCALE) + 2 * BORDER)
    assert out[:BORDER].min() == 255
    assert np.median(out) > 200


def test_prepare_rejects_unknown_mode():
    with pytest.raises(ValueError):
        prepare(_line(), "otsu")


@pytest.mark.skipif(not available("vie"), reason="chưa cài Tesseract 5 + gói vie")
@pytest.mark.parametrize("mode", ["gray", "binary"])
def test_reads_synthetic_line(mode):
    from recognize.tesseract_engine import TesseractEngine
    ocr = TesseractEngine(mode)
    boxes = [TextBox(box=[0, 0, 260, 40])]
    ocr.recognize(_line(), boxes)
    assert "28" in boxes[0].text and 0.0 <= boxes[0].text_conf <= 1.0
    assert ocr.read(np.zeros((0, 0, 3), np.uint8)) == ("", 0.0)
