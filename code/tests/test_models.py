import cv2
import pytest

from common.contracts import BOX_LABELS
from common.paths import SAMPLES_DIR, VIETOCR_FINETUNED, YOLO_WEIGHTS

SAMPLE = SAMPLES_DIR / "rx_vaipe_01.png"


@pytest.fixture(scope="module")
def img():
    im = cv2.imread(str(SAMPLE))
    assert im is not None, f"thiếu ảnh mẫu {SAMPLE}"
    return im


def test_yolo_detects_prescription_fields(img):
    pytest.importorskip("ultralytics")
    if not YOLO_WEIGHTS.exists():
        pytest.skip("chưa tải yolo_vaipe.pt")
    from detect.yolo import detect_yolo, load_yolo
    assert tuple(load_yolo().names[i] for i in range(6)) == BOX_LABELS
    boxes = detect_yolo(img)
    labels = [b.label for b in boxes]
    for b in boxes:
        b.validate()
    assert labels.count("drugname") >= 1
    assert labels.count("drugname") == labels.count("quantity")
    assert labels.count("date") == 1


def test_vietocr_reads_detected_boxes(img):
    pytest.importorskip("vietocr")
    if not (YOLO_WEIGHTS.exists() and VIETOCR_FINETUNED.exists()):
        pytest.skip("chưa tải trọng số")
    pytest.importorskip("ultralytics")
    from detect.yolo import detect_yolo
    from recognize.vietocr_engine import VietOCREngine
    boxes = [b for b in detect_yolo(img) if b.label in ("quantity", "date")]
    VietOCREngine("finetuned").recognize(img, boxes)
    for b in boxes:
        b.validate()
        if b.label == "quantity":
            assert b.text.startswith("SL"), b.text
        else:
            assert b.text.startswith("Ngày"), b.text
