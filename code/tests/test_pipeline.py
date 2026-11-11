import json

import numpy as np
import pytest

from assemble.load import load_khang
from common.contracts import Item, PrescriptionResult, TextBox
from common.paths import SAMPLES_DIR, VIETOCR_FINETUNED, YOLO_WEIGHTS
from pipeline import PRESETS, Pipeline, draw_overlay, handoff_record
from vaipe.dataset import VAIPEAnnotation


def _boxes():
    return [TextBox([46, 386, 294, 413], "drugname", 0.95, "1) RENAPRIL 5MG 5mg", 0.9),
            TextBox([595, 387, 708, 409], "quantity", 0.94, "SL: 28 Viên", 0.9),
            TextBox([900, 900, 950, 920], "other", 0.5, "abc", 0.2)]


def test_handoff_roundtrip_with_gt(tmp_path):
    gt = [VAIPEAnnotation(1, "1) RENAPRIL 5MG 5mg", "drugname", [46, 386, 294, 413]),
          VAIPEAnnotation(2, "SL: 28 Viên", "quantity", [596, 387, 708, 410])]
    rec = handoff_record("img1", _boxes(), gt=gt)
    assert [b["gt"] for b in rec["boxes"]] == ["1) RENAPRIL 5MG 5mg", "SL: 28 Viên", ""]
    p = tmp_path / "h.jsonl"
    p.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    pred = load_khang(p)["img1"]
    assert [b.label for b in pred] == ["drugname", "quantity", "other"]
    assert pred[1].text == "SL: 28 Viên" and pred[0].box == [46, 386, 294, 413]
    assert load_khang(p, use_gt_text=True)["img1"][2].text == ""


def test_handoff_without_gt_has_no_gt_field():
    assert all("gt" not in b for b in handoff_record("x", _boxes())["boxes"])


def test_overlay_shape():
    img = np.full((1070, 760, 3), 255, np.uint8)
    res = PrescriptionResult("x", items=[Item(name="A", name_raw="1) A", quantity=1, box=[46, 386, 708, 413])],
                             boxes=_boxes())
    out = draw_overlay(img, res)
    assert out.shape == img.shape and (out != 255).any()


def test_presets_and_validation():
    assert set(PRESETS) == {"classical", "hybrid", "dl_c2", "dl_c3"}
    with pytest.raises(ValueError):
        Pipeline(detector="ssd", lexicon=None)


@pytest.mark.skipif(not (YOLO_WEIGHTS.exists() and VIETOCR_FINETUNED.exists()), reason="chưa tải trọng số")
def test_end_to_end_sample():
    pytest.importorskip("ultralytics")
    pytest.importorskip("vietocr")
    from common.io import read_image
    res = Pipeline.preset("dl_c3")(read_image(SAMPLES_DIR / "rx_test_VAIPE_P_TRAIN_139.png"), "s")
    assert len(res.items) == 5
    assert all(it.quantity is not None for it in res.items)
    assert set(res.timing) == {"A", "B", "C", "D"}
