"""Crop geometry and blind review contracts, not OCR accuracy or CER tests."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import t4_ocr_prepare as prep


def test_padding_clips_to_original_frame():
    assert prep.crop_box([0,1,10,8],20,15)==[0,0,12,11]
    assert prep.crop_box([15,10,5,5],20,15)==[13,8,7,7]


@pytest.mark.parametrize('bbox',[[-1,0,2,2],[0,0,0,1],[19,1,2,2],[1.5,0,2,2]])
def test_invalid_annotation_not_silently_clipped(bbox):
    with pytest.raises(ValueError):prep.crop_box(bbox,20,15)


def test_all_variants_use_identical_rectangle_and_preserve_inputs():
    image=np.arange(20*30*3,dtype=np.uint8).reshape(20,30,3)
    record=dict(width=30,height=20,crop_bbox_xywh=[4,5,9,6])
    before=image.copy();crop=prep.same_crop(image,record)
    expected=prep.cv2.cvtColor(image,prep.cv2.COLOR_BGR2GRAY)[5:11,4:13]
    assert np.array_equal(crop,expected) and np.array_equal(image,before)
    for delta in (0,10,20):
        variant=np.clip(image.astype(int)+delta,0,255).astype(np.uint8)
        assert prep.same_crop(variant,record).shape==(6,9)
    assert not np.shares_memory(crop,image)


def test_changed_variant_frame_rejected():
    with pytest.raises(ValueError):
        prep.same_crop(np.zeros((21,30,3),np.uint8),dict(width=30,height=20,crop_bbox_xywh=[0,0,4,4]))


def test_review_is_pending_and_html_escapes_source_text():
    text='</script><script>alert(1)</script>'
    html=prep.review_html([dict(crop_id='a',text_source=text,reviewed_text=text,review_status='pending')])
    assert text not in html and '\\u003c/script>' in html
    assert 'ocr_smoke.json' not in html and 'text_pred' not in html
    assert 'text_source_sha256' in html and 'id="load"' in html


def test_existing_output_refused_before_any_ocr(tmp_path,monkeypatch):
    def fail(*a):raise AssertionError('Candidate loading called')
    monkeypatch.setattr(prep,'candidates',fail)
    with pytest.raises(FileExistsError):prep.run(tmp_path,None,None,tmp_path)
