"""Known-answer mask metrics and safety contracts for TRAIN-only T4 sanity."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import t4_detection as d


def test_positive_known_answer_and_no_mutation():
    gt=np.array([[255,255,0],[0,0,0]],np.uint8)
    pred=np.array([[255,0,255],[0,0,0]],np.uint8)
    before=pred.copy(); score=d.positive_scores(pred,gt)
    assert (score['tp'],score['fp'],score['fn'])==(1,1,1)
    assert score['iou']==pytest.approx(1/3)
    assert score['precision']==score['recall']==.5
    assert np.array_equal(pred,before)


def test_empty_prediction_on_positive_is_failure_not_na():
    mask=np.array([[0,255]],np.uint8)
    score=d.positive_scores(np.zeros_like(mask),mask)
    assert score['iou']==score['precision']==score['recall']==0
    assert score['empty_prediction']


def test_identical_positive_is_one():
    mask=np.array([[0,255]],np.uint8)
    score=d.positive_scores(mask,mask)
    assert score['iou']==score['precision']==score['recall']==1


def test_empty_gt_must_use_negative_table():
    with pytest.raises(ValueError,match='negative-control'):
        d.positive_scores(np.zeros((2,2),np.uint8),np.zeros((2,2),np.uint8))


def test_negative_metrics_and_eight_connected_components():
    mask=np.eye(2,dtype=np.uint8)*255
    score=d.negative_scores(mask)
    assert score['fp_pixels']==2 and score['fp_pixel_ratio']==.5
    assert score['fp_components_8_connected']==1 and score['any_false_positive']
    assert score['iou'] is score['precision'] is score['recall'] is None
    empty=d.negative_scores(np.zeros((2,2),np.uint8))
    assert not empty['any_false_positive'] and empty['fp_components_8_connected']==0


@pytest.mark.parametrize('mask',[np.zeros((2,2),bool),np.ones((2,2),np.uint8),
                               np.zeros((0,2),np.uint8),np.zeros((2,2,3),np.uint8)])
def test_invalid_mask_conventions_rejected(mask):
    with pytest.raises(ValueError):d.negative_scores(mask)


def test_frame_mismatch_rejected():
    with pytest.raises(ValueError):
        d.positive_scores(np.zeros((2,3),np.uint8),np.ones((3,2),np.uint8)*255)


def test_selection_ignores_val_test_and_is_deterministic():
    rows=[]
    for i,(kind,level,color) in enumerate(sorted(d.CELLS)):
        for split,sid in [('test','000'),('val','001'),('train','b'),('train','a')]:
            rows.append(dict(source_id=f'{sid}_{i}',sample_id=f'{split}_{sid}_{i}',split=split,
                             recipe_name='single_'+('stamp' if kind=='stamp' else 'pen'),
                             stamp_level=level if kind=='stamp' else 0,
                             pen_level=level if kind=='pen' else 0,pen_color=color))
    selected=d.select_train(rows)
    assert len(selected)==7 and all(r['split']=='train' and r['source_id'].startswith('a_') for r in selected)
    assert selected==d.select_train(list(reversed(rows)))
    with pytest.raises(ValueError):d.select_train([])


def test_black_mask_never_exposed_as_signature(monkeypatch):
    image=np.full((3,4,3),255,np.uint8)
    zero=np.zeros((3,4),np.uint8)
    monkeypatch.setattr(d,'ink_masks',lambda image:dict(red=zero.copy(),blue=zero.copy(),black=np.full_like(zero,255)))
    monkeypatch.setattr(d,'split_stamp_vs_text',lambda red:(zero.copy(),zero.copy()))
    masks=d.predictions(image)
    assert set(masks)==set(d.DETECTORS) and all(not mask.any() for mask in masks.values())


def test_expanding_stamp_partition_rejected(monkeypatch):
    image=np.full((3,4,3),255,np.uint8);zero=np.zeros((3,4),np.uint8)
    monkeypatch.setattr(d,'ink_masks',lambda image:dict(red=zero.copy(),blue=zero.copy(),black=zero.copy()))
    monkeypatch.setattr(d,'split_stamp_vs_text',lambda red:(np.full_like(red,255),zero.copy()))
    with pytest.raises(ValueError,match='expanded'):d.predictions(image)


def test_overlay_does_not_change_input_or_background():
    image=np.full((3,4,3),200,np.uint8);original=image.copy()
    gt=np.zeros((3,4),np.uint8);gt[1,1]=255
    out=d.overlay(image,np.zeros_like(gt),gt)
    assert np.array_equal(image,original)
    assert np.array_equal(out[gt==0],image[gt==0])
    assert out.dtype==image.dtype and out.shape==image.shape


def test_existing_output_refused_before_preflight(tmp_path,monkeypatch):
    def fail(*a):raise AssertionError('preflight called')
    monkeypatch.setattr(d,'preflight',fail)
    with pytest.raises(FileExistsError):d.run(tmp_path,None,tmp_path)


def test_macro_average_not_pixel_pooled():
    rows=[dict(detector='red_color',source_id='a',iou=0.,precision=0.,recall=0.,empty_prediction=True),
          dict(detector='red_color',source_id='b',iou=1.,precision=1.,recall=1.,empty_prediction=False)]
    pos,neg=d.summarize(rows,[])
    assert pos[0]['mean_iou']==.5 and pos[0]['n_sources']==2 and not neg
