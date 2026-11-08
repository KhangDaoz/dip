"""T4 pilot invariants and known-answer fixed-GT-region image scores."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import t4_restoration as r


@pytest.mark.parametrize('method',r.METHODS)
def test_methods_preserve_input_mask_and_outside(method):
    image=np.full((40,40,3),220,np.uint8);image[15:24,15:24]=(20,50,180)
    mask=np.zeros((40,40),np.uint8);mask[14:25,14:25]=255
    a=image.copy();b=mask.copy();out=r.apply_method(image,mask,method)
    assert np.array_equal(image,a) and np.array_equal(mask,b)
    assert np.array_equal(out[mask==0],image[mask==0])
    assert not np.shares_memory(out,image) and out.dtype==np.uint8
    if method=='none':assert np.array_equal(out,image)
    if method=='fill_white':assert np.all(out[mask>0]==255)


@pytest.mark.parametrize('method',r.METHODS)
def test_empty_prediction_is_retained_noop(method):
    image=np.full((30,30,3),123,np.uint8);mask=np.zeros((30,30),np.uint8)
    assert np.array_equal(r.apply_method(image,mask,method),image)


def test_outside_mask_backend_violation_rejected(monkeypatch):
    image=np.full((30,30,3),123,np.uint8);mask=np.zeros((30,30),np.uint8);mask[5,5]=255
    monkeypatch.setattr(r,'remove_marks',lambda *args,**kwargs:np.zeros_like(image))
    with pytest.raises(ValueError,match='invariants'):r.apply_method(image,mask,'telea')


def test_fixed_gt_roi_even_when_prediction_empty():
    clean=np.full((3,4,3),100,np.uint8);image=clean.copy();image[1,1]=120
    gt=np.zeros((3,4),np.uint8);gt[1,1]=255
    no_text=np.zeros_like(gt)
    score=r.evaluate(clean,image,gt,no_text)
    assert score['roi_n_pixels']==1 and score['roi_mae']==20 and score['roi_mse']==400
    assert score['text_proxy_n_pixels']==0 and score['text_proxy_mae'] is None
    assert score['text_proxy_psnr_db'] is None
    assert np.isfinite(score['roi_psnr_db'])


def test_exact_match_psnr_infinite_not_capped():
    clean=np.full((3,4),100,np.uint8);roi=np.ones_like(clean,dtype=bool)
    score=r.region_scores(clean,clean,roi)
    assert score['mse']==0 and score['psnr_db']==float('inf')


def test_black_pen_has_only_gt_diagnostic_branch():
    gt=np.zeros((3,4),np.uint8)
    row=dict(recipe_name='single_pen',pen_color='black')
    assert list(r.mask_branches(row,gt,{}))==['gt_diagnostic']


def test_stamp_candidate_mask_is_not_expanded():
    gt=np.full((3,4),255,np.uint8);pred=np.zeros_like(gt);pred[1,1]=255
    branch=r.mask_branches(dict(recipe_name='single_stamp'),gt,{'stamp_candidate':pred})
    assert branch['predicted_stamp_candidate'] is pred and np.count_nonzero(pred)==1


def test_roi_frame_mismatch_rejected():
    with pytest.raises(ValueError):
        r.region_scores(np.zeros((3,4),np.uint8),np.zeros((4,3),np.uint8),np.ones((3,4),bool))


def test_invalid_method_rejected():
    with pytest.raises(ValueError):r.apply_method(np.zeros((3,4,3),np.uint8),np.zeros((3,4),np.uint8),'auto')


def test_macro_summary_excludes_na_text_but_reports_denominator():
    a=dict(split='val',kind='stamp',pen_color='none',mask_source='gt_diagnostic',method='none',
           source_id='a',empty_mask=False,roi_mae=10.,roi_mse=100.,roi_n_pixels=1,
           text_proxy_n_pixels=0,text_proxy_mae=None,text_proxy_mse=None)
    b=dict(a,source_id='b',roi_mae=30.,roi_mse=900.,text_proxy_n_pixels=1,
           text_proxy_mae=8.,text_proxy_mse=64.)
    row=r.summarize([a,b])[0]
    assert row['n_samples']==2 and row['n_text_proxy_samples']==1
    assert row['mean_roi_mae']==20 and row['mean_roi_mse']==500
    assert row['mean_text_proxy_mae']==8


def test_existing_output_rejected_before_pixel_or_preflight(tmp_path,monkeypatch):
    def fail(*args):raise AssertionError('preflight called')
    monkeypatch.setattr(r,'preflight',fail)
    with pytest.raises(FileExistsError):r.run(tmp_path,None,None,tmp_path)


def test_focus_crop_contains_occlusion_even_for_long_box():
    from t4_restoration_review import focus_crop
    gt=np.zeros((300,1000),np.uint8);gt[100,950]=255
    label={'boxes_clean':[dict(id=1,text='long line',bbox=[0,90,1000,30])]}
    (x,y,w,h),reason=focus_crop(dict(width=1000,height=300),gt,gt>0,label)
    assert x<=950<x+w and y<=100<y+h and w==320 and h==240
    assert 'existing_GT_ink_pixel' in reason


def test_focus_crop_handles_no_text_overlap():
    from t4_restoration_review import focus_crop
    gt=np.zeros((300,1000),np.uint8);gt[150,20]=255
    crop,reason=focus_crop(dict(width=1000,height=300),gt,np.zeros_like(gt,bool),{'boxes_clean':[]})
    assert crop[0]<=20<crop[0]+crop[2] and reason=='GT_pixel_no_reference_text_overlap'
