"""T4 inventory contracts, not detector/inpainting/OCR benchmark tests."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import t4_marks as t4


def manifest():
    rows=[]
    for split,n in t4.SPLITS.items():
        for i in range(n):
            for recipe,(variant,kind) in t4.RECIPES.items():
                rows.append(dict(source_id=f'{split}_{i}',sample_id=f'{split}_{i}_{variant}',
                                 split=split,recipe_name=recipe,variant_index=variant,
                                 applied_degradations=kind,angle_deg=0,perspective_strength=0,
                                 blur_level=0,shadow_level=0,glare_level=0,jpeg_noise_level=0,
                                 exposure_gain=1,stamp_level=int(kind=='stamp'),pen_level=int(kind=='pen')))
    return dict(version='v2',n_sources=100,samples=rows)


def test_full_source_membership():
    rows=t4.select_samples(manifest())
    assert len(rows)==200 and len({r['source_id'] for r in rows})==100


@pytest.mark.parametrize('error',['duplicate','cross_split','missing','extra_harm','geometry','wrong_recipe'])
def test_reject_confounded_or_incomplete_metadata(error):
    m=manifest()
    if error=='duplicate':m['samples'].append(copy.deepcopy(m['samples'][0]))
    if error=='cross_split':m['samples'][0]['split']='test'
    if error=='missing':m['samples'].pop()
    if error=='extra_harm':m['samples'][0]['pen_level']=1
    if error=='geometry':m['samples'][0]['angle_deg']=2
    if error=='wrong_recipe':m['samples'][0]['variant_index']=14
    with pytest.raises(ValueError):t4.select_samples(m)


def test_ocr_pool_overlap_and_unconfirmed_text():
    mask=np.zeros((20,30),bool);mask[5,5]=True
    ink=np.zeros_like(mask);ink[5,5]=True
    box=dict(id=1,label='drugname',bbox=[3,3,5,5],text='Drug 1.5mg')
    result=t4.inspect_box(box,mask,ink)
    assert result['bbox_valid'] and result['intersects_mark']
    assert result['overlapped_reference_ink_pixels']==1
    assert result['transcript_review']=='unverified_source_annotation'
    no_ink=t4.inspect_box(box,mask,np.zeros_like(mask))
    assert no_ink['intersects_mark'] and no_ink['overlapped_reference_ink_pixels']==0


@pytest.mark.parametrize('bbox',[[-1,2,3,4],[2,2,0,4],[28,1,5,5],[2.5,2,3,4]])
def test_invalid_boxes_reported_not_clipped_silently(bbox):
    result=t4.inspect_box(dict(id=1,bbox=bbox,text='x'),np.zeros((20,30),bool),np.zeros((20,30),bool))
    assert not result['bbox_valid'] and result['issue']


def test_identity_box_mapping_allows_added_metadata():
    clean=[dict(id=1,label='usage',text='1.5',bbox=[1,2,3,4])]
    mapped=[dict(clean[0],polygon=[[1,2],[4,2],[4,6],[1,6]],clipped=False,valid=True)]
    t4.validate_box_mapping(clean,mapped)
    mapped[0]['polygon'][0][0]=2
    with pytest.raises(ValueError):t4.validate_box_mapping(clean,mapped)


def test_identity_box_mapping_rejects_changed_transcript():
    clean=[dict(id=1,text='1.5',bbox=[1,2,3,4])]
    mapped=[dict(clean[0],text='15',polygon=[[1,2],[4,2],[4,6],[1,6]])]
    with pytest.raises(ValueError):t4.validate_box_mapping(clean,mapped)


@pytest.mark.parametrize('image',[np.zeros((5,5),float),np.ones((5,5),np.uint8),np.zeros((5,5,3),np.uint8)])
def test_binary_masks_reject_wrong_convention(image):
    with pytest.raises(ValueError):t4.binary_mask(image,(5,5))


def test_inventory_refuses_overwrite_before_reading(tmp_path,monkeypatch):
    (tmp_path/t4.FILES[0]).write_text('original')
    def fail(*args):raise AssertionError('inventory called')
    monkeypatch.setattr(t4,'inventory',fail)
    with pytest.raises(FileExistsError):t4.main(['--output-dir',str(tmp_path)])
    assert (tmp_path/t4.FILES[0]).read_text()=='original'


def test_outside_asset_rejected(tmp_path):
    with pytest.raises(ValueError):t4.asset(tmp_path,'../outside.png')
