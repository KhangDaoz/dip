"""Official T4 metadata selection and macro-score regression tests; not pixel benchmark."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import t4_official as t


def grid():
    rows = []
    for i in range(30):
        rows.append(dict(sample_id=f's{i}', source_id=f'id{i}', split='test',
            recipe_name='single_stamp', stamp_level=i//10+1, pen_color='none'))
        rows.append(dict(sample_id=f'p{i}', source_id=f'id{i}', split='test',
            recipe_name='single_pen', pen_level=i//15+1, pen_color='blue' if i%15<8 else 'black'))
    return rows


def test_exact_cohort():
    rows=grid()
    assert len(t.cohort(rows)) == 60
    assert t.cohort(rows) == t.cohort(list(reversed(rows)))


@pytest.mark.parametrize('mutation', ['missing','duplicate','split','source','level','color'])
def test_changed_cohort_rejected(mutation):
    rows=grid()
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows[-1]['sample_id']=rows[0]['sample_id']
    elif mutation=='split':rows[-1]['split']='val'
    elif mutation=='source':rows[-1]['source_id']='wrong'
    elif mutation=='level':rows[0]['stamp_level']=2
    else:rows[-1]['pen_color']='blue'
    with pytest.raises(ValueError):t.cohort(rows)


def test_macro_detection_is_per_image_not_pooled_pixels():
    base=dict(kind='stamp',pen_color='none',level=1,detector='stamp_candidate',empty_prediction=False)
    rows=[dict(base,source_id='a',iou=0.,precision=0.,recall=0.),
          dict(base,source_id='b',iou=1.,precision=1.,recall=1.)]
    summary=t.detection_summary(rows)
    assert len(summary)==2
    assert all(x['mean_iou']==.5 and x['n_sources']==2 for x in summary)


def test_negative_change_measures_color_and_text_proxy():
    clean=np.full((3,4,3),100,np.uint8);out=clean.copy();out[0,0]=255
    binary=np.zeros((3,4),np.uint8);binary[0,0]=255
    result=t.negative_change(clean,out,binary)
    assert result['changed_pixels']==result['changed_text_proxy_pixels']==1
    assert result['n_pixels']==12 and np.isclose(result['mae'],155/12)


def test_refuse_overwrite_before_preflight_or_pixels(tmp_path,monkeypatch):
    monkeypatch.setattr(t.d,'preflight',lambda *args:pytest.fail('Preflight called'))
    monkeypatch.setattr(t.d,'read_image',lambda *args,**kwargs:pytest.fail('Pixel read'))
    with pytest.raises(FileExistsError):t.run(tmp_path,tmp_path)


def test_black_pen_no_predicted_branch():
    row={'recipe_name':'single_pen','pen_color':'black'}
    gt=np.ones((2,2),np.uint8)*255
    assert list(t.r.mask_branches(row,gt,{}))==['gt_diagnostic']


def test_summary_tampering_rejected():
    with pytest.raises(ValueError,match='value mismatch'):
        t.compare_table([{'score':.5}],[{'score':'.6'}])


def test_summary_na_not_zero():
    with pytest.raises(ValueError,match='N/A mismatch'):
        t.compare_table([{'score':None}],[{'score':'0'}])
