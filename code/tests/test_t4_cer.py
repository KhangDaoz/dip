"""Metadata gates and known-answer arithmetic only; these tests are NOT OCR CER results."""
import copy
from pathlib import Path
import sys
import unicodedata

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import t4_cer as c


def examples():
    rows=[dict(crop_id=s,split=s,text_source_sha256='hash-'+s) for s in ('train','val','test')]
    review=dict(schema=1,reviewer='Unit-test human schema fixture, not real labels',items=[dict(
        crop_id=r['crop_id'],text_source_sha256=r['text_source_sha256'],review_status='human_confirmed',
        reviewed_text='Liều 1,5 mg.',reviewer='Unit-test human schema fixture, not real labels',review_note='') for r in rows])
    return rows,review


def test_missing_review_is_blocked_not_fake_cer():
    rows,_=examples();gate,selected=c.review_gate(rows,None)
    assert not gate['ready'] and selected==[] and gate['pending']=={'val':1,'test':1}


def test_test_requires_val_and_test_review():
    rows,review=examples();review['items'][1]['review_status']='pending'
    gate,_=c.review_gate(rows,review)
    assert not gate['ready']


def test_val_preparation_can_finish_before_test_labels():
    rows,review=examples();review['items'][2]['review_status']='pending'
    gate,selected=c.review_gate(rows,review,'val')
    assert gate['ready'] and [x['crop_id'] for x in selected]==['val']


@pytest.mark.parametrize('mutation', ['ai','hash','duplicate','unknown','reviewer','blank','exclusion','schema','status'])
def test_bad_reviews_rejected(mutation):
    rows,review=examples()
    if mutation=='ai':review['review_type']='ai_assisted_proposal'
    elif mutation=='hash':review['items'][2]['text_source_sha256']='changed'
    elif mutation=='duplicate':review['items'].append(copy.deepcopy(review['items'][0]))
    elif mutation=='unknown':review['items'][0]['crop_id']='unknown'
    elif mutation=='reviewer':review['items'][2]['reviewer']='different person'
    elif mutation=='blank':review['items'][2]['reviewed_text']='  \n'
    elif mutation=='exclusion':review['items'][2]['review_status']='excluded'
    elif mutation=='schema':review['schema']=2
    else:review['items'][2]['review_status']='ai_confirmed'
    with pytest.raises(ValueError):c.review_gate(rows,review)


def test_excluded_needs_reason_and_is_not_scored():
    rows,review=examples()
    rows.append(dict(crop_id='extra',split='test',text_source_sha256='hash-extra'))
    review['items'].append(dict(crop_id='extra',text_source_sha256='hash-extra',review_status='excluded',
        reviewed_text='',reviewer=review['reviewer'],review_note='Clipped, unreadable reference'))
    gate,selected=c.review_gate(rows,review)
    assert gate['ready'] and gate['excluded']=={'test':1}
    assert [x['crop_id'] for x in selected]==['test']


def test_nfc_whitespace_but_not_case_punctuation_or_numbers():
    assert c.score('Uống 1 viên',unicodedata.normalize('NFD','Uống  1 viên'))['errors']==0
    assert c.score('1,5 mg','1.5 mg')['errors']==1
    assert c.score('ABC','abc')['errors']==3


def test_empty_hypothesis_is_deletions_no_cap_on_cer():
    assert c.score('abc','')['cer']==1
    assert c.score('a','aaaa')['cer']==3
    with pytest.raises(ValueError):c.score('  ','anything')


def scored_rows():
    rows=[]
    for cid,text,hyp in [('short','a','b'),('long','abcdefghij','abcdefghij')]:
        for method in c.METHODS:
            rows.append(dict(crop_id=cid,source_id=cid,split='test',kind='stamp',pen_color='none',level=1,
                mask_source='predicted_stamp_candidate',method=method,**c.score(text,hyp)))
    return rows


def test_corpus_cer_total_errors_over_total_chars_not_macro():
    summary=c.summarize(scored_rows())
    assert len(summary)==10
    assert all(x['total_errors']==1 and x['total_reference_characters']==11 and x['n_lines']==2 and x['n_sources']==2 for x in summary)
    assert all(x['corpus_cer']==pytest.approx(1/11) and x['delta_vs_none']==0 for x in summary)


@pytest.mark.parametrize('mutation',['duplicate','missing','reference'])
def test_unpaired_comparison_rejected(mutation):
    rows=scored_rows()
    if mutation=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif mutation=='missing':rows.pop()
    else:rows[0]['reference_normalized']='changed'
    with pytest.raises(ValueError):c.summarize(rows)


def test_blocked_run_never_calls_ocr_or_decodes_or_creates_output(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'preflight',lambda *args:dict(status='BLOCKED_TRANSCRIPTS',transcripts={'blockers':['pending']}))
    monkeypatch.setattr(c,'real_ocr',lambda *args:pytest.fail('OCR must not run'))
    monkeypatch.setattr(c.official.d,'read_image',lambda *args,**kwargs:pytest.fail('Pixel decode must not run'))
    out=tmp_path/'cer'
    with pytest.raises(ValueError,match='blocked'):c.run(tmp_path,tmp_path,tmp_path,None,tmp_path,out)
    assert not out.exists()


def test_no_overwrite_before_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'preflight',lambda *args:pytest.fail('Should refuse existing directory first'))
    with pytest.raises(FileExistsError):c.run(tmp_path,tmp_path,tmp_path,None,tmp_path,tmp_path)


def test_changed_runtime_rejected(monkeypatch):
    baseline=dict(binary_sha256='old')
    monkeypatch.setattr(c.prep,'configure_runtime',lambda path:dict(binary_sha256='new'))
    with pytest.raises(ValueError,match='runtime differs'):c.runtime_gate('not-a-runtime',baseline)


def test_fixed_crop_identical_geometry_across_variants():
    image=np.full((15,20,3),200,np.uint8)
    row=dict(width=20,height=15,crop_bbox_xywh=[2,3,7,8])
    assert c.prep.same_crop(image,row).shape==(8,7)
    altered=image.copy();altered[3:11,2:9]=100
    assert np.all(c.prep.same_crop(altered,row)==100)


@pytest.mark.parametrize('color,branches',[('blue',['gt_diagnostic','predicted_blue_color']),('black',['gt_diagnostic'])])
def test_case_plan_keeps_black_gt_only(color,branches):
    rows=[dict(sample_id='s',kind='pen',pen_color=color)]
    cases=[dict(sample_id='s',mask_source=b) for b in branches]
    chosen,_=c.case_plan(cases,rows)
    assert len(chosen)==len(branches)


def test_case_plan_cannot_silently_drop_confirmed_crop():
    with pytest.raises(ValueError,match='branch'):
        c.case_plan([], [dict(sample_id='s',kind='stamp',pen_color='none')])


def test_coverage_preserves_no_eligible_region_and_exclusion():
    samples={'a':dict(sample_id='a',source_id='s1',split='test',recipe_name='single_stamp',pen_color='none'),
             'b':dict(sample_id='b',source_id='s2',split='test',recipe_name='single_pen',pen_color='black')}
    manifest=[dict(crop_id='line',sample_id='a')]
    review={'items':[dict(crop_id='line',review_status='excluded',review_note='unreadable',reviewer='fixture')]}
    coverage,excluded=c.coverage_rows(manifest,review,samples,'test')
    assert len(coverage)==2 and coverage[1]['no_eligible_text_region']
    assert not coverage[1]['predicted_detector_supported']
    assert excluded[0]['reason']=='unreadable'
