"""Reduced-cohort order and reserve rules, without OCR or real transcript claims."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import t4_cer_subset as s
import t4_cer as cer


def lock():
    rows=[]
    for split,n in [('val',15),('test',47)]:
        for i in range(n):
            for j in range(2):rows.append(dict(sample_id=f'{split}_{i}',crop_id=f'{split}_{i}_{j}',split=split,
                text_source_sha256=f'hash{i}_{j}',clean_crop=f'crops/{split}_{i}_{j}.png'))
    return rows,s.build(rows,'parent-hash')


def test_order_independent_and_one_per_image():
    rows,l=lock()
    assert s.build(list(reversed(rows)),'parent-hash')==l
    active,_=s.resolve(l,None)
    assert len(active)==62 and len({x['sample_id'] for x in active})==62
    assert all(x['reserve_rank']==1 for x in active)


def test_lock_tamper_rejected():
    rows,l=lock();l['seed']='other'
    with pytest.raises(ValueError):s.validate(l,rows,'parent-hash')


def test_reserve_cannot_be_confirmed_before_primary_excluded():
    _,l=lock();row=next(x for x in l['candidates'] if x['reserve_rank']==2)
    with pytest.raises(ValueError,match='Reserve'):
        s.resolve(l,{'items':[dict(crop_id=row['crop_id'],review_status='human_confirmed')]})


def test_reserve_after_documented_exclusion():
    _,l=lock();first=l['candidates'][0]
    active,ex=s.resolve(l,{'items':[dict(crop_id=first['crop_id'],review_status='excluded',review_note='clipped')]})
    replacement=next(x for x in active if x['sample_id']==first['sample_id'])
    assert replacement['reserve_rank']==2 and len(ex)==1 and len(active)==62


def test_exclusion_without_reason_rejected():
    _,l=lock()
    with pytest.raises(ValueError,match='reason'):
        s.resolve(l,{'items':[dict(crop_id=l['candidates'][0]['crop_id'],review_status='excluded')]})


def test_pending_reserves_do_not_block_active_review():
    _,l=lock();active,_=s.resolve(l,None)
    review=dict(schema=1,reviewer='fixture',items=[dict(crop_id=x['crop_id'],text_source_sha256=x['text_source_sha256'],
        review_status='human_confirmed',reviewed_text='fixture',reviewer='fixture',review_note='') for x in active])
    gate,selected=cer.review_gate(active,review)
    assert gate['ready'] and len(selected)==47


def test_all_unreadable_preserves_exclusion_chain_not_fake_score():
    _,l=lock();sid=l['candidates'][0]['sample_id']
    review={'items':[dict(crop_id=x['crop_id'],review_status='excluded',review_note='unreadable') for x in l['candidates'] if x['sample_id']==sid]}
    active,excluded=s.resolve(l,review)
    assert len(active)==61 and len(excluded)==2


def test_html_escapes_embedded_data_and_never_preconfirms():
    _,l=lock();l['candidates'][0]['text_source']='</script>bad'
    html=s.page(l,'hash',{})
    assert '\\u003c/script>' in html and "r.status='pending'" in html


def test_embedded_crop_validates_bytes_and_removes_relative_dependency(tmp_path):
    import hashlib
    import base64
    data=b'known PNG bytes fixture'
    (tmp_path/'crop.png').write_bytes(data)
    row=dict(crop_id='one',clean_crop='crop.png',clean_crop_sha256=hashlib.sha256(data).hexdigest())
    images=s.embedded_crops({'candidates':[row]},tmp_path)
    assert base64.b64decode(images['one'].split(',',1)[1])==data
    html=s.page({'candidates':[row]},'hash',{},images)
    assert 'data:image/png;base64,' in html
    (tmp_path/'crop.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):s.embedded_crops({'candidates':[row]},tmp_path)
