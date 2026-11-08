"""Deterministic reduced CER cohort; metadata-only selection before OCR."""
from collections import Counter,defaultdict
import hashlib
import base64
import json
from pathlib import Path

SEED='MedTrackAI-T4-CER-reduced-v1-20261104'


def build(manifest,manifest_sha):
    groups=defaultdict(list)
    for row in manifest:
        if row['split'] in ('val','test'):groups[row['sample_id']].append(row)
    rows=[]
    for sid,group in sorted(groups.items()):
        ranked=sorted(group,key=lambda r:(hashlib.sha256((SEED+'|'+r['crop_id']).encode()).hexdigest(),r['crop_id']))
        rows.extend(dict(r,reserve_rank=i+1) for i,r in enumerate(ranked))
    primary=[r for r in rows if r['reserve_rank']==1]
    if Counter(r['split'] for r in primary)!={'val':15,'test':47}:raise ValueError('Reduced cohort cardinality changed')
    return dict(schema=1,cohort_id='t4_cer_reduced_v1',seed=SEED,parent_manifest_sha256=manifest_sha,
        policy='One crop per eligible sample image; SHA256(seed|crop_id), ascending. Reserve only after all earlier crops human-excluded with legibility reasons. No OCR-based replacement.',
        primary_counts=dict(Counter(r['split'] for r in primary)),candidates=rows)


def validate(lock,manifest,manifest_sha):
    if lock!=build(manifest,manifest_sha):raise ValueError('Reduced cohort/seed/rank/parent changed')


def resolve(lock,review):
    """Resolve predetermined primary/reserve chain; never consult OCR or AI decisions."""
    items={x['crop_id']:x for x in review['items']} if review else {}
    groups=defaultdict(list)
    for row in lock['candidates']:groups[row['sample_id']].append(row)
    active=[];excluded=[]
    for sid,rows in sorted(groups.items()):
        chosen=None
        for row in rows:
            item=items.get(row['crop_id'],{})
            if item.get('review_status')=='excluded':
                if not str(item.get('review_note','')).strip():raise ValueError('Reserve requires exclusion reason')
                excluded.append(row);continue
            chosen=row;break
        for row in rows:
            item=items.get(row['crop_id'],{})
            if item.get('review_status')=='human_confirmed' and (chosen is None or row['crop_id']!=chosen['crop_id']):
                raise ValueError('Reserve confirmed before earlier candidates excluded / multiple confirmed boxes')
        if chosen:active.append(chosen)
    return active,excluded


def embedded_crops(lock,folder):
    images={}
    for row in lock['candidates']:
        path=(Path(folder)/row['clean_crop']).resolve()
        if not path.is_relative_to(Path(folder).resolve()):raise ValueError('Out-of-root crop')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['clean_crop_sha256']:raise ValueError('Crop checksum mismatch')
        images[row['crop_id']]='data:image/png;base64,'+base64.b64encode(data).decode('ascii')
    return images


def page(lock,lock_sha,suggestions,images=None):
    display=[dict(r,clean_crop=(images or {}).get(r['crop_id'],r['clean_crop'])) for r in lock['candidates']]
    payload=json.dumps(display,ensure_ascii=False).replace('<','\\u003c')
    hints=json.dumps(suggestions,ensure_ascii=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>T4 — duyệt 62 box</title>
<style>body{font:17px Arial;margin:24px;max-width:1250px}header{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #aaa}article{border:1px solid #aaa;padding:14px;margin:18px 0}.img{overflow:auto;background:#eee}img{max-width:none}textarea{width:98%;min-height:65px;font:18px Arial}button,input,select{padding:8px;margin:5px}.hint{background:#fff4cb;padding:10px}</style>
<header><b>T4 — tập CER nhỏ: 15 val + 47 test</b><p>Đọc ảnh sạch, giữ đúng chữ/số/dấu/đơn vị. Gợi ý AI chưa phải nhãn xác nhận. Chỉ loại vì crop không đọc được/cắt chữ/nhiều dòng/ẩn thông tin; không loại vì OCR khó hoặc dấu che nhiều.</p>
Tên người duyệt <input id="who"><select id="split"><option>val</option><option>test</option></select>
<button id="save">Tải JSON đã duyệt</button><label>Nạp JSON để tiếp tục <input type="file" id="load" accept=".json"></label><div id="stats"></div></header><main id="cards"></main>
<script>const rows='''+payload+''',hints='''+hints+''',cohortSha='''+json.dumps(lock_sha)+''';
for(const r of rows){r.status='pending';r.text=r.text_source;r.note=''}
const groups=new Map();for(const r of rows){if(!groups.has(r.sample_id))groups.set(r.sample_id,[]);groups.get(r.sample_id).push(r)}
function active(){return [...groups.values()].map(g=>g.find(r=>r.status!=='excluded')).filter(Boolean)}
function render(){const split=document.getElementById('split').value,list=active().filter(r=>r.split===split),cards=document.getElementById('cards');cards.replaceChildren();document.getElementById('stats').textContent=`${split}: ${list.filter(r=>r.status==='human_confirmed').length}/${list.length} box đang chọn đã xác nhận. Chỉ một box/ảnh; loại sẽ hiện dự phòng theo thứ tự đã khóa. Nhớ tải JSON để lưu.`;
for(const r of list){const a=document.createElement('article'),h=document.createElement('h3');h.textContent=`${r.crop_id} | ${r.kind}/${r.pen_color} mức ${r.level} | thứ tự ${r.reserve_rank}`;a.append(h);const div=document.createElement('div');div.className='img';const im=document.createElement('img');im.src=r.clean_crop;div.append(im);a.append(div);const p=document.createElement('p');p.textContent='Nhãn nguồn: '+r.text_source;a.append(p);
if(hints[r.crop_id]){const hint=document.createElement('p');hint.className='hint';hint.textContent='AI đề xuất (chưa xác nhận): '+hints[r.crop_id].proposed_text+' | '+hints[r.crop_id].decision+' | '+hints[r.crop_id].note;a.append(hint)}
const text=document.createElement('textarea');text.value=r.text;text.oninput=()=>r.text=text.value;a.append(text);const note=document.createElement('input');note.placeholder='Lý do loại / ghi chú';note.value=r.note;note.oninput=()=>r.note=note.value;a.append(note);
const ok=document.createElement('button');ok.textContent=r.status==='human_confirmed'?'Bỏ xác nhận':'Tôi đã đọc và xác nhận';ok.onclick=()=>{if(!r.text.trim()){alert('Không xác nhận chữ trống');return}r.status=r.status==='human_confirmed'?'pending':'human_confirmed';render()};a.append(ok);
const ex=document.createElement('button');ex.textContent='Không đọc được — lấy dự phòng';ex.onclick=()=>{if(!r.note.trim()){alert('Nhập lý do loại trước');return}r.status='excluded';render()};a.append(ex);cards.append(a)}}
document.getElementById('split').onchange=render;
document.getElementById('save').onclick=()=>{const who=document.getElementById('who').value.trim();if(!who){alert('Nhập tên người duyệt');return}const data={schema:1,review_type:'human_review',reviewer:who,cohort_id:'t4_cer_reduced_v1',cohort_sha256:cohortSha,exported_at:new Date().toISOString(),items:rows.map(r=>({crop_id:r.crop_id,text_source_sha256:r.text_source_sha256,review_status:r.status,reviewed_text:r.text,review_note:r.note,reviewer:r.status==='pending'?'':who}))};const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json;charset=utf-8'})),a=document.createElement('a');a.href=u;a.download='t4_transcript_review_reduced_v1.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
document.getElementById('load').onchange=async e=>{try{const f=JSON.parse(await e.target.files[0].text());if(f.schema!==1||f.cohort_sha256!==cohortSha||f.review_type!=='human_review')throw Error('Không đúng bản duyệt tập nhỏ');const by=new Map(rows.map(r=>[r.crop_id,r])),seen=new Set();for(const x of f.items){const r=by.get(x.crop_id);if(!r||seen.has(x.crop_id)||x.text_source_sha256!==r.text_source_sha256||!['pending','excluded','human_confirmed'].includes(x.review_status)||typeof x.reviewed_text!=='string'||(x.review_status==='excluded'&&!x.review_note.trim()))throw Error('ID/hash/trạng thái sai');seen.add(x.crop_id)}for(const x of f.items){const r=by.get(x.crop_id);r.status=x.review_status;r.text=x.reviewed_text;r.note=x.review_note||''}document.getElementById('who').value=f.reviewer||'';render()}catch(err){alert(err.message)}};render();</script></html>'''


def main():
    import t4_ocr_prepare as prep
    folder=prep.audit.REPO_DIR/'results/t4/ocr_preparation'
    manifest=json.loads((folder/'crop_manifest.json').read_text(encoding='utf-8'))
    lock=build(manifest,prep.audit.sha(folder/'crop_manifest.json'))
    dest=folder/'cohort_reduced_v1.json'
    if dest.exists():raise FileExistsError('Reduced cohort already frozen')
    suggestions={};proposal=folder/'ai_val_review.json'
    if proposal.exists():
        source=json.loads(proposal.read_text(encoding='utf-8'))
        if source.get('review_type')!='ai_assisted_proposal' or source.get('manifest_sha256')!=lock['parent_manifest_sha256']:raise ValueError('AI proposal parent mismatch')
        by={x['crop_id']:x for x in manifest}
        for item in source['items']:
            row=by[item['crop_id']]
            if item['original_text_sha256']!=row['text_source_sha256'] or item['clean_crop_sha256']!=row['clean_crop_sha256']:raise ValueError('AI proposal source hash mismatch')
            if item['crop_id'] in suggestions:raise ValueError('Duplicate AI proposal')
            suggestions[item['crop_id']]=item
    dest.write_text(json.dumps(lock,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'review_reduced_v1.html').write_text(page(lock,prep.audit.sha(dest),suggestions,embedded_crops(lock,folder)),encoding='utf-8')
    print(json.dumps(lock['primary_counts']))


if __name__=='__main__':main()
