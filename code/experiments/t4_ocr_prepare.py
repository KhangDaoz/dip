"""T4 session 4: fixed crops, blind transcript review, real TRAIN/VAL OCR smoke."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import t4_marks as audit
from common.io import read_image
from ocr.engines import ocr_line

PAD=2


def crop_box(bbox,width,height,pad=PAD):
    if (not isinstance(bbox,list) or len(bbox)!=4 or any(type(v)is not int for v in bbox)
            or type(pad)is not int or pad<0):raise ValueError('Expected integer xywh/padding')
    x,y,w,h=bbox
    if x<0 or y<0 or w<=0 or h<=0 or x+w>width or y+h>height:raise ValueError('Annotation outside frame')
    x0,y0=max(0,x-pad),max(0,y-pad);x1,y1=min(width,x+w+pad),min(height,y+h+pad)
    return [x0,y0,x1-x0,y1-y0]


def same_crop(image,record):
    """All future restoration variants must use this same stored rectangle."""
    if image.dtype!=np.uint8 or image.shape[:2]!=(record['height'],record['width']):
        raise ValueError('Variant frame/type differs from frozen crop frame')
    if image.ndim==3 and image.shape[2]==3:gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    elif image.ndim==2:gray=image
    else:raise ValueError('Expected BGR or grayscale image')
    x,y,w,h=record['crop_bbox_xywh']
    if x<0 or y<0 or w<=0 or h<=0 or x+w>gray.shape[1] or y+h>gray.shape[0]:
        raise ValueError('Crop outside frame')
    return gray[y:y+h,x:x+w].copy()


def candidates(root,inventory_path):
    # Re-audit annotations/mask overlap, not the result of any detector/OCR.
    report,_,pool=audit.inventory(root)
    prior=json.loads(Path(inventory_path).read_text(encoding='utf-8'))
    if (prior['assets_sha256']!=report['assets_sha256'] or prior['manifest_sha256']!=report['manifest_sha256']
            or prior['inventory_script_sha256']!=audit.sha(Path(audit.__file__))):
        raise ValueError('Session-1 dataset/evidence changed')
    samples=audit.select_samples(json.loads((root/'manifest_v2.json').read_text(encoding='utf-8')))
    by_id={s['sample_id']:s for s in samples};labels={}
    rows=[]
    for entry in pool:
        if not (entry['bbox_valid'] and entry['transcript_present'] and entry['overlapped_reference_ink_pixels']>0):continue
        source=by_id[entry['sample_id']]
        label=labels.setdefault(source['sample_id'],json.loads((root/source['rel_gt_path']).read_text(encoding='utf-8')))
        box=next(b for b in label['boxes_clean'] if b['id']==entry['box_id'])
        crop_id=source['sample_id']+'__box'+str(box['id'])
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',crop_id):raise ValueError('Unsafe crop ID')
        rows.append(dict(crop_id=crop_id,sample_id=source['sample_id'],source_id=source['source_id'],
            split=source['split'],kind=entry['kind'],level=entry['level'],pen_color=entry['pen_color'],
            box_id=box['id'],label=box['label'],annotation_bbox_xywh=box['bbox'],
            crop_bbox_xywh=crop_box(box['bbox'],source['width'],source['height']),
            width=source['width'],height=source['height'],padding=PAD,
            text_source=box['text'],text_source_sha256=entry['text_sha256'],
            review_status='pending',reviewed_text=box['text'],reviewer='',review_note='',
            overlapped_reference_ink_pixels=entry['overlapped_reference_ink_pixels'],
            rel_clean_path=source['rel_clean_path'],rel_degraded_path=source['rel_degraded_path'],
            rel_label_path=source['rel_gt_path'],
            clean_crop='crops/'+crop_id+'_clean.png',degraded_crop='crops/'+crop_id+'_degraded.png'))
    rows.sort(key=lambda r:({'train':0,'val':1,'test':2}[r['split']],r['source_id'],r['sample_id'],
                            r['annotation_bbox_xywh'][1],r['annotation_bbox_xywh'][0],r['box_id']))
    if len({r['crop_id'] for r in rows})!=len(rows):raise ValueError('Duplicate crop ID')
    return rows,report


def configure_runtime(path):
    runtime=json.loads(Path(path).read_text(encoding='utf-8'))
    binary=Path(runtime['binary'])
    if audit.sha(binary)!=runtime['binary_sha256']:raise ValueError('OCR binary changed')
    for model in runtime['models'].values():
        if audit.sha(model['path'])!=model['sha256']:raise ValueError('OCR model changed')
    modeldir=Path(runtime['models']['vie']['path']).parent
    if Path(runtime['models']['eng']['path']).parent!=modeldir:raise ValueError('Different model directories')
    langs=subprocess.run([str(binary),'--list-langs','--tessdata-dir',str(modeldir)],capture_output=True,text=True,check=True).stdout
    if not {'vie','eng'}<=set(langs.splitlines()):raise ValueError('Vietnamese/English models unavailable')
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd=str(binary)
    os.environ['TESSDATA_PREFIX']=str(modeldir)  # This process only; no system PATH mutation.
    runtime.update(language='vie+eng',wrapper='ocr.engines.ocr_line',wrapper_sha256=audit.sha(audit.REPO_DIR/'code/src/ocr/engines.py'),
                   psm=7,oem=3,oem_note='wrapper passes --psm 7; OEM default 3 auto, official fast models are LSTM-only',
                   preprocessing='original resolution grayscale, padding2, no resize/binarization/CLAHE',
                   normalization='Unicode NFC + collapse whitespace; preserve case/punctuation/decimal separators/units',
                   pytesseract_version=importlib.metadata.version('pytesseract'))
    runtime['runtime_dll_sha256']={f.name:audit.sha(f) for f in binary.parent.glob('*.dll')}
    return runtime


def review_html(rows):
    data=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Duyệt transcript T4</title>
<style>body{font:16px Arial;margin:24px;max-width:1300px}header{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #ccc}article{border:1px solid #ccc;border-radius:8px;margin:16px 0;padding:14px}.image{overflow:auto;background:#eee;padding:8px}img{max-width:none}textarea{width:98%;min-height:65px;font:17px Arial}button,select,input{padding:8px;margin:4px}small{color:#555}</style>
<header><b>T4 — duyệt trên ảnh sạch, không xem dự đoán OCR</b><br>
<small>Ảnh hiển thị ở kích thước gốc, cuộn ngang nếu cần. Sửa transcript theo ảnh sạch; giữ nguyên số, đơn vị, dấu chấm/phẩy. Chỉ xác nhận khi đã đọc rõ. Nếu bị che thông tin/crop nhiều dòng/không đọc được, loại và ghi lý do.</small><br>
<label>Người duyệt <input id="reviewer"></label><label>Tập <select id="split"><option>val</option><option>train</option><option>test</option><option value="all">Tất cả</option></select></label>
<button id="prev">Trang trước</button><button id="next">Trang sau</button><button id="save">Tải JSON kết quả duyệt</button><label>Nạp bản đã lưu <input id="load" type="file" accept=".json"></label><br><small>Lưu JSON định kỳ; dùng Nạp bản đã lưu để tiếp tục sau khi đóng trang.</small><div id="stats"></div></header><main id="cards"></main>
<script>const rows='''+data+''';let page=0;const size=20;
const cards=document.getElementById('cards'),split=document.getElementById('split');
function render(){const list=rows.filter(r=>split.value==='all'||r.split===split.value);page=Math.max(0,Math.min(page,Math.ceil(list.length/size)-1));cards.replaceChildren();
document.getElementById('stats').textContent=`${list.length} box trong tập này; trang ${page+1}/${Math.max(1,Math.ceil(list.length/size))}. Đã xác nhận ${rows.filter(r=>r.review_status==='human_confirmed').length}/${rows.length}.`;
for(const row of list.slice(page*size,(page+1)*size)){const a=document.createElement('article'),h=document.createElement('h3');h.textContent=`${row.crop_id} | ${row.split} | ${row.kind} mức ${row.level}`;a.append(h);
const div=document.createElement('div');div.className='image';const im=document.createElement('img');im.src=row.clean_crop;im.alt='Crop sạch dùng để duyệt transcript';div.append(im);a.append(div);
const original=document.createElement('p');original.textContent='Nhãn nguồn: '+row.text_source;a.append(original);
const txt=document.createElement('textarea');txt.value=row.reviewed_text;txt.oninput=()=>row.reviewed_text=txt.value;a.append(txt);
const status=document.createElement('select');for(const [v,t] of [['pending','Chưa duyệt'],['human_confirmed','Tôi đã kiểm tra và xác nhận'],['excluded','Loại khỏi CER (ghi lý do)']]){const o=document.createElement('option');o.value=v;o.textContent=t;status.append(o)}status.value=row.review_status;status.onchange=()=>{row.review_status=status.value};a.append(status);
const note=document.createElement('input');note.placeholder='Ghi chú / lý do loại';note.value=row.review_note;note.oninput=()=>row.review_note=note.value;a.append(note);cards.append(a)}}
split.onchange=()=>{page=0;render()};document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};
document.getElementById('load').onchange=async(e)=>{try{const saved=JSON.parse(await e.target.files[0].text());if(saved.schema!==1||!Array.isArray(saved.items))throw Error('Định dạng JSON không hợp lệ');const byId=new Map(rows.map(r=>[r.crop_id,r]));const seen=new Set();for(const item of saved.items){const row=byId.get(item.crop_id);if(!row||seen.has(item.crop_id)||item.text_source_sha256!==row.text_source_sha256)throw Error('ID/hash không khớp bộ crop này');seen.add(item.crop_id);if(!['pending','human_confirmed','excluded'].includes(item.review_status)||typeof item.reviewed_text!=='string')throw Error('Trạng thái/text không hợp lệ')}for(const item of saved.items){const row=byId.get(item.crop_id);row.review_status=item.review_status;row.reviewed_text=item.reviewed_text;row.review_note=item.review_note||''}document.getElementById('reviewer').value=saved.reviewer||'';render()}catch(err){alert('Không nạp được: '+err.message)}};
document.getElementById('save').onclick=()=>{const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('Điền tên người duyệt trước khi lưu.');return}if(rows.some(r=>r.review_status==='excluded'&&!r.review_note.trim())){alert('Cần lý do cho box bị loại.');return}if(rows.some(r=>r.review_status==='human_confirmed'&&!r.reviewed_text.trim())){alert('Transcript xác nhận không được trống.');return}
const now=new Date().toISOString();const data={schema:1,task:'T4',reviewer,exported_at:now,items:rows.map(r=>({crop_id:r.crop_id,text_source_sha256:r.text_source_sha256,review_status:r.review_status,reviewed_text:r.reviewed_text,review_note:r.review_note,reviewer:r.review_status==='pending'?'':reviewer}))};
const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='t4_transcript_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();</script></html>'''


def run(root,inventory_path,runtime_path,output):
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError('Preparation output already exists')
    rows,report=candidates(root,inventory_path)
    runtime=configure_runtime(runtime_path)
    smoke_ids=[r['crop_id'] for split in ('train','val') for r in [x for x in rows if x['split']==split][:3]]
    output.mkdir(parents=True);(output/'crops').mkdir()
    config=dict(phase='OCR_PREPARATION_NOT_CER',padding=PAD,selection='all valid annotated boxes with injected GT support overlapping clean-binary ink',
        n_candidates=len(rows),counts=dict(Counter(r['split'] for r in rows)),runtime=runtime,
        engine_comparison_note='Khang reports Tesseract5.5.4; this local portable build5.5.3 is not the same binary',
        inventory_sha256=audit.sha(inventory_path),manifest_sha256=report['manifest_sha256'],input_sha256=report['assets_sha256'],
        script_sha256=audit.sha(Path(__file__)),smoke_ids=smoke_ids,smoke_test_split=False,
        transcript_status='all pending; source text is not OCR output',test_cer_run=False,
        future_methods_must_use_same_stored_crop=True,complete=False)
    with (output/'config.json').open('x',encoding='utf-8') as f:json.dump(config,f,indent=2)
    config_hash=audit.sha(output/'config.json');current=None
    for row in rows:
        if current!=row['sample_id']:
            current=row['sample_id'];clean=read_image(root/row['rel_clean_path']);degraded=read_image(root/row['rel_degraded_path'])
        for field,image in [('clean_crop',clean),('degraded_crop',degraded)]:
            crop=same_crop(image,row)
            if not cv2.imwrite(str(output/row[field]),crop):raise OSError('Crop write failed')
            row[field+'_sha256']=audit.sha(output/row[field])
    with (output/'crop_manifest.json').open('x',encoding='utf-8') as f:json.dump(rows,f,ensure_ascii=False,indent=2)
    (output/'review.html').write_text(review_html(rows),encoding='utf-8')
    with (output/'review_template.json').open('x',encoding='utf-8') as f:
        json.dump(dict(schema=1,items=[dict(crop_id=r['crop_id'],text_source_sha256=r['text_source_sha256'],
            review_status='pending',reviewed_text=r['text_source'],reviewer='',review_note='') for r in rows]),f,ensure_ascii=False,indent=2)
    # Genuine OCR smoke only; text_source is never replaced by these predictions.
    smoke=[]
    for row in rows:
        if row['crop_id'] not in smoke_ids:continue
        for field in ('clean_crop','degraded_crop'):
            text,confidence=ocr_line(read_image(output/row[field],gray=True),engine='tesseract',lang='vie+eng')
            smoke.append(dict(crop_id=row['crop_id'],split=row['split'],variant=field,text_pred=text,confidence=confidence,
                              crop_sha256=row[field+'_sha256'],is_transcript_gt=False))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',32)
    phrase='Uống 1 viên buổi sáng. Hàm lượng 500 mg.'
    synthetic=Image.new('RGB',(850,70),'white');ImageDraw.Draw(synthetic).text((12,12),phrase,font=font,fill='black')
    synthetic.save(output/'vietnamese_runtime_probe.png')
    probe_text,probe_conf=ocr_line(np.array(synthetic)[:,:,::-1].copy(),engine='tesseract',lang='vie+eng')
    with (output/'ocr_smoke.json').open('x',encoding='utf-8') as f:
        json.dump(dict(engine='tesseract',language='vie+eng',model_family='tessdata_fast',items=smoke,
            synthetic_runtime_probe=dict(text_rendered=phrase,text_pred=probe_text,confidence=probe_conf),
            test_predictions_run=False,cer_computed=False,not_for_transcript_review=True),f,ensure_ascii=False,indent=2)
    if audit.sha(output/'config.json')!=config_hash:raise ValueError('Config changed')
    with (output/'completion.json').open('x',encoding='utf-8') as f:
        json.dump(dict(complete=True,candidates=len(rows),png_crops=len(rows)*2,all_review_status_pending=True,
            genuine_ocr_smoke_calls=len(smoke)+1,test_ocr_run=False,cer_computed=False,config_sha256=config_hash,
            manifest_sha256=audit.sha(output/'crop_manifest.json'),review_html_sha256=audit.sha(output/'review.html'),
            smoke_sha256=audit.sha(output/'ocr_smoke.json')),f,indent=2)
    print(json.dumps(dict(candidates=len(rows),splits=config['counts'],genuine_ocr_calls=len(smoke)+1,test_cer_run=False)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=audit.REPO_DIR/'data/5_synthetic')
    parser.add_argument('--inventory',type=Path,default=audit.REPO_DIR/'results/t4/step1_inventory.json')
    parser.add_argument('--runtime',type=Path,default=audit.REPO_DIR/'code/.venv/t4_ocr_setup/runtime.json')
    parser.add_argument('--output-dir',type=Path,default=audit.REPO_DIR/'results/t4/ocr_preparation')
    a=parser.parse_args(argv);run(a.data_dir,a.inventory,a.runtime,a.output_dir)


if __name__=='__main__':main()
