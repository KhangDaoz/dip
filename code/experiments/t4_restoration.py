"""T4 session 3: fixed-radius TRAIN/VAL restoration pilot, no test or OCR."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform

import cv2
import numpy as np
from PIL import Image, ImageDraw

import t4_detection as det
from cleaning.inpaint import remove_marks

METHODS=('none','telea','ns','fill_white')
RADIUS=3


def choose_samples(all_samples):
    selected=det.select_train(all_samples)+[r for r in all_samples if r['split']=='val']
    if (len(selected)!=27 or len({r['sample_id'] for r in selected})!=27
            or any(r['split'] not in ('train','val') for r in selected)):
        raise ValueError('Expected 7 TRAIN sanity + 20 VAL samples, never TEST')
    return sorted(selected,key=lambda r:(r['split'],r['source_id'],r['variant_index']))


def mask_branches(row,gt,masks):
    kind=det.audit.RECIPES[row['recipe_name']][1]
    result={'gt_diagnostic':gt}
    if kind=='stamp':result['predicted_stamp_candidate']=masks['stamp_candidate']
    elif row['pen_color']=='blue':result['predicted_blue_color']=masks['blue_color']
    # No full-page black mask, no fake zero prediction for unsupported black pen.
    return result


def apply_method(image,mask,method):
    if method not in METHODS:raise ValueError('Invalid restoration method')
    original=det.array_sha(image);mask_hash=det.array_sha(mask)
    selected=det.mask_pixels(mask)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or selected.shape!=image.shape[:2]:
        raise ValueError('Expected BGR uint8 and same-frame mask')
    out=image.copy() if method=='none' else remove_marks(image,mask,method=method,radius=RADIUS)
    if (out.dtype!=image.dtype or out.shape!=image.shape or np.shares_memory(out,image)
            or not np.array_equal(out[~selected],image[~selected])
            or det.array_sha(image)!=original or det.array_sha(mask)!=mask_hash):
        raise ValueError('Restoration violated input/frame/outside-mask invariants')
    if not selected.any() and not np.array_equal(out,image):raise ValueError('Empty mask not a no-op')
    return out


def region_scores(clean_gray,output_gray,roi):
    if (clean_gray.dtype!=np.uint8 or output_gray.dtype!=np.uint8 or clean_gray.ndim!=2
            or clean_gray.shape!=output_gray.shape or roi.dtype!=bool or roi.shape!=clean_gray.shape):
        raise ValueError('Metric frames/types disagree')
    n=int(np.count_nonzero(roi))
    if not n:return dict(n_pixels=0,mae=None,mse=None,psnr_db=None)
    errors=output_gray[roi].astype(np.float64)-clean_gray[roi].astype(np.float64)
    mse=float(np.mean(errors**2))
    return dict(n_pixels=n,mae=float(np.mean(np.abs(errors))),mse=mse,
                psnr_db=float('inf') if mse==0 else float(10*np.log10(255.**2/mse)))


def evaluate(clean,output,gt,clean_binary):
    roi=det.mask_pixels(gt);ink=det.mask_pixels(clean_binary)
    gray_clean=cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY);gray_out=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
    scores={}
    for prefix,region in [('roi',roi),('text_proxy',roi&ink)]:
        scores.update({prefix+'_'+k:v for k,v in region_scores(gray_clean,gray_out,region).items()})
    return scores


def preflight(root,inventory_file,detection_config):
    selected_train,_=det.preflight(root,inventory_file)
    prior=json.loads(Path(detection_config).read_text(encoding='utf-8'))
    if prior['selected_samples']!=selected_train:raise ValueError('TRAIN selection changed')
    for name,digest in prior['code_sha256'].items():
        if det.audit.sha(det.audit.REPO_DIR/name)!=digest:raise ValueError('Detector code changed since session 2')
    manifest=json.loads((root/'manifest_v2.json').read_text(encoding='utf-8'))
    rows=choose_samples(det.audit.select_samples(manifest))
    inventory=json.loads(Path(inventory_file).read_text(encoding='utf-8'))
    delivered={}
    for line in (root/'checksums_v2.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split(maxsplit=1);delivered[name.lstrip('*')]=digest
    evidence={}
    for row in rows:
        for key in ('rel_degraded_path','rel_clean_path','rel_clean_binary_path','rel_gt_path',
                    'rel_stamp_mask_path','rel_pen_mask_path','rel_glare_mask_path'):
            relative=row[key];actual=det.audit.sha(det.audit.asset(root,relative))
            if delivered.get(relative)!=actual or inventory['assets_sha256'].get(relative)!=actual:
                raise ValueError('Input checksum changed: '+relative)
            evidence[relative]=actual
    return rows,evidence


def preview_ids(samples):
    chosen={}
    for row in sorted(samples,key=lambda r:r['source_id']):
        kind=det.audit.RECIPES[row['recipe_name']][1]
        cell=(row['split'],kind,row[kind+'_level'],row['pen_color'])
        if (kind=='stamp' and row['stamp_level'] in (1,3)) or (kind=='pen' and row['pen_level']==2):
            chosen.setdefault(cell,row['sample_id'])
    return sorted(chosen.values())


def crop_region(row,gt,text_mask,label):
    candidates=[]
    for box in label['boxes_clean']:
        details=det.audit.inspect_box(box,gt>0,text_mask)
        if details['bbox_valid'] and details['overlapped_reference_ink_pixels']:
            candidates.append((-details['overlapped_reference_ink_pixels'],box['id'],box['bbox']))
    if candidates:
        _,_,(x,y,w,h)=min(candidates);cx,cy=x+w//2,y+h//2;reason='most_GT_overlapped_ink_box_then_id'
    else:
        yy,xx=np.where(gt>0);cx,cy=int((xx.min()+xx.max())/2),int((yy.min()+yy.max())/2)
        reason='no_text_overlap_GT_bbox_center'
    w=min(320,row['width']);h=min(240,row['height'])
    x=max(0,min(row['width']-w,cx-w//2));y=max(0,min(row['height']-h,cy-h//2))
    return (x,y,w,h),reason


def save_preview(output,prefix,clean,outputs,mask,crop):
    x,y,w,h=crop
    sheet=Image.new('RGB',(6*w,h+30),'white');draw=ImageDraw.Draw(sheet)
    panels=[('clean reference',clean)]+[(method,outputs[method]) for method in METHODS]
    panels.append(('erasure mask (inverted)',255-mask))
    for i,(name,array) in enumerate(panels):
        if array.ndim==3:array=cv2.cvtColor(array,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(array).convert('RGB')
        sheet.paste(image.crop((x,y,x+w,y+h)),(i*w,30));draw.text((i*w+4,5),name,fill='black')
    sheet.save(output/(prefix+'_native.png'))


def summarize(records):
    groups=defaultdict(list)
    for r in records:groups[(r['split'],r['kind'],r['pen_color'],r['mask_source'],r['method'])].append(r)
    summary=[]
    for key,rows in sorted(groups.items()):
        row=dict(zip(('split','kind','pen_color','mask_source','method'),key))
        row.update(n_samples=len(rows),n_sources=len({r['source_id'] for r in rows}),
                   empty_masks=sum(r['empty_mask'] for r in rows),
                   n_text_proxy_samples=sum(r['text_proxy_n_pixels']>0 for r in rows))
        for prefix in ('roi','text_proxy'):
            valid=[r for r in rows if r[prefix+'_n_pixels']>0]
            for metric in ('mae','mse'):
                row['mean_'+prefix+'_'+metric]=float(np.mean([r[prefix+'_'+metric] for r in valid])) if valid else None
            mse=row['mean_'+prefix+'_mse']
            row[prefix+'_psnr_from_mean_mse_db']=None if mse is None else float('inf') if mse==0 else float(10*np.log10(255.**2/mse))
        summary.append(row)
    return summary


def run(root,inventory_file,detection_config,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Output exists; choose a new directory')
    root=Path(root).resolve();samples,evidence=preflight(root,inventory_file,detection_config)
    previews=preview_ids(samples)
    paths=[Path(__file__),Path(det.__file__),Path(det.audit.__file__),
           det.audit.REPO_DIR/'code/src/cleaning/ink.py',det.audit.REPO_DIR/'code/src/cleaning/inpaint.py',
           det.audit.REPO_DIR/'code/src/cleaning/binarize.py',det.audit.REPO_DIR/'code/src/common/io.py']
    output.mkdir(parents=True)
    config=dict(phase='TRAIN_VAL_RESTORATION_PILOT',samples=samples,methods=list(METHODS),radius=RADIUS,
        radius_grid=None,tuning=False,expected_rows=192,expected_mask_cases=48,preview_sample_ids=previews,
        preview_rule='first source per split: stamp levels1/3, blue/black pen level2; native crop at most-occluded annotated box',
        roi='GT mask for all branches; text_proxy = GT & auto-Otsu(clean)',mask_expansion=False,
        predicted_masks='stamp_candidate on v11, blue_color on blue v12; black unsupported',
        negative_oracle_claim='GT mask diagnostic is not a detector or quality upper bound',
        inventory_sha256=det.audit.sha(inventory_file),detection_config_sha256=det.audit.sha(detection_config),
        input_sha256=evidence,code_sha256={str(p.relative_to(det.audit.REPO_DIR)):det.audit.sha(p) for p in paths},
        versions=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__),
        test_pixels_processed=False,ocr_run=False,complete=False)
    det.json_new(output/'config.json',config);config_hash=det.audit.sha(output/'config.json')
    records=[];cases=[];unsupported=[];previews_meta=[]
    for index,row in enumerate(samples):
        kind=det.audit.RECIPES[row['recipe_name']][1];sid=row['sample_id']
        image=det.read_image(det.audit.asset(root,row['rel_degraded_path']))
        clean=det.read_image(det.audit.asset(root,row['rel_clean_path']))
        binary=det.read_image(det.audit.asset(root,row['rel_clean_binary_path']),gray=True)
        gt=det.read_image(det.audit.asset(root,row['rel_'+kind+'_mask_path']),gray=True)
        if image.shape!=clean.shape or image.shape!=(row['height'],row['width'],3):raise ValueError('Color frame mismatch')
        det.audit.binary_mask(binary,image.shape[:2]);gt_pixels=det.mask_pixels(gt)
        if not gt_pixels.any():raise ValueError('Unexpected empty positive GT')
        label=json.loads(det.audit.asset(root,row['rel_gt_path']).read_text(encoding='utf-8'))
        masks=det.predictions(image) if kind=='stamp' or row['pen_color']=='blue' else {}
        branches=mask_branches(row,gt,masks)
        if kind=='pen' and row['pen_color']=='black':
            unsupported.append(dict(sample_id=sid,source_id=row['source_id'],split=row['split'],status='predicted_black_pen_unsupported_GT_diagnostic_only'))
        crop,reason=crop_region(row,gt,binary>0,label)
        for source,mask in branches.items():
            selected=det.mask_pixels(mask);mask_hash=det.array_sha(mask);prefix=sid+'_'+source
            if not cv2.imwrite(str(output/(prefix+'_mask.png')),mask):raise OSError('Mask write failed')
            cases.append(dict(sample_id=sid,source_id=row['source_id'],split=row['split'],mask_source=source,
                              mask_pixels_sha256=mask_hash,gt_pixels_sha256=det.array_sha(gt),
                              input_pixels_sha256=det.array_sha(image),clean_pixels_sha256=det.array_sha(clean)))
            outputs={}
            for method in METHODS:
                result=apply_method(image,mask,method)
                outputs[method]=result
                records.append(dict(sample_id=sid,source_id=row['source_id'],split=row['split'],kind=kind,
                    level=row[kind+'_level'],pen_color=row['pen_color'],mask_source=source,method=method,radius=RADIUS,
                    mask_pixels_sha256=mask_hash,roi_pixels_sha256=det.array_sha(gt),empty_mask=not bool(selected.any()),
                    unchanged_output=np.array_equal(result,image),outside_mask_unchanged=True,input_and_mask_unchanged=True,
                    output_pixels_sha256=det.array_sha(result),
                    predicted_outside_gt_text_proxy_pixels=int(np.count_nonzero(selected & ~gt_pixels & (binary>0))),
                    **evaluate(clean,result,gt,binary)))
            if sid in previews:
                save_preview(output,prefix,clean,outputs,mask,crop)
                previews_meta.append(dict(sample_id=sid,split=row['split'],mask_source=source,
                                          crop_xywh=list(crop),crop_reason=reason,resized=False))
        print(f'train/val restoration: {index+1}/27 {sid}',flush=True)
    if len(records)!=192 or len(cases)!=48 or len(unsupported)!=6:raise ValueError('Incomplete pilot grid')
    det.csv_new(output/'per_image.csv',records);det.csv_new(output/'summary.csv',summarize(records))
    det.csv_new(output/'unsupported.csv',unsupported)
    det.json_new(output/'mask_cases.json',cases);det.json_new(output/'previews.json',previews_meta)
    if det.audit.sha(output/'config.json')!=config_hash:raise ValueError('Frozen config changed')
    for relative,sha in config['code_sha256'].items():
        if det.audit.sha(det.audit.REPO_DIR/relative)!=sha:raise ValueError('Code changed during pilot')
    det.json_new(output/'completion.json',dict(complete=True,rows=192,mask_cases=48,samples=27,
        unsupported_predicted_black_rows=6,radius=3,tuning=False,test_pixels_processed=False,ocr_run=False,
        config_sha256=config_hash,artifact_sha256={p.name:det.audit.sha(p) for p in output.iterdir() if p.is_file()}))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=det.audit.REPO_DIR/'data/5_synthetic')
    parser.add_argument('--inventory',type=Path,default=det.audit.REPO_DIR/'results/t4/step1_inventory.json')
    parser.add_argument('--detection-config',type=Path,default=det.audit.REPO_DIR/'results/t4/train_detection/config.json')
    parser.add_argument('--output-dir',type=Path,default=det.audit.REPO_DIR/'results/t4/train_val_restoration')
    args=parser.parse_args(argv);run(args.data_dir,args.inventory,args.detection_config,args.output_dir)


if __name__=='__main__':main()
