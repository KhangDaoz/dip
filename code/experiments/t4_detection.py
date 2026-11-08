"""T4 session 2: fixed TRAIN detection sanity; no erasure, tuning or test mode."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import platform

import cv2
import numpy as np

import t4_marks as audit
from common.io import read_image
from cleaning.ink import ink_masks, split_stamp_vs_text

DETECTORS = ('red_color', 'stamp_candidate', 'blue_color')
CELLS = {('stamp',1,'none'),('stamp',2,'none'),('stamp',3,'none'),
         ('pen',1,'blue'),('pen',2,'blue'),('pen',1,'black'),('pen',2,'black')}


def mask_pixels(mask):
    if (not isinstance(mask,np.ndarray) or mask.dtype!=np.uint8 or mask.ndim!=2
            or mask.size==0 or not np.all((mask==0)|(mask==255))):
        raise ValueError('Expected nonempty uint8 2D mask, foreground=255/background=0')
    return mask==255


def positive_scores(prediction, target):
    pred, gt = mask_pixels(prediction), mask_pixels(target)
    if pred.shape!=gt.shape:
        raise ValueError('Mask frames differ')
    if not gt.any():
        raise ValueError('Empty GT belongs in the separate negative-control table')
    tp=int(np.count_nonzero(pred & gt)); fp=int(np.count_nonzero(pred & ~gt))
    fn=int(np.count_nonzero(~pred & gt)); n_pred=tp+fp
    return dict(tp=tp,fp=fp,fn=fn,gt_pixels=tp+fn,predicted_pixels=n_pred,
                iou=tp/(tp+fp+fn),precision=tp/n_pred if n_pred else 0.,
                recall=tp/(tp+fn),empty_prediction=not bool(n_pred))


def negative_scores(prediction):
    pred=mask_pixels(prediction)
    count=int(np.count_nonzero(pred))
    n_components=cv2.connectedComponents(pred.astype(np.uint8),connectivity=8)[0]-1
    return dict(fp_pixels=count,fp_pixel_ratio=count/pred.size,
                any_false_positive=bool(count),fp_components_8_connected=int(n_components),
                iou=None,precision=None,recall=None)


def select_train(samples):
    chosen={}
    for row in sorted((r for r in samples if r['split']=='train'),key=lambda r:r['source_id']):
        kind=audit.RECIPES[row['recipe_name']][1]
        key=(kind,int(row[kind+'_level']),row['pen_color'])
        if key in CELLS: chosen.setdefault(key,row)
    if set(chosen)!=CELLS:
        raise ValueError('Missing required TRAIN kind/level/color cell')
    return [chosen[key] for key in sorted(CELLS)]


def array_sha(array):
    import hashlib
    header=json.dumps([str(array.dtype),list(array.shape)]).encode()
    return hashlib.sha256(header+array.tobytes()).hexdigest()


def predictions(image):
    original=array_sha(image)
    colors=ink_masks(image)
    stamp,keep=split_stamp_vs_text(colors['red'])
    result=dict(red_color=colors['red'],stamp_candidate=stamp,blue_color=colors['blue'])
    for pred in result.values():
        mask_pixels(pred)
        if pred.shape!=image.shape[:2]: raise ValueError('Prediction frame changed')
    if (np.any((stamp>0)&(keep>0)) or not np.array_equal(stamp|keep,colors['red'])
            or np.any((stamp>0)&(colors['red']==0)) or array_sha(image)!=original):
        raise ValueError('Mutation or stamp partition expanded/redrawn pixels')
    # Black is deliberately NOT exposed as a signature detector.
    return result


def overlay(image, prediction, target):
    pred,gt=mask_pixels(prediction),mask_pixels(target)
    if pred.shape!=image.shape[:2] or pred.shape!=gt.shape: raise ValueError('Overlay frame mismatch')
    out=image.copy()
    for selected,color in ((pred & gt,(0,220,0)),(pred & ~gt,(255,0,255)),(~pred & gt,(0,165,255))):
        out[selected]=np.round(.45*image[selected]+.55*np.array(color)).astype(np.uint8)
    return out


def preflight(root, inventory_file):
    root=Path(root).resolve()
    if audit.sha(root/'manifest_v2.json')!=audit.MANIFEST_SHA or audit.sha(root/'checksums_v2.sha256')!=audit.CHECKSUM_SHA:
        raise ValueError('Dataset baseline changed')
    prior=json.loads(Path(inventory_file).read_text(encoding='utf-8'))
    if (prior['status']!='PASS_DATA_INVENTORY_NOT_BENCHMARK' or
            prior['manifest_sha256']!=audit.MANIFEST_SHA or prior['checksum_list_sha256']!=audit.CHECKSUM_SHA
            or prior['inventory_script_sha256']!=audit.sha(Path(audit.__file__))):
        raise ValueError('Session-1 evidence mismatch')
    samples=audit.select_samples(json.loads((root/'manifest_v2.json').read_text(encoding='utf-8')))
    selected=select_train(samples)
    delivered={}
    for line in (root/'checksums_v2.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split(maxsplit=1);delivered[name.lstrip('*')]=digest
    hashes={}
    for row in selected:
        for key in ('rel_degraded_path','rel_clean_path','rel_clean_binary_path','rel_gt_path',
                    'rel_stamp_mask_path','rel_pen_mask_path','rel_glare_mask_path'):
            relative=row[key]; digest=audit.sha(audit.asset(root,relative))
            if delivered.get(relative)!=digest or prior['assets_sha256'].get(relative)!=digest:
                raise ValueError('Selected input checksum changed: '+relative)
            hashes[relative]=digest
    return selected,hashes


def json_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def csv_new(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(positives,negatives):
    groups=defaultdict(list)
    for row in positives:groups[row['detector']].append(row)
    pos=[dict(detector=name,n_samples=len(rows),n_sources=len({r['source_id'] for r in rows}),
              mean_iou=float(np.mean([r['iou'] for r in rows])),
              mean_precision=float(np.mean([r['precision'] for r in rows])),
              mean_recall=float(np.mean([r['recall'] for r in rows])),
              empty_predictions=sum(r['empty_prediction'] for r in rows)) for name,rows in sorted(groups.items())]
    groups=defaultdict(list)
    for row in negatives:groups[row['detector']].append(row)
    neg=[dict(detector=name,n_controls=len(rows),fp_pixels=sum(r['fp_pixels'] for r in rows),
              mean_fp_pixel_ratio=float(np.mean([r['fp_pixel_ratio'] for r in rows])),
              images_with_fp=sum(r['any_false_positive'] for r in rows),
              image_fp_rate=float(np.mean([r['any_false_positive'] for r in rows])),
              components=sum(r['fp_components_8_connected'] for r in rows)) for name,rows in sorted(groups.items())]
    return pos,neg


def run(root, inventory_file, output):
    output=Path(output)
    if output.exists():raise FileExistsError('Output exists; choose a new directory')
    selected,hashes=preflight(root,inventory_file)
    controls={}
    for row in selected:
        sid=row['source_id']
        if sid in controls and hashes[controls[sid]['rel_clean_path']]!=hashes[row['rel_clean_path']]:
            raise ValueError('Conflicting clean controls')
        controls.setdefault(sid,row)
    code_paths=[Path(__file__),Path(audit.__file__),audit.REPO_DIR/'code/src/cleaning/ink.py',
                audit.REPO_DIR/'code/src/cleaning/binarize.py',audit.REPO_DIR/'code/src/common/io.py']
    output.mkdir(parents=True)
    config=dict(phase='TRAIN_SANITY_ONLY',selection_rule='smallest train source_id per kind/level/color',
                selected_samples=selected,negative_control_sources=sorted(controls),
                manifest_sha256=audit.MANIFEST_SHA,checksum_list_sha256=audit.CHECKSUM_SHA,
                inventory_sha256=audit.sha(inventory_file),input_sha256=hashes,
                code_sha256={str(p.relative_to(audit.REPO_DIR)):audit.sha(p) for p in code_paths},
                versions=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__),
                detector_definitions=dict(red_color='ink_masks.red; color baseline',
                    stamp_candidate='split_stamp_vs_text(red)[0]; closed-rim candidate',
                    blue_color='ink_masks.blue; color baseline, not general signature detector'),
                black_pen='unsupported; never replaced by full-page black mask',
                empty_positive_precision=0,negative_iou='N/A; never pooled with positives',
                expected_positive_rows=8,expected_controls=len(controls)*3,
                detector_parameters='existing defaults unchanged; ink.py SHA identifies current local v5',
                tuning=False,mask_expansion=False,erasure_run=False,complete=False)
    json_new(output/'config.json',config)
    config_sha=audit.sha(output/'config.json')
    positives=[];negatives=[];unsupported=[]
    for row in selected:
        kind=audit.RECIPES[row['recipe_name']][1]
        if kind=='pen' and row['pen_color']=='black':
            unsupported.append(dict(sample_id=row['sample_id'],source_id=row['source_id'],split='train',
                                    level=row['pen_level'],status='unsupported_black_pen_detector'))
            continue
        image=read_image(audit.asset(root,row['rel_degraded_path']))
        if image.shape!=(row['height'],row['width'],3) or image.dtype!=np.uint8:
            raise ValueError('Unexpected image frame/type')
        gt=read_image(audit.asset(root,row['rel_'+kind+'_mask_path']),gray=True)
        audit.binary_mask(gt,image.shape[:2])
        masks=predictions(image)
        names=('red_color','stamp_candidate') if kind=='stamp' else ('blue_color',)
        for name in names:
            pred=masks[name]
            record=dict(sample_id=row['sample_id'],source_id=row['source_id'],split='train',
                        kind=kind,level=row[kind+'_level'],pen_color=row['pen_color'],detector=name,
                        input_pixels_sha256=array_sha(image),gt_pixels_sha256=array_sha(gt),
                        prediction_pixels_sha256=array_sha(pred),**positive_scores(pred,gt))
            positives.append(record)
            prefix=f'{row["sample_id"]}_{name}'
            if not cv2.imwrite(str(output/(prefix+'_mask.png')),pred):raise OSError('Mask write failed')
            if not cv2.imwrite(str(output/(prefix+'_overlay.png')),overlay(image,pred,gt)):raise OSError('Overlay write failed')
        print('train positive:',row['sample_id'],flush=True)
    for sid,row in sorted(controls.items()):
        image=read_image(audit.asset(root,row['rel_clean_path']))
        for name,pred in predictions(image).items():
            negatives.append(dict(source_id=sid,split='train',detector=name,
                                  clean_path=row['rel_clean_path'],prediction_pixels_sha256=array_sha(pred),
                                  **negative_scores(pred)))
        print('train control:',sid,flush=True)
    if len(positives)!=8 or len(negatives)!=len(controls)*3 or len(unsupported)!=2:
        raise ValueError('Incomplete sanity grid')
    ps,ns=summarize(positives,negatives)
    for name,rows in [('positives.csv',positives),('negative_controls.csv',negatives),
                      ('positive_summary.csv',ps),('negative_summary.csv',ns),('unsupported.csv',unsupported)]:
        csv_new(output/name,rows)
    if audit.sha(output/'config.json')!=config_sha or any(audit.sha(p)!=h for p,h in
            ((audit.REPO_DIR/k,v) for k,v in config['code_sha256'].items())):
        raise ValueError('Code/config changed during run')
    json_new(output/'completion.json',dict(complete=True,positive_rows=len(positives),negative_rows=len(negatives),
               unsupported_rows=len(unsupported),train_sources=len(controls),config_sha256=config_sha,
               val_pixels_processed=False,test_pixels_processed=False,tuning=False,erasure_run=False,
               artifacts_sha256={p.name:audit.sha(p) for p in output.iterdir() if p.is_file()}))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=audit.REPO_DIR/'data/5_synthetic')
    parser.add_argument('--inventory',type=Path,default=audit.REPO_DIR/'results/t4/step1_inventory.json')
    parser.add_argument('--output-dir',type=Path,default=audit.REPO_DIR/'results/t4/train_detection')
    args=parser.parse_args(argv)
    run(args.data_dir,args.inventory,args.output_dir)


if __name__=='__main__':main()
