"""Native focused previews from saved T4 pilot masks; no new selection/tuning."""
import argparse
import json
from pathlib import Path

import numpy as np
import t4_restoration as pilot


def focus_crop(row,gt,ink,label):
    candidates=[]
    for box in label['boxes_clean']:
        details=pilot.det.audit.inspect_box(box,gt>0,ink)
        if details['bbox_valid'] and details['overlapped_reference_ink_pixels']:
            candidates.append((-details['overlapped_reference_ink_pixels'],box['id'],box['bbox']))
    region=(gt>0)&ink
    if candidates:
        _,_,(x,y,w,h)=min(candidates)
        restricted=np.zeros_like(region);restricted[y:y+h,x:x+w]=region[y:y+h,x:x+w]
        region=restricted;reason='existing_GT_ink_pixel_in_most_covered_box'
    else:
        region=gt>0;reason='GT_pixel_no_reference_text_overlap'
    yy,xx=np.where(region)
    if not len(xx):raise ValueError('No positive GT pixel for focused preview')
    # An actual selected pixel, not the center of a long text rectangle.
    nearest=np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2)
    cx,cy=int(xx[nearest]),int(yy[nearest])
    w,h=min(320,row['width']),min(240,row['height'])
    x=max(0,min(row['width']-w,cx-w//2));y=max(0,min(row['height']-h,cy-h//2))
    return (x,y,w,h),reason


def run(root,output):
    root,output=Path(root),Path(output)
    if (output/'focus_previews.json').exists():raise FileExistsError('Focused previews already exist')
    config=json.loads((output/'config.json').read_text(encoding='utf-8'))
    completion=json.loads((output/'completion.json').read_text(encoding='utf-8'))
    if pilot.det.audit.sha(output/'config.json')!=completion['config_sha256']:raise ValueError('Config changed')
    for name,digest in completion['artifact_sha256'].items():
        if pilot.det.audit.sha(output/name)!=digest:raise ValueError('Pilot artifact changed')
    rows=list(pilot.det.csv.DictReader((output/'per_image.csv').open(encoding='utf-8')))
    previous=json.loads((output/'previews.json').read_text(encoding='utf-8'))
    by_sample={s['sample_id']:s for s in config['samples']}
    metadata=[]
    for preview in previous:
        sid,mask_source=preview['sample_id'],preview['mask_source'];row=by_sample[sid]
        image=pilot.det.read_image(root/row['rel_degraded_path']);clean=pilot.det.read_image(root/row['rel_clean_path'])
        binary=pilot.det.read_image(root/row['rel_clean_binary_path'],gray=True)
        kind=pilot.det.audit.RECIPES[row['recipe_name']][1]
        gt=pilot.det.read_image(root/row['rel_'+kind+'_mask_path'],gray=True)
        mask=pilot.det.read_image(output/(sid+'_'+mask_source+'_mask.png'),gray=True)
        label=json.loads((root/row['rel_gt_path']).read_text(encoding='utf-8'))
        crop,reason=focus_crop(row,gt,binary>0,label)
        outputs={}
        for method in pilot.METHODS:
            result=pilot.apply_method(image,mask,method)
            recorded=next(r for r in rows if r['sample_id']==sid and r['mask_source']==mask_source and r['method']==method)
            if pilot.det.array_sha(result)!=recorded['output_pixels_sha256']:raise ValueError('Replay output differs')
            outputs[method]=result
        prefix=sid+'_'+mask_source+'_focus'
        pilot.save_preview(output,prefix,clean,outputs,mask,crop)
        metadata.append(dict(sample_id=sid,split=row['split'],mask_source=mask_source,crop_xywh=list(crop),
                             crop_reason=reason,resized=False,method_output_hashes_match=True,
                             image_sha256=pilot.det.audit.sha(output/(prefix+'_native.png'))))
    pilot.det.json_new(output/'focus_previews.json',dict(renderer_sha256=pilot.det.audit.sha(Path(__file__)),
        scope='Crop placement only; original pilot images/config/numerical scores unchanged',
        replayed_preview_outputs=len(previous)*4,previews=metadata))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=pilot.det.audit.REPO_DIR/'data/5_synthetic')
    parser.add_argument('--results-dir',type=Path,default=pilot.det.audit.REPO_DIR/'results/t4/train_val_restoration')
    args=parser.parse_args();run(args.data_dir,args.results_dir)


if __name__=='__main__':main()
