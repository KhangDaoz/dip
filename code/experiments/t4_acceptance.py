"""T4 session7 read-only acceptance and metadata-preselected real OCR replay."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

import t4_cer as c

d=c.official.d
a=c.a
BASE=a.REPO_DIR/'results/t4'


def assert_equal_scores(actual,saved):
    for key,value in actual.items():
        if value is None:
            if saved[key]!='':raise ValueError('N/A changed: '+key)
        elif isinstance(value,bool):
            if saved[key]!=str(value):raise ValueError('Flag changed: '+key)
        elif isinstance(value,str):
            if saved[key]!=value:raise ValueError('Text changed: '+key)
        elif not np.isclose(float(saved[key]),value,rtol=1e-12,atol=1e-12):raise ValueError('Number changed: '+key)


def metadata_plan(config):
    first={}
    for row in sorted(config['reviewed_cohort'],key=lambda x:(x['source_id'],x['sample_id'],x['crop_id'])):
        first.setdefault((row['kind'],row['pen_color'],row['level']),row['crop_id'])
    return sorted(first.values())


def check_cer(folder):
    cfg=c.json_load(folder/'config.json')
    c.verify_artifacts(folder,'completion.json','artifact_sha256')
    for name,digest in cfg['code_sha256'].items():
        if a.sha(a.REPO_DIR/name)!=digest:raise ValueError('Frozen CER code changed: '+name)
    for name,digest in cfg['input_sha256'].items():
        p=Path(name) if Path(name).is_absolute() else a.REPO_DIR/'data/5_synthetic'/name
        if a.sha(p)!=digest:raise ValueError('CER input changed: '+name)
    if cfg['tuning'] or cfg['engine_fallback'] or cfg['clean_ocr_is_gt']:raise ValueError('CER policy changed')
    if a.sha(folder/'review_export.json')!=cfg['review_sha256']:
        # Export is serialized independently; compare canonical JSON with current saved reviewed cohort instead.
        exported=c.json_load(folder/'review_export.json')
    else:exported=c.json_load(folder/'review_export.json')
    by_review={x['crop_id']:x for x in exported['items']}
    rows=[]
    for raw in c.official.load_csv(folder/'per_crop.csv'):
        row=dict(raw,errors=int(raw['errors']),reference_characters=int(raw['reference_characters']),level=int(raw['level']))
        if row['split']!=cfg['split']:raise ValueError('CER cross split')
        human=by_review[row['crop_id']]
        if human['review_status']!='human_confirmed' or human['reviewed_text']!=row['reference_text']:raise ValueError('Reference is not confirmed export')
        assert_equal_scores(c.score(row['reference_text'],row['hypothesis_text']),raw)
        rows.append(row)
    if len(rows)!=cfg['expected_rows']:raise ValueError('Missing CER rows')
    expected={(row['crop_id'],case['mask_source'],m) for row in cfg['reviewed_cohort'] for case in cfg['mask_cases']
              if case['sample_id']==row['sample_id'] for m in c.METHODS}
    if len(rows)!=len(expected) or {(r['crop_id'],r['mask_source'],r['method']) for r in rows}!=expected:raise ValueError('Incomplete CER pairing')
    c.official.compare_table(c.summarize(rows),c.official.load_csv(folder/'summary.csv'))
    for cid in {r['crop_id'] for r in rows}:
        group=[r for r in rows if r['crop_id']==cid]
        if len({r['crop_bbox_xywh'] for r in group})!=1 or len({r['reference_text'] for r in group})!=1:raise ValueError('CER crop/ref mismatch')
    return cfg,rows


def image_audit(root,folder):
    cfg=c.json_load(folder/'config.json');c.official.check_frozen(cfg,root)
    c.verify_artifacts(folder,'completion.json','artifact_sha256')
    samples=c.official.cohort(cfg['samples']);by={r['sample_id']:r for r in samples}
    detections=c.official.load_csv(folder/'detection.csv');dn=[]
    for row in detections:
        sample=by[row['sample_id']]
        mask=d.read_image(folder/row['mask_path'],gray=True)
        gt=d.read_image(root/sample['rel_'+row['kind']+'_mask_path'],gray=True)
        assert_equal_scores(d.positive_scores(mask,gt),row)
        if d.array_sha(mask)!=row['prediction_pixels_sha256']:raise ValueError('Detector pixels changed')
        dn.append(dict(row,level=int(row['level']),empty_prediction=row['empty_prediction']=='True',
                       **{k:float(row[k]) for k in ('iou','precision','recall')}))
    c.official.compare_table(c.official.detection_summary(dn),c.official.load_csv(folder/'detection_summary.csv'))
    negatives=c.official.load_csv(folder/'negative_controls.csv');nn=[]
    for row in negatives:
        mask=d.read_image(folder/row['mask_path'],gray=True)
        scores=d.negative_scores(mask);assert_equal_scores(scores,row)
        nn.append(dict(row,**scores))
    c.official.compare_table(d.summarize([],nn)[1],c.official.load_csv(folder/'negative_summary.csv'))
    if len(detections)!=76 or len(negatives)!=90:raise ValueError('Detection grid incomplete')
    records=c.official.load_csv(folder/'per_image.csv');numeric=[]
    for row in records:
        x=dict(row,level=int(row['level']),empty_mask=row['empty_mask']=='True',roi_n_pixels=int(row['roi_n_pixels']),text_proxy_n_pixels=int(row['text_proxy_n_pixels']))
        for key in ('roi_mae','roi_mse','text_proxy_mae','text_proxy_mse'):x[key]=float(row[key]) if row[key] else None
        numeric.append(x)
    c.official.compare_table(c.official.restoration_summary(numeric),c.official.load_csv(folder/'summary.csv'))
    cases=c.json_load(folder/'mask_cases.json')
    for case in cases:
        row=by[case['sample_id']];mask=d.read_image(folder/case['mask_path'],gray=True)
        gt=d.read_image(root/row['rel_'+case['kind']+'_mask_path'],gray=True)
        binary=d.read_image(root/row['rel_clean_binary_path'],gray=True)
        d.mask_pixels(mask);d.mask_pixels(gt);d.mask_pixels(binary)
        group=[r for r in records if r['sample_id']==case['sample_id'] and r['mask_source']==case['mask_source']]
        if len(group)!=4 or {r['method'] for r in group}!=set(c.official.r.METHODS):raise ValueError('Restoration pairing')
        for r in group:
            if (r['mask_pixels_sha256']!=d.array_sha(mask) or r['roi_pixels_sha256']!=d.array_sha(gt)
                or int(r['text_proxy_n_pixels'])!=int(((gt>0)&(binary>0)).sum())):raise ValueError('ROI/frame mismatch')
    if len(cases)!=106 or len(records)!=424:raise ValueError('Restoration grid incomplete')
    return dict(detection_masks=76,negative_masks=90,restoration_cases=106,restoration_rows=424,all_summaries_recomputed=True)


def native_sheet(path,outputs):
    width=max(x.shape[1] for x in outputs.values());height=max(x.shape[0] for x in outputs.values())
    sheet=Image.new('RGB',(width*len(outputs),height+28),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,array) in enumerate(outputs.items()):
        sheet.paste(Image.fromarray(array).convert('RGB'),(i*width,28));draw.text((i*width+3,4),name,fill='black')
    sheet.save(path)


def run():
    out=BASE/'acceptance_v1'
    if out.exists():raise FileExistsError('Keep acceptance history')
    root=a.REPO_DIR/'data/5_synthetic';official=BASE/'official_test_v1'
    runs={'val':BASE/'cer_reduced_val_v2','test':BASE/'cer_reduced_test_v1'}
    configs={split:c.json_load(path/'config.json') for split,path in runs.items()}
    plan=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        selection='first source/sample/crop lexicographic per kind/color/level in each frozen CER cohort; metadata only, not selected by errors',
        crops={split:metadata_plan(cfg) for split,cfg in configs.items()},
        input_config_sha256={split:a.sha(path/'config.json') for split,path in runs.items()},
        auditor_sha256=a.sha(__file__),no_tuning=True)
    out.mkdir();d.json_new(out/'plan.json',plan)  # before loading predictions or doing replay.
    image_proof=image_audit(root,official)
    report={};replays=[];ocr_calls=0
    for split,folder in runs.items():
        cfg,records=check_cer(folder)
        runtime=c.runtime_gate(a.REPO_DIR/'code/.venv/t4_ocr_setup/runtime.json',cfg['runtime'])
        parent=official if split=='test' else BASE/'train_val_restoration'
        parent_cfg=c.json_load(parent/'config.json');samples={r['sample_id']:r for r in parent_cfg['samples']}
        image_scores=c.official.load_csv(parent/'per_image.csv')
        selected=[r for r in cfg['reviewed_cohort'] if r['crop_id'] in plan['crops'][split]]
        by_sid=defaultdict(list)
        for row in selected:by_sid[row['sample_id']].append(row)
        for sid,crops in by_sid.items():
            sample=samples[sid];image=d.read_image(root/sample['rel_degraded_path']);clean=d.read_image(root/sample['rel_clean_path'])
            binary=d.read_image(root/sample['rel_clean_binary_path'],gray=True)
            kind=a.RECIPES[sample['recipe_name']][1];gt=d.read_image(root/sample['rel_'+kind+'_mask_path'],gray=True)
            masks=d.predictions(image) if kind=='stamp' or sample['pen_color']=='blue' else {}
            caches={}
            for case in (x for x in cfg['mask_cases'] if x['sample_id']==sid):
                path=parent/case.get('mask_path',sid+'_'+case['mask_source']+'_mask.png');mask=d.read_image(path,gray=True)
                if case['mask_source'].startswith('predicted_'):
                    name='stamp_candidate' if kind=='stamp' else 'blue_color'
                    if not np.array_equal(masks[name],mask):raise ValueError('Real detector replay mismatch')
                for row in crops:
                    outputs={}
                    for method in c.METHODS:
                        result=clean if method=='clean' else c.official.r.apply_method(image,mask,method)
                        if method!='clean':
                            prior=next(x for x in image_scores if x['sample_id']==sid and x['mask_source']==case['mask_source'] and x['method']==method)
                            if d.array_sha(result)!=prior['output_pixels_sha256']:raise ValueError('Restoration pixel replay differs')
                            assert_equal_scores(c.official.r.evaluate(clean,result,gt,binary),prior)
                        crop=c.prep.same_crop(result,row);outputs[method]=crop
                        saved=next(x for x in records if x['crop_id']==row['crop_id'] and x['mask_source']==case['mask_source'] and x['method']==method)
                        if d.array_sha(crop)!=saved['crop_pixels_sha256']:raise ValueError('CER pixel replay differs')
                        key=(row['crop_id'],d.array_sha(crop))
                        if key not in caches:
                            caches[key]=c.real_ocr(crop);ocr_calls+=1
                        text,confidence=caches[key]
                        if text!=saved['hypothesis_text'] or not np.isclose(confidence,float(saved['confidence']),rtol=1e-12,atol=1e-12):raise ValueError('Real OCR replay differs')
                        assert_equal_scores(c.score(row['reviewed_text'],text),saved)
                        replays.append(dict(split=split,crop_id=row['crop_id'],mask_source=case['mask_source'],method=method,pixels_and_text_match=True))
                    native_sheet(out/(split+'_'+row['crop_id']+'_'+case['mask_source']+'.png'),outputs)
            print('ACCEPTANCE replay: '+split+' '+sid,flush=True)
        report[split]=dict(rows=len(records),n_lines=len(cfg['reviewed_cohort']),n_sources=len({r['source_id'] for r in cfg['reviewed_cohort']}),
            replayed_crops=len(selected),all_errors_and_summaries_recomputed=True)
    if a.sha(__file__)!=plan['auditor_sha256']:raise ValueError('Auditor changed during run')
    d.json_new(out/'replay.json',replays)
    d.json_new(out/'verification.json',dict(status='PASS_T4_MEASURED_SCOPE',image=image_proof,cer=report,
        replayed_variant_rows=len(replays),real_ocr_calls=ocr_calls,full_benchmark_rerun=False,
        no_parameter_changes=True,T5_completed=False,plan_sha256=a.sha(out/'plan.json'),
        artifact_sha256={p.name:a.sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':run()
