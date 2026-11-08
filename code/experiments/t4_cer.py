"""Gated T4 real OCR/CER: no measurement without verified human transcripts."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import cv2

import t4_official as official
import t4_ocr_prepare as prep
import t4_cer_subset as subset
from ocr.metrics import normalize_text, edit_distance
from ocr.engines import ocr_line

a = official.a
METHODS = ('clean',) + official.r.METHODS


def json_load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def review_gate(manifest, review, split='test'):
    """Metadata only. A human export is required; AI proposals never count."""
    if split not in ('val', 'test'):raise ValueError('Only val/test CER permitted')
    if len({x['crop_id'] for x in manifest}) != len(manifest):raise ValueError('Duplicate crop IDs')
    required_splits = ('val', 'test') if split == 'test' else ('val',)
    required = [x for x in manifest if x['split'] in required_splits]
    if review is None:
        return dict(ready=False, blockers=['Missing human review export'], pending=dict(Counter(x['split'] for x in required))), []
    if review.get('schema') != 1 or not isinstance(review.get('items'),list):raise ValueError('Invalid human review schema')
    if review.get('review_type') not in (None, 'human_review'):
        raise ValueError('AI-assisted proposal is not human-confirmed ground truth')
    reviewer = review.get('reviewer')
    if not isinstance(reviewer,str) or not reviewer.strip():raise ValueError('Missing named human reviewer')
    by_id={x['crop_id']:x for x in manifest}; items={}
    for item in review['items']:
        cid=item.get('crop_id')
        if cid in items or cid not in by_id:raise ValueError('Duplicate or unknown reviewed crop ID')
        source=by_id[cid]
        if item.get('text_source_sha256')!=source['text_source_sha256']:raise ValueError('Original transcript hash mismatch')
        status=item.get('review_status')
        if status not in ('pending','excluded','human_confirmed'):raise ValueError('Invalid review status')
        if status!='pending':
            if item.get('reviewer')!=reviewer:raise ValueError('Record reviewer differs from export reviewer')
            if not isinstance(item.get('reviewed_text'),str):raise ValueError('Reviewed text must be a string')
            if status=='human_confirmed' and not normalize_text(item['reviewed_text']):raise ValueError('Empty confirmed transcript')
            if status=='excluded' and not str(item.get('review_note','')).strip():raise ValueError('Missing exclusion reason')
        items[cid]=item
    pending=Counter(x['split'] for x in required if items.get(x['crop_id'],{}).get('review_status','pending')=='pending')
    selected=[]
    for row in required:
        item=items.get(row['crop_id'],{})
        if item.get('review_status')=='human_confirmed':
            selected.append(dict(row,reviewed_text=item['reviewed_text'],reviewer=reviewer,
                                 review_note=item.get('review_note','')))
    blockers=[]
    if pending:blockers.append('Unresolved val/test crop decisions')
    for s in required_splits:
        if not any(x['split']==s for x in selected):blockers.append('No confirmed transcript in '+s)
    return dict(ready=not blockers,blockers=blockers,pending=dict(pending),
                confirmed=dict(Counter(x['split'] for x in selected)),
                excluded=dict(Counter(by_id[cid]['split'] for cid,item in items.items() if item['review_status']=='excluded'))), [x for x in selected if x['split']==split]


def runtime_gate(path, expected):
    runtime=prep.configure_runtime(path)  # binary/models/list-langs only; never OCR a test crop.
    for key in ('binary_sha256','models','runtime_dll_sha256','language','psm','oem',
                'wrapper_sha256','pytesseract_version','preprocessing','normalization'):
        if runtime.get(key)!=expected.get(key):raise ValueError('OCR runtime differs from phase4: '+key)
    return runtime


def verify_artifacts(folder, completion_name, field):
    done=json_load(folder/completion_name)
    if not done.get('complete'):raise ValueError('Incomplete prerequisite: '+str(folder))
    if a.sha(folder/'config.json')!=done['config_sha256']:raise ValueError('Prerequisite config changed')
    for name,digest in done[field].items():
        if a.sha(folder/name)!=digest:raise ValueError('Prerequisite artifact changed: '+name)
    return done


def preflight(root, preparation, official_dir, review_path, runtime_path, split='test',cohort_path=None):
    """Check raw hashes/schema/runtime first; no image pixel decode or OCR."""
    root,preparation,official_dir=Path(root),Path(preparation),Path(official_dir)
    pconfig=json_load(preparation/'config.json'); pdone=json_load(preparation/'completion.json')
    if (not pdone.get('complete') or a.sha(preparation/'config.json')!=pdone['config_sha256']
            or a.sha(preparation/'crop_manifest.json')!=pdone['manifest_sha256']
            or a.sha(Path(prep.__file__))!=pconfig['script_sha256']):raise ValueError('Preparation provenance changed')
    manifest=json_load(preparation/'crop_manifest.json')
    if Counter(x['split'] for x in manifest)!={'train':440,'val':81,'test':251}:raise ValueError('Prepared crop cohort changed')
    done=verify_artifacts(official_dir,'completion.json','artifact_sha256')
    if done.get('cer_computed') or done.get('ocr_run'):raise ValueError('Unexpected official prerequisite scope')
    config=json_load(official_dir/'config.json');official.check_frozen(config,root)
    pilot=a.REPO_DIR/'results/t4/train_val_restoration'
    verify_artifacts(pilot,'completion.json','artifact_sha256')
    inventory=a.REPO_DIR/'results/t4/step1_inventory.json'
    if a.sha(inventory)!=pconfig['inventory_sha256']:raise ValueError('Inventory changed')
    rows={x['sample_id']:x for x in config['samples']}
    rows.update({x['sample_id']:x for x in json_load(pilot/'config.json')['samples'] if x['split']=='val'})
    # Hash source labels and image bytes before any pixel/OCR use.
    inputs={}
    for row in manifest:
        if row['split'] not in ('val','test'):continue
        if row['sample_id'] not in rows or rows[row['sample_id']]['split']!=row['split']:raise ValueError('Crop split/source mismatch')
        sample=rows[row['sample_id']]
        if sample['source_id']!=row['source_id']:raise ValueError('Crop source identity mismatch')
        if row['crop_bbox_xywh']!=prep.crop_box(row['annotation_bbox_xywh'],row['width'],row['height']):raise ValueError('Crop rectangle changed')
        for key in ('rel_clean_path','rel_degraded_path','rel_label_path'):
            name=row[key];digest=a.sha(a.asset(root,name))
            if pconfig['input_sha256'].get(name)!=digest:raise ValueError('Prepared source changed: '+name)
            inputs[name]=digest
        for key in ('clean_crop','degraded_crop'):
            if a.sha(a.asset(preparation,row[key]))!=row[key+'_sha256']:raise ValueError('Prepared crop changed')
    runtime=runtime_gate(runtime_path,pconfig['runtime'])
    review=json_load(review_path) if review_path and Path(review_path).is_file() else None
    cohort_sha=None
    if cohort_path is not None:
        lock=json_load(cohort_path);cohort_sha=a.sha(cohort_path)
        subset.validate(lock,manifest,a.sha(preparation/'crop_manifest.json'))
        if review is not None:
            review_gate(manifest,review,split)  # validate ALL IDs/hash/status, including reserves.
            if review.get('cohort_id')!=lock['cohort_id'] or review.get('cohort_sha256')!=cohort_sha:
                raise ValueError('Review export is not bound to reduced cohort')
        active,excluded=subset.resolve(lock,review)
        active_ids={x['crop_id'] for x in active}
        filtered=dict(review,items=[x for x in review['items'] if x['crop_id'] in active_ids]) if review else None
        gate,selected=review_gate(active,filtered,split)
        manifest=active+excluded
    else:gate,selected=review_gate(manifest,review,split)
    return dict(status='READY_FOR_FROZEN_CER' if gate['ready'] else 'BLOCKED_TRANSCRIPTS',
                split=split,transcripts=gate,ocr_runtime_ready=True,ocr_run=False,cer_computed=False,
                runtime=runtime,selected=selected,manifest=manifest,source_samples=rows,
                input_sha256=inputs,review_sha256=a.sha(review_path) if review is not None else None,
                crop_manifest_sha256=a.sha(preparation/'crop_manifest.json'),
                cohort_sha256=cohort_sha,cohort_path=str(cohort_path) if cohort_path else None,
                runner_sha256=a.sha(Path(__file__)),protocol_sha256=a.sha(Path(__file__).with_suffix('.md')),
                ocr_metrics_sha256=a.sha(a.REPO_DIR/'code/src/ocr/metrics.py'),
                official_config_sha256=a.sha(official_dir/'config.json'))


def score(reference,hypothesis):
    if not isinstance(reference,str) or not isinstance(hypothesis,str):raise ValueError('Text must be strings')
    ref,hyp=normalize_text(reference),normalize_text(hypothesis)
    if not ref:raise ValueError('Empty normalized reference is not a CER sample')
    errors=edit_distance(ref,hyp)
    return dict(reference_normalized=ref,hypothesis_normalized=hyp,errors=errors,
                reference_characters=len(ref),cer=errors/len(ref))


def summarize(records):
    """Corpus CER and paired deltas; never average per-crop CER."""
    groups=defaultdict(list)
    seen=set()
    for row in records:
        identity=(row['crop_id'],row['mask_source'],row['method'])
        if identity in seen:raise ValueError('Duplicate scored crop/branch/method')
        seen.add(identity)
        if row['method'] not in METHODS or row['reference_characters']<=0:raise ValueError('Invalid CER record')
        if not isinstance(row['errors'],int) or row['errors']<0:raise ValueError('Invalid edit count')
        for level in (row['level'],'all'):
            key=(row['split'],row['kind'],row['pen_color'],level,row['mask_source'])
            groups[key].append(row)
    result=[]
    for key,group in sorted(groups.items(),key=lambda x:str(x[0])):
        by_method={m:[x for x in group if x['method']==m] for m in METHODS}
        pairs=None
        for method,rows in by_method.items():
            refs={x['crop_id']:(x['reference_characters'],x['reference_normalized']) for x in rows}
            if not refs:raise ValueError('Missing comparison method')
            if pairs is not None and refs!=pairs:raise ValueError('Unpaired comparison cohort or reference')
            pairs=refs
        ratios={m:sum(x['errors'] for x in rows)/sum(x['reference_characters'] for x in rows) for m,rows in by_method.items()}
        for method,rows in by_method.items():
            result.append(dict(zip(('split','kind','pen_color','level','mask_source'),key),method=method,
                n_sources=len({x['source_id'] for x in rows}),n_lines=len(rows),
                total_errors=sum(x['errors'] for x in rows),total_reference_characters=sum(x['reference_characters'] for x in rows),
                corpus_cer=ratios[method],delta_vs_none=ratios[method]-ratios['none'],
                delta_vs_clean=ratios[method]-ratios['clean']))
    return result


def real_ocr(crop):
    """Hard-coded real engine: no fallback, mock flag or OCR-derived reference."""
    text,confidence=ocr_line(crop,engine='tesseract',lang='vie+eng')
    if not isinstance(text,str) or not np.isfinite(confidence):raise ValueError('Invalid real OCR response')
    return text,float(confidence)


def case_plan(cases, selected):
    by_sample=defaultdict(list)
    for row in selected:by_sample[row['sample_id']].append(row)
    chosen=[c for c in cases if c['sample_id'] in by_sample]
    for sid,rows in by_sample.items():
        expected={'gt_diagnostic'}
        first=rows[0]
        if first['kind']=='stamp':expected.add('predicted_stamp_candidate')
        elif first['pen_color']=='blue':expected.add('predicted_blue_color')
        elif first['pen_color']!='black':raise ValueError('Unknown pen color')
        sources=[x['mask_source'] for x in chosen if x['sample_id']==sid]
        if len(sources)!=len(expected) or set(sources)!=expected:raise ValueError('Missing/duplicated/unsupported mask branch')
    return chosen,by_sample


def coverage_rows(manifest, review, samples, split):
    items={x['crop_id']:x for x in review['items']};coverage=[];excluded=[]
    for sample in sorted((s for s in samples.values() if s['split']==split),key=lambda s:s['sample_id']):
        boxes=[x for x in manifest if x['sample_id']==sample['sample_id']]
        statuses=Counter(items[x['crop_id']]['review_status'] for x in boxes)
        if statuses['pending']:raise ValueError('Unresolved crop in run coverage')
        kind=a.RECIPES[sample['recipe_name']][1]
        coverage.append(dict(sample_id=sample['sample_id'],source_id=sample['source_id'],split=split,
            kind=kind,pen_color=sample['pen_color'],candidate_lines=len(boxes),confirmed_lines=statuses['human_confirmed'],
            excluded_lines=statuses['excluded'],no_eligible_text_region=not bool(boxes),
            predicted_detector_supported=kind=='stamp' or sample['pen_color']=='blue'))
        for box in boxes:
            item=items[box['crop_id']]
            if item['review_status']=='excluded':excluded.append(dict(crop_id=box['crop_id'],sample_id=sample['sample_id'],
                source_id=sample['source_id'],reviewer=item['reviewer'],reason=item['review_note']))
    return coverage,excluded


def run(root,preparation,official_dir,review_path,runtime_path,output,split='test',cohort_path=None):
    output=Path(output)
    if output.exists():raise FileExistsError('Keep history: CER output already exists')
    gate=preflight(root,preparation,official_dir,review_path,runtime_path,split,cohort_path)
    if gate['status']!='READY_FOR_FROZEN_CER':raise ValueError('CER blocked: '+ '; '.join(gate['transcripts']['blockers']))
    root,preparation,official_dir=Path(root),Path(preparation),Path(official_dir)
    parent=official_dir if split=='test' else a.REPO_DIR/'results/t4/train_val_restoration'
    cases=json_load(parent/'mask_cases.json'); old_rows=official.load_csv(parent/'per_image.csv')
    rows=gate['selected'];chosen_cases,by_sample=case_plan(cases,rows)
    review=json_load(review_path)
    coverage,excluded=coverage_rows(gate['manifest'],review,gate['source_samples'],split)
    # Expected five OCR variants per confirmed crop, per supported/GT branch.
    expected=sum(len(by_sample[c['sample_id']])*len(METHODS) for c in chosen_cases)
    inputs=dict(gate['input_sha256']);mask_files={}
    for case in chosen_cases:
        name=case.get('mask_path',case['sample_id']+'_'+case['mask_source']+'_mask.png')
        path=parent/name;mask_files[(case['sample_id'],case['mask_source'])]=path
        inputs[str(path.resolve())]=a.sha(path)
    code_paths=[Path(__file__),Path(__file__).with_suffix('.md'),Path(subset.__file__),Path(prep.__file__),a.REPO_DIR/'code/src/ocr/engines.py',a.REPO_DIR/'code/src/ocr/metrics.py']
    config=dict(phase='REAL_T4_CER',split=split,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        engine='tesseract',runtime=gate['runtime'],methods=list(METHODS),radius=3,
        normalization='ocr.metrics.normalize_text: NFC + collapse whitespace; preserve case/punctuation/numbers/units',
        reviewed_cohort=rows,review_sha256=gate['review_sha256'],crop_manifest_sha256=gate['crop_manifest_sha256'],
        cohort_sha256=gate['cohort_sha256'],cohort_path=gate['cohort_path'],
        expected_rows=expected,mask_cases=chosen_cases,input_sha256=inputs,
        official_config_sha256=gate['official_config_sha256'],parent_config_sha256=a.sha(parent/'config.json'),
        code_sha256={str(p.resolve().relative_to(a.REPO_DIR)):a.sha(p) for p in code_paths},
        python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,
        tuning=False,engine_fallback=False,clean_ocr_is_gt=False,
        clean_and_none_repeated_across_branches=True,aggregation='sum edit errors / sum normalized human-reference chars within each separate paired cohort',
        failures='abort, no success marker; empty OCR is scored as deletions, never dropped')
    output.mkdir(parents=True);official.d.json_new(output/'config.json',config)
    official.d.json_new(output/'review_export.json',review)
    official.d.csv_new(output/'coverage.csv',coverage)
    if excluded:official.d.csv_new(output/'exclusions.csv',excluded)
    else:(output/'exclusions.csv').write_text('crop_id,sample_id,source_id,reviewer,reason\n',encoding='utf-8')
    (output/'README.md').write_text(
        '# T4 real OCR/CER\n\nOnly a successful completion.json marks a finished run.\n'
        'Reference = named human-reviewed transcript, never clean-image OCR.\n'
        'Engine/model/language/crop/radius/normalization frozen in config.json before OCR.\n'
        'summary.csv reports sum errors / sum human-reference characters, not mean line CER.\n'
        'Negative delta_vs_none means fewer errors; no per-test method selection.\n'
        'Predicted/GT masks are separate; black pen is GT diagnostic only.\n'
        'Clean and no-removal OCR are cached then repeated in paired branch tables; do not pool branches.\n'
        'coverage.csv includes images without eligible text regions; exclusions.csv preserves human reasons.\n'
        'Empty OCR hypotheses count as deletions. Engine failures abort rather than dropping samples.\n'
        'No inference about general signature detection or T5.\n',encoding='utf-8')
    config_hash=a.sha(output/'config.json');records=[];clean_cache={};none_cache={}
    for sid,crops in by_sample.items():
        sample=gate['source_samples'][sid]
        clean=official.d.read_image(root/sample['rel_clean_path']);image=official.d.read_image(root/sample['rel_degraded_path'])
        for row in crops:
            for field,arr in (('clean_crop',clean),('degraded_crop',image)):
                crop=prep.same_crop(arr,row)
                stored=official.d.read_image(preparation/row[field],gray=True)
                if not np.array_equal(stored,crop):raise ValueError('Crop pixel replay changed')
            clean_cache[row['crop_id']]=real_ocr(prep.same_crop(clean,row))
            none_cache[row['crop_id']]=real_ocr(prep.same_crop(image,row))
        for case in (c for c in chosen_cases if c['sample_id']==sid):
            mask=official.d.read_image(mask_files[(sid,case['mask_source'])],gray=True)
            if official.d.array_sha(mask)!=case['mask_pixels_sha256']:raise ValueError('Measured mask changed')
            for method in METHODS:
                result=clean if method=='clean' else official.r.apply_method(image,mask,method)
                if method!='clean':
                    previous=next(x for x in old_rows if x['sample_id']==sid and x['mask_source']==case['mask_source'] and x['method']==method)
                    if official.d.array_sha(result)!=previous['output_pixels_sha256']:raise ValueError('Restoration differs from phase3/5')
                for row in crops:
                    crop=prep.same_crop(result,row)
                    text,confidence=(clean_cache[row['crop_id']] if method=='clean' else none_cache[row['crop_id']] if method=='none' else real_ocr(crop))
                    records.append(dict(crop_id=row['crop_id'],sample_id=sid,source_id=row['source_id'],split=split,
                        kind=row['kind'],pen_color=row['pen_color'],level=row['level'],mask_source=case['mask_source'],
                        method=method,crop_bbox_xywh=json.dumps(row['crop_bbox_xywh']),crop_pixels_sha256=official.d.array_sha(crop),
                        mask_pixels_sha256=case['mask_pixels_sha256'],reviewer=row['reviewer'],
                        reference_text=row['reviewed_text'],hypothesis_text=text,confidence=confidence,
                        **score(row['reviewed_text'],text)))
        print('Real OCR scored source image: '+sid,flush=True)
    if len(records)!=expected:raise ValueError('Incomplete OCR grid')
    official.d.csv_new(output/'per_crop.csv',records)
    official.d.csv_new(output/'summary.csv',summarize(records))
    # Revalidate review/runtime/artifacts/code after OCR; never silently accept changes mid-run.
    after=preflight(root,preparation,official_dir,review_path,runtime_path,split,cohort_path)
    if after['cohort_sha256']!=config['cohort_sha256'] or after['review_sha256']!=config['review_sha256'] or after['selected']!=rows:raise ValueError('Review/cohort changed during run')
    for name,digest in config['code_sha256'].items():
        if a.sha(a.REPO_DIR/name)!=digest:raise ValueError('CER code changed during run')
    if a.sha(output/'config.json')!=config_hash:raise ValueError('CER config changed')
    official.d.json_new(output/'completion.json',dict(complete=True,real_ocr=True,rows=len(records),
        n_sources=len({x['source_id'] for x in rows}),n_lines=len(rows),
        config_sha256=config_hash,artifact_sha256={p.name:a.sha(p) for p in output.iterdir() if p.is_file()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=a.REPO_DIR/'data/6_synthetic')
    p.add_argument('--preparation',type=Path,default=a.REPO_DIR/'results/t4/ocr_preparation')
    p.add_argument('--official-dir',type=Path,default=a.REPO_DIR/'results/t4/official_test_v1')
    p.add_argument('--review',type=Path,default=a.REPO_DIR/'results/t4/ocr_preparation/t4_transcript_review_reduced_v1.json')
    p.add_argument('--cohort',type=Path,default=a.REPO_DIR/'results/t4/ocr_preparation/cohort_reduced_v1.json')
    p.add_argument('--runtime',type=Path,default=a.REPO_DIR/'code/.venv/t4_ocr_setup/runtime.json')
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--run',action='store_true')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--readiness-report',type=Path,default=a.REPO_DIR/'results/t4/step6_reduced_readiness.json')
    args=p.parse_args()
    if args.run:run(args.data_dir,args.preparation,args.official_dir,args.review,args.runtime,
                    args.output_dir or a.REPO_DIR/f'results/t4/cer_reduced_{args.split}_v1',args.split,args.cohort)
    else:
        gate=preflight(args.data_dir,args.preparation,args.official_dir,args.review,args.runtime,args.split,args.cohort)
        report={k:v for k,v in gate.items() if k not in ('selected','manifest','source_samples','input_sha256')}
        official.d.json_new(args.readiness_report,report)
        print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
