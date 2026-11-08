"""Frozen T4 TEST detection/restoration; no OCR, tuning or mask expansion."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import inspect
import importlib.metadata
import json
from pathlib import Path
import platform

import cv2
import numpy as np
import PIL

import t4_restoration as r
from t4_restoration_review import focus_crop

d = r.det
a = d.audit
ASSETS = ('rel_degraded_path', 'rel_clean_path', 'rel_clean_binary_path', 'rel_gt_path',
          'rel_stamp_mask_path', 'rel_pen_mask_path', 'rel_glare_mask_path')


def cohort(rows):
    test = sorted((x for x in rows if x['split'] == 'test'), key=lambda x: x['sample_id'])
    stamp = [x for x in test if x['recipe_name'] == 'single_stamp']
    pen = [x for x in test if x['recipe_name'] == 'single_pen']
    if (len(test) != 60 or len({x['sample_id'] for x in test}) != 60
            or len({x['source_id'] for x in test}) != 30
            or Counter(x['stamp_level'] for x in stamp) != {1: 10, 2: 10, 3: 10}
            or Counter((x['pen_color'], x['pen_level']) for x in pen)
            != {('blue', 1): 8, ('blue', 2): 8, ('black', 1): 7, ('black', 2): 7}
            or {x['source_id'] for x in stamp} != {x['source_id'] for x in pen}):
        raise ValueError('Incomplete or changed TEST cohort')
    return test


def stratified(rows, summarize):
    groups = defaultdict(list)
    for x in rows:
        groups[(x['kind'], x['pen_color'], x['level'])].append(x)
        groups[(x['kind'], x['pen_color'], 'all')].append(x)
    out = []
    for (kind, color, level), group in sorted(groups.items(), key=lambda x: str(x[0])):
        summaries = summarize(group)
        out.extend(dict(kind=kind, pen_color=color, level=level, **{
            k: v for k, v in row.items() if k not in ('kind', 'pen_color')}) for row in summaries)
    return out


def detection_summary(rows):
    return stratified(rows, lambda group: d.summarize(group, [])[0])


def restoration_summary(rows):
    return stratified(rows, r.summarize)


def freeze(root, output):
    """Only metadata/raw-byte hashing before config is closed; no decoded pixels."""
    if output.exists():
        raise FileExistsError('Keep history: output already exists')
    inventory = a.REPO_DIR / 'results/t4/step1_inventory.json'
    prior_path = a.REPO_DIR / 'results/t4/train_val_restoration/config.json'
    completion_path = prior_path.with_name('completion.json')
    d.preflight(root, inventory)
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    done = json.loads(completion_path.read_text(encoding='utf-8'))
    if not done['complete'] or done['config_sha256'] != a.sha(prior_path):
        raise ValueError('Pilot config/completion mismatch')
    if prior['methods'] != list(r.METHODS) or prior['radius'] != 3 or prior['radius_grid'] is not None:
        raise ValueError('Pilot method/radius baseline changed')
    for name, digest in prior['code_sha256'].items():
        if a.sha(a.REPO_DIR / name) != digest:
            raise ValueError('Code differs from frozen TRAIN/VAL pilot: ' + name)
    all_rows = a.select_samples(json.loads((root / 'manifest_v2.json').read_text(encoding='utf-8')))
    samples = cohort(all_rows)
    inv = json.loads(inventory.read_text(encoding='utf-8'))
    delivered = {}
    for line in (root / 'checksums_v2.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split(maxsplit=1)
        delivered[name.lstrip('*')] = digest
    evidence = {}
    for row in samples:
        for key in ASSETS:
            name = row[key]
            actual = a.sha(a.asset(root, name))
            if delivered.get(name) != actual or inv['assets_sha256'].get(name) != actual:
                raise ValueError('Input checksum mismatch: ' + name)
            evidence[name] = actual
    controls = {}
    for row in samples:
        sid = row['source_id']
        if sid in controls and any(evidence[row[key]] != evidence[controls[sid][key]]
                                  for key in ('rel_clean_path', 'rel_clean_binary_path')):
            raise ValueError('Conflicting source clean controls')
        controls.setdefault(sid, row)
    chosen = {}
    for row in sorted(samples, key=lambda x: x['source_id']):
        kind = a.RECIPES[row['recipe_name']][1]
        chosen.setdefault((kind, row[kind + '_level'], row['pen_color']), row['sample_id'])
    paths = dict(prior['code_sha256'])
    for p in (Path(__file__), Path(inspect.getfile(focus_crop)),
              Path(__file__).with_name('t4_official_report.py'), a.REPO_DIR / 'code/experiments/t4_marks.md'):
        paths[str(p.resolve().relative_to(a.REPO_DIR))] = a.sha(p)
    config = dict(phase='OFFICIAL_TEST_DETECTION_RESTORATION_NO_OCR',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), samples=samples,
        controls=[controls[k] for k in sorted(controls)], methods=list(r.METHODS), radius=3,
        expected=dict(samples=60, stamp_samples=30, blue_pen_samples=16, black_pen_samples=14,
                      detection_rows=76, negative_rows=90, mask_cases=106,
                      restoration_rows=424, negative_restoration_rows=240, unsupported_rows=14),
        preview_ids=sorted(chosen.values()), preview_rule='first source per kind/level/color; focus GT ink then annotation ID',
        input_sha256=evidence, code_sha256=paths,
        manifest_sha256=a.MANIFEST_SHA, checksum_list_sha256=a.CHECKSUM_SHA,
        inventory_sha256=a.sha(inventory), pilot_config_sha256=a.sha(prior_path),
        versions=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__, pillow=PIL.__version__,
                      matplotlib=importlib.metadata.version('matplotlib')),
        policy=dict(tuning=False, mask_expansion=False, radius_grid=None, ocr_run=False, cer_run=False,
            detector='exact code/defaults from TRAIN/VAL; black pen unsupported',
            masks='stamp_candidate/blue_color predictions; GT diagnostic on all60; red color detection only',
            roi='GT support; text_proxy=GT & auto-Otsu(clean), zero text pixels => N/A',
            aggregation='macro per image; level strata have different sources, not paired severity curves',
            negatives='30 unique clean sources; injected-mark negatives, not human certified real-mark-free',
            negative_restoration='two primary masks x four methods; full-page/text-proxy changes vs same clean',
            failure='fail immediately, retain partial output without success marker; never skip samples',
            replay='first source per each of seven kind/level/color cells; controls first/last source',
            previous_test_exposure='T1-T3 reused source split; session1 inventory and session4 crops already decoded, not unseen test images'))
    output.mkdir(parents=True)
    d.json_new(output / 'config.json', config)
    return config


def check_frozen(config, root):
    if a.sha(root / 'manifest_v2.json') != config['manifest_sha256'] or a.sha(root / 'checksums_v2.sha256') != config['checksum_list_sha256']:
        raise ValueError('Dataset metadata changed')
    for name, digest in config['code_sha256'].items():
        if a.sha(a.REPO_DIR / name) != digest:
            raise ValueError('Code changed: ' + name)
    for name, digest in config['input_sha256'].items():
        if a.sha(a.asset(root, name)) != digest:
            raise ValueError('Data changed: ' + name)


def save_mask(output, name, mask):
    path = output / 'masks' / (name + '.png')
    if path.exists() or not cv2.imwrite(str(path), mask):
        raise ValueError('Mask collision/write failure')
    return str(path.relative_to(output))


def negative_change(clean, out, binary):
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    changed = np.any(clean != out, axis=2)
    return dict(changed_pixels=int(changed.sum()),
                changed_text_proxy_pixels=int((changed & (binary > 0)).sum()),
                **r.region_scores(gray, cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), np.ones_like(binary, dtype=bool)))


def run(root, output):
    root, output = Path(root).resolve(), Path(output)
    config = freeze(root, output)
    frozen_hash = a.sha(output / 'config.json')
    # Config is now durable before any TEST image is decoded or detector called.
    for relative, digest in config['code_sha256'].items():
        data=(a.REPO_DIR/relative).read_bytes()
        dest=output/'code_snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        if a.sha(dest)!=digest:raise ValueError('Source changed during snapshot')
    (output / 'masks').mkdir(); (output / 'previews').mkdir()
    positives, negatives, records, cases, unsupported, previews, neg_rest = [], [], [], [], [], [], []
    for index, row in enumerate(config['samples']):
        sid = row['sample_id']; kind = a.RECIPES[row['recipe_name']][1]
        image = d.read_image(a.asset(root, row['rel_degraded_path']))
        clean = d.read_image(a.asset(root, row['rel_clean_path']))
        binary = d.read_image(a.asset(root, row['rel_clean_binary_path']), gray=True)
        gt = d.read_image(a.asset(root, row['rel_' + kind + '_mask_path']), gray=True)
        if image.shape != clean.shape or image.shape != (row['height'], row['width'], 3):
            raise ValueError('Unexpected TEST frame')
        a.binary_mask(binary, image.shape[:2]); gt_pixels = a.binary_mask(gt, image.shape[:2])
        if not gt_pixels.any(): raise ValueError('Unexpected empty positive GT')
        meta = dict(sample_id=sid, source_id=row['source_id'], split='test', kind=kind,
                    level=row[kind + '_level'], pen_color=row['pen_color'])
        masks = d.predictions(image) if kind == 'stamp' or row['pen_color'] == 'blue' else {}
        if not masks:
            unsupported.append(dict(**meta, status='no_black_pen_detector_GT_diagnostic_only'))
        for name in (('red_color', 'stamp_candidate') if kind == 'stamp' else ('blue_color',) if masks else ()):
            pred = masks[name]
            path = save_mask(output, sid + '_' + name, pred)
            positives.append(dict(**meta, detector=name, mask_path=path,
                input_pixels_sha256=d.array_sha(image), gt_pixels_sha256=d.array_sha(gt),
                prediction_pixels_sha256=d.array_sha(pred), **d.positive_scores(pred, gt)))
            if sid in config['preview_ids']:
                cv2.imwrite(str(output / 'previews' / (sid + '_' + name + '_overlay.png')), d.overlay(image, pred, gt))
        label = json.loads(a.asset(root, row['rel_gt_path']).read_text(encoding='utf-8'))
        crop, reason = focus_crop(row, gt, binary > 0, label)
        for source, mask in r.mask_branches(row, gt, masks).items():
            name = 'stamp_candidate' if source == 'predicted_stamp_candidate' else 'blue_color'
            mask_path = save_mask(output, sid + '_gt', mask) if source == 'gt_diagnostic' else 'masks/' + sid + '_' + name + '.png'
            cases.append(dict(**meta, mask_source=source, mask_path=mask_path,
                              mask_pixels_sha256=d.array_sha(mask)))
            outputs = {}
            for method in r.METHODS:
                result = r.apply_method(image, mask, method); outputs[method] = result
                records.append(dict(**meta, mask_source=source, method=method, radius=3,
                    mask_pixels_sha256=d.array_sha(mask), roi_pixels_sha256=d.array_sha(gt),
                    empty_mask=not bool(np.any(mask)), unchanged_output=np.array_equal(result, image),
                    outside_mask_unchanged=True, input_and_mask_unchanged=True,
                    output_pixels_sha256=d.array_sha(result),
                    predicted_outside_gt_text_proxy_pixels=int(np.count_nonzero((mask > 0) & ~gt_pixels & (binary > 0))),
                    **r.evaluate(clean, result, gt, binary)))
            if sid in config['preview_ids']:
                prefix = sid + '_' + source
                r.save_preview(output / 'previews', prefix, clean, outputs, mask, crop)
                previews.append(dict(sample_id=sid, mask_source=source, crop_xywh=list(crop), reason=reason, resized=False))
        print(f'TEST {index+1}/60 {sid}', flush=True)
    for row in config['controls']:
        clean = d.read_image(a.asset(root, row['rel_clean_path']))
        binary = d.read_image(a.asset(root, row['rel_clean_binary_path']), gray=True)
        a.binary_mask(binary, clean.shape[:2])
        for name, mask in d.predictions(clean).items():
            path = save_mask(output, row['sample_id'] + '_negative_' + name, mask)
            negatives.append(dict(source_id=row['source_id'], split='test', detector=name, mask_path=path,
                prediction_pixels_sha256=d.array_sha(mask), **d.negative_scores(mask)))
            if name != 'red_color':
                for method in r.METHODS:
                    out = r.apply_method(clean, mask, method)
                    neg_rest.append(dict(source_id=row['source_id'], detector=name, method=method,
                        mask_pixels_sha256=d.array_sha(mask), output_pixels_sha256=d.array_sha(out),
                        outside_mask_unchanged=True, **negative_change(clean, out, binary)))
        print('TEST clean control: ' + row['source_id'], flush=True)
    actual = dict(samples=len(config['samples']), stamp_samples=30, blue_pen_samples=16, black_pen_samples=14,
                  detection_rows=len(positives), negative_rows=len(negatives), mask_cases=len(cases),
                  restoration_rows=len(records), negative_restoration_rows=len(neg_rest), unsupported_rows=len(unsupported))
    if actual != config['expected']: raise ValueError('Incomplete benchmark grid')
    for name, rows in [('detection.csv', positives), ('detection_summary.csv', detection_summary(positives)),
        ('negative_controls.csv', negatives), ('negative_summary.csv', d.summarize([], negatives)[1]),
        ('per_image.csv', records), ('summary.csv', restoration_summary(records)),
        ('unsupported.csv', unsupported), ('negative_restoration.csv', neg_rest)]:
        d.csv_new(output / name, rows)
    d.json_new(output / 'mask_cases.json', cases); d.json_new(output / 'previews.json', previews)
    check_frozen(config, root)
    if a.sha(output / 'config.json') != frozen_hash: raise ValueError('Frozen config changed')
    d.json_new(output / 'completion.json', dict(complete=True, counts=actual, config_sha256=frozen_hash,
        ocr_run=False, cer_computed=False,
        artifact_sha256={str(p.relative_to(output)):a.sha(p) for p in output.rglob('*') if p.is_file()}))


def load_csv(path):
    with path.open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))


def compare_table(expected, saved):
    if len(expected) != len(saved): raise ValueError('Summary size mismatch')
    for x,y in zip(expected,saved):
        for key,value in x.items():
            if value is None:
                if y[key] != '':raise ValueError('N/A mismatch')
            elif isinstance(value,float):
                if not np.isclose(float(y[key]),value,rtol=1e-12,atol=1e-12):raise ValueError('Summary value mismatch')
            elif str(value)!=y[key]:raise ValueError('Summary metadata mismatch')


def verify(root, output):
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    done = json.loads((output / 'completion.json').read_text(encoding='utf-8'))
    if not done['complete'] or a.sha(output / 'config.json') != done['config_sha256']: raise ValueError('Config/completion mismatch')
    check_frozen(config, root)
    for name, digest in done['artifact_sha256'].items():
        if a.sha(output / name) != digest: raise ValueError('Artifact changed: ' + name)
    rows = load_csv(output / 'per_image.csv'); detection = load_csv(output / 'detection.csv')
    cases = json.loads((output / 'mask_cases.json').read_text(encoding='utf-8'))
    by_sample = {x['sample_id']:x for x in config['samples']}
    if len(rows) != 424 or len(cases) != 106 or len(detection) != 76: raise ValueError('Wrong grid')
    if len({(x['sample_id'],x['mask_source'],x['method']) for x in rows}) != 424: raise ValueError('Duplicate rows')
    numeric = []
    for x in rows:
        y = dict(x)
        for k in ('roi_mae','roi_mse','text_proxy_mae','text_proxy_mse'):
            y[k] = float(x[k]) if x[k] else None
        for k in ('roi_n_pixels','text_proxy_n_pixels'): y[k] = int(x[k])
        y['level'] = int(x['level']); y['empty_mask'] = x['empty_mask'] == 'True'
        numeric.append(y)
    expected = restoration_summary(numeric)
    saved = load_csv(output / 'summary.csv')
    compare_table(expected,saved)
    det_numeric=[]
    for x in detection:
        y=dict(x,level=int(x['level']),empty_prediction=x['empty_prediction']=='True')
        for k in ('iou','precision','recall'):y[k]=float(x[k])
        det_numeric.append(y)
    compare_table(detection_summary(det_numeric),load_csv(output/'detection_summary.csv'))
    masks_checked = 0; replayed = 0
    for case in cases:
        sample = by_sample[case['sample_id']]
        kind = case['kind']
        mask = d.read_image(output / case['mask_path'], gray=True)
        gt = d.read_image(root / sample['rel_' + kind + '_mask_path'], gray=True)
        binary = d.read_image(root / sample['rel_clean_binary_path'], gray=True)
        d.mask_pixels(mask); d.mask_pixels(gt); d.mask_pixels(binary)
        subset = [x for x in rows if x['sample_id'] == case['sample_id'] and x['mask_source'] == case['mask_source']]
        if (len(subset) != 4 or {x['method'] for x in subset} != set(r.METHODS)
                or any(x['mask_pixels_sha256'] != d.array_sha(mask) or x['roi_pixels_sha256'] != d.array_sha(gt) for x in subset)
                or any(int(x['text_proxy_n_pixels']) != int(((gt>0)&(binary>0)).sum()) for x in subset)):
            raise ValueError('Mask/ROI/method pairing mismatch')
        masks_checked += 1
        if case['sample_id'] in config['preview_ids']:
            image = d.read_image(root / sample['rel_degraded_path']); clean = d.read_image(root / sample['rel_clean_path'])
            for x in subset:
                out = r.apply_method(image, mask, x['method'])
                if d.array_sha(out) != x['output_pixels_sha256']: raise ValueError('Replay pixels differ')
                for key, score in r.evaluate(clean, out, gt, binary).items():
                    if score is None:
                        if x[key] != '': raise ValueError('Replay N/A mismatch')
                    elif not np.isclose(float(x[key]),score,rtol=1e-12,atol=1e-12): raise ValueError('Replay score differs')
                replayed += 1
    for x in detection:
        sample = by_sample[x['sample_id']]
        mask = d.read_image(output / x['mask_path'], gray=True)
        gt = d.read_image(root / sample['rel_' + x['kind'] + '_mask_path'], gray=True)
        if d.array_sha(mask) != x['prediction_pixels_sha256']: raise ValueError('Detection mask hash mismatch')
        for key, value in d.positive_scores(mask, gt).items():
            if isinstance(value, bool):
                if str(value) != x[key]: raise ValueError('Detection flag mismatch')
            elif not np.isclose(float(x[key]),value,rtol=1e-12,atol=1e-12): raise ValueError('Detection metric mismatch')
    negatives = load_csv(output / 'negative_controls.csv')
    if len(negatives) != 90 or len({(x['source_id'],x['detector']) for x in negatives}) != 90: raise ValueError('Negative grid')
    for x in negatives:
        mask = d.read_image(output / x['mask_path'], gray=True)
        for key, value in d.negative_scores(mask).items():
            if value is None:
                if x[key] != '': raise ValueError('Negative N/A mismatch')
            elif isinstance(value, bool):
                if str(value) != x[key]: raise ValueError('Negative flag mismatch')
            elif not np.isclose(float(x[key]),value): raise ValueError('Negative metric mismatch')
    neg_numeric=[]
    for x in negatives:
        y=dict(x,fp_pixels=int(x['fp_pixels']),fp_pixel_ratio=float(x['fp_pixel_ratio']),
               any_false_positive=x['any_false_positive']=='True',fp_components_8_connected=int(x['fp_components_8_connected']))
        neg_numeric.append(y)
    compare_table(d.summarize([],neg_numeric)[1],load_csv(output/'negative_summary.csv'))
    negative_restore=load_csv(output/'negative_restoration.csv')
    if (len(negative_restore)!=240 or len({(x['source_id'],x['detector'],x['method']) for x in negative_restore})!=240):
        raise ValueError('Negative restoration grid')
    negative_replayed=0
    for sample in (config['controls'][0],config['controls'][-1]):
        clean=d.read_image(root/sample['rel_clean_path']);binary=d.read_image(root/sample['rel_clean_binary_path'],gray=True)
        predictions=d.predictions(clean)
        for x in (x for x in negative_restore if x['source_id']==sample['source_id']):
            mask=predictions[x['detector']];out=r.apply_method(clean,mask,x['method'])
            if d.array_sha(mask)!=x['mask_pixels_sha256'] or d.array_sha(out)!=x['output_pixels_sha256']:
                raise ValueError('Negative restoration replay changed')
            for key,value in negative_change(clean,out,binary).items():
                if not np.isclose(float(x[key]),value,rtol=1e-12,atol=1e-12):raise ValueError('Negative restoration metric changed')
            negative_replayed+=1
    proof = dict(status='PASS_OFFICIAL_IMAGE_MEASUREMENTS_NO_CER', detection_masks_recomputed=76,
        restoration_cases_checked=masks_checked, restoration_rows=424, summary_recomputed=True,
        negative_masks_recomputed=90, replayed_restoration_outputs=replayed,
        detection_and_negative_summaries_recomputed=True,negative_restoration_rows=240,
        negative_restoration_outputs_replayed=negative_replayed,
        replay_rule=config['policy']['replay'], full_restoration_rerun=False, cer_computed=False,
        config_sha256=a.sha(output/'config.json'))
    d.json_new(output / 'verification.json', proof)
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=a.REPO_DIR/'data/6_synthetic')
    p.add_argument('--output-dir', type=Path, default=a.REPO_DIR/'results/t4/official_test_v1')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if args.verify_only: print(json.dumps(verify(args.data_dir,args.output_dir)))
    else: run(args.data_dir,args.output_dir)


if __name__ == '__main__': main()
