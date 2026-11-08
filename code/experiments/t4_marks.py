"""T4 session 1: data/annotation inventory only, no detector or benchmark."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common.io import read_image
from common.paths import REPO_DIR

MANIFEST_SHA = '05dd1a8da9268610d5194e334d7173716865bc14ac6d2da333f9fdb8e4dcd8ad'
CHECKSUM_SHA = '39dbf3164f0e426beae4b2884bc0a71c137ccc2953651b3307f77179abd503fc'
SPLITS = {'train': 60, 'val': 10, 'test': 30}
RECIPES = {'single_stamp': (11, 'stamp'), 'single_pen': (12, 'pen')}
FILES = ('step1_inventory.json', 'step1_samples.csv', 'step1_ocr_pool.csv')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError('Empty asset path')
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('Missing or out-of-root asset: ' + relative)
    return path


def select_samples(manifest):
    if manifest['version'] != 'v2' or manifest['n_sources'] != 100:
        raise ValueError('Expected v2 / 100 sources')
    all_rows = manifest['samples']
    if len({r['sample_id'] for r in all_rows}) != len(all_rows):
        raise ValueError('Duplicate sample ID')
    memberships = defaultdict(set)
    for row in all_rows:
        memberships[row['source_id']].add(row['split'])
    if len(memberships) != 100 or any(len(s) != 1 for s in memberships.values()):
        raise ValueError('Wrong source count or cross-split source')
    selected = []
    for name, (variant, kind) in RECIPES.items():
        rows = [r for r in all_rows if r['recipe_name'] == name]
        if (Counter(r['source_id'] for r in rows) != Counter({sid:1 for sid in memberships})
                or dict(Counter(r['split'] for r in rows)) != SPLITS):
            raise ValueError('Expected one sample per source and split 60/10/30')
        for row in rows:
            if row['variant_index'] != variant or row['applied_degradations'] != kind:
                raise ValueError('Wrong single-degradation recipe')
            for field in ('angle_deg','perspective_strength','blur_level','shadow_level','glare_level','jpeg_noise_level'):
                if row[field] != 0:
                    raise ValueError('Geometry or additional degradation confound')
            if row['exposure_gain'] != 1 or row['pen_level' if kind=='stamp' else 'stamp_level'] != 0:
                raise ValueError('Exposure or mark confound')
            if row[kind+'_level'] not in ((1,2,3) if kind=='stamp' else (1,2)):
                raise ValueError('Invalid mark level')
            selected.append(row)
    return sorted(selected, key=lambda r:(r['split'],r['source_id'],r['variant_index']))


def binary_mask(image, shape):
    if (image.dtype != np.uint8 or image.ndim != 2 or image.shape != shape
            or not np.all((image==0)|(image==255))):
        raise ValueError('Expected same-frame uint8 0/255 mask')
    return image > 0


def inspect_box(box, mask, clean_ink):
    """OCR pool only: no transcript is human-confirmed by this function."""
    result = dict(box_id=box.get('id'), label=box.get('label'),
                  transcript_present=isinstance(box.get('text'),str) and bool(box['text'].strip()),
                  transcript_review='unverified_source_annotation', bbox_valid=False,
                  intersects_mark=False, overlapped_reference_ink_pixels=0)
    bbox=box.get('bbox')
    if (not isinstance(bbox,list) or len(bbox)!=4 or
            any(type(v) is not int for v in bbox)):
        result['issue']='non_integer_xywh'; return result
    x,y,w,h=bbox
    if x<0 or y<0 or w<=0 or h<=0 or x+w>mask.shape[1] or y+h>mask.shape[0]:
        result['issue']='out_of_frame_or_empty'; return result
    result.update(bbox_valid=True, intersects_mark=bool(mask[y:y+h,x:x+w].any()),
                  overlapped_reference_ink_pixels=int(np.count_nonzero(
                      mask[y:y+h,x:x+w] & clean_ink[y:y+h,x:x+w])))
    result['issue']='' if result['transcript_present'] else 'missing_text'
    return result


def validate_box_mapping(clean, degraded):
    """Identity geometry permits extra degraded metadata, not changed text/coords."""
    by_id={b['id']:b for b in degraded}
    if len({b['id'] for b in clean})!=len(clean) or len(by_id)!=len(degraded) or set(by_id)!={b['id'] for b in clean}:
        raise ValueError('Box ID mapping disagrees')
    for box in clean:
        mapped=by_id[box['id']]
        if any(box.get(k)!=mapped.get(k) for k in ('id','label','text','mapping')):
            raise ValueError('Box annotation mapping disagrees')
        x,y,w,h=box['bbox']
        expected=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
        if not np.allclose(np.asarray(mapped['polygon']),np.asarray(expected),rtol=0,atol=1e-6):
            raise ValueError('Box coordinates not identity-mapped')
        if not mapped.get('clipped',False) and mapped['bbox']!=box['bbox']:
            raise ValueError('Unclipped box bbox disagrees')


def inventory(root):
    root=Path(root).resolve()
    if sha(root/'manifest_v2.json')!=MANIFEST_SHA or sha(root/'checksums_v2.sha256')!=CHECKSUM_SHA:
        raise ValueError('Dataset version or delivered checksum list changed')
    manifest=json.loads((root/'manifest_v2.json').read_text(encoding='utf-8'))
    samples=select_samples(manifest)
    checksums={}
    for line in (root/'checksums_v2.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split(maxsplit=1)
        name=name.lstrip('*')
        if name in checksums and checksums[name]!=digest:
            raise ValueError('Conflicting checksum entries')
        checksums[name]=digest
    evidence={}; records=[]; pool=[]; source_clean={}; source_boxes={}
    for row in samples:
        sid=row['source_id']; split=row['split']; kind=RECIPES[row['recipe_name']][1]
        keys=('rel_degraded_path','rel_clean_path','rel_clean_binary_path','rel_gt_path',
              'rel_stamp_mask_path','rel_pen_mask_path','rel_glare_mask_path')
        for key in keys:
            relative=row[key]; path=asset(root,relative); digest=sha(path)
            if checksums.get(relative)!=digest:
                raise ValueError('Missing or changed delivered checksum: '+relative)
            evidence[relative]=digest
        label=json.loads(asset(root,row['rel_gt_path']).read_text(encoding='utf-8'))
        for key in ('sample_id','source_id','split','recipe_name','variant_index'):
            if label[key]!=row[key]: raise ValueError('Label provenance mismatch: '+key)
        shape=(row['height'],row['width'])
        if label['image_size']!=[shape[1],shape[0]]:
            raise ValueError('Label/image frame mismatch')
        for key in ('H','H_clean_to_degraded','H_degraded_to_clean'):
            if not np.array_equal(np.asarray(label[key]),np.eye(3)):
                raise ValueError('Unexpected nonidentity geometry')
        expected_levels={k:row[k+'_level'] if k in ('stamp','pen') else 0
                         for k in ('perspective','rotate','blur','shadow','glare','stamp','pen','jpeg_noise')}
        if label['degradation_levels']!=expected_levels or label['active_masks']!=[kind+'_mask']:
            raise ValueError('Additional degradation or active mask mismatch')
        # Inspect PNG headers only; never pass test pages through a detector.
        for key in ('rel_degraded_path','rel_clean_path'):
            with Image.open(asset(root,row[key])) as image:
                if image.size!=(shape[1],shape[0]) or image.mode!='RGB':
                    raise ValueError('Unexpected color image header')
        masks={}
        for target in ('stamp','pen','glare'):
            relative=row['rel_'+target+'_mask_path']
            if str(Path(split)/label['masks'][target+'_mask']).replace('\\','/')!=relative:
                raise ValueError('Mask paths disagree')
            masks[target]=binary_mask(read_image(asset(root,relative),gray=True),shape)
            if bool(masks[target].any()) != (target==kind):
                raise ValueError('Positive mask empty or inactive mask nonempty')
        clean_ink=binary_mask(read_image(asset(root,row['rel_clean_binary_path']),gray=True),shape)
        clean_hash=evidence[row['rel_clean_path']]
        binary_hash=evidence[row['rel_clean_binary_path']]
        if sid in source_clean and source_clean[sid]!=(clean_hash,binary_hash):
            raise ValueError('v11/v12 clean references differ')
        source_clean[sid]=(clean_hash,binary_hash)
        boxes=label['boxes_clean']
        validate_box_mapping(boxes,label['boxes_degraded'])
        if sid in source_boxes and source_boxes[sid]!=boxes:
            raise ValueError('v11/v12 box/transcript source mismatch')
        source_boxes[sid]=boxes
        mark=masks[kind]
        details=[inspect_box(box,mark,clean_ink) for box in boxes]
        for box,d in zip(boxes,details,strict=True):
            if d['intersects_mark'] or d['issue']:
                pool.append(dict(sample_id=row['sample_id'],source_id=sid,split=split,kind=kind,
                                 level=row[kind+'_level'],pen_color=row['pen_color'],
                                 rel_gt_path=row['rel_gt_path'],bbox_json=json.dumps(box['bbox']),
                                 text_sha256=hashlib.sha256(str(box.get('text','')).encode()).hexdigest(),
                                 text_characters=len(box.get('text') or ''),**d))
        records.append(dict(sample_id=row['sample_id'],source_id=sid,split=split,kind=kind,
                            level=row[kind+'_level'],pen_color=row['pen_color'],
                            width=shape[1],height=shape[0],gt_mark_pixels=int(mark.sum()),
                            gt_overlapped_reference_ink_pixels=int(np.count_nonzero(mark & clean_ink)),
                            boxes=len(boxes),invalid_boxes=sum(not d['bbox_valid'] for d in details),
                            missing_transcripts=sum(not d['transcript_present'] for d in details),
                            touched_boxes=sum(d['intersects_mark'] for d in details),
                            eligible_ocr_boxes=sum(d['bbox_valid'] and d['transcript_present'] and
                                                   d['overlapped_reference_ink_pixels']>0 for d in details)))
    count=lambda field: {str(k):v for k,v in Counter(field(r) for r in records).items()}
    report=dict(status='PASS_DATA_INVENTORY_NOT_BENCHMARK',manifest_sha256=MANIFEST_SHA,
                checksum_list_sha256=CHECKSUM_SHA,samples=len(records),sources=len(source_clean),
                counts_by_kind_split_level_color=count(lambda r:(r['kind'],r['split'],r['level'],r['pen_color'])),
                sums={k:sum(r[k] for r in records) for k in ('boxes','invalid_boxes','missing_transcripts','touched_boxes','eligible_ocr_boxes')},
                empty_text_overlap_samples=[r['sample_id'] for r in records if not r['gt_overlapped_reference_ink_pixels']],
                unique_files_verified=len(evidence),assets_sha256=evidence,
                negative_controls='100 unique clean sources, deduplicated across v11/v12 by source_id and exact clean/binary hashes',
                transcript_status='Inherited annotations, not human-confirmed CER references',
                test_pixel_scope='GT masks and clean_binary decoded for inventory/overlap only; color page headers inspected; no detector/inpainting/OCR',
                benchmark_run=False,ocr_runtime=dict(tesseract_on_path=shutil.which('tesseract'),
                    known_windows_binary_exists=Path('C:/Program Files/Tesseract-OCR/tesseract.exe').is_file(),
                    python_packages={k:importlib.util.find_spec(k) is not None for k in ('pytesseract','paddleocr','vietocr')}))
    return report,records,pool


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=REPO_DIR/'data/6_synthetic')
    parser.add_argument('--output-dir',type=Path,default=REPO_DIR/'results/t4')
    args=parser.parse_args(argv)
    if any((args.output_dir/name).exists() for name in FILES):
        raise FileExistsError('Inventory exists; choose a new output directory')
    report,records,pool=inventory(args.data_dir)
    report['inventory_script_sha256']=sha(Path(__file__))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,data in zip(FILES[1:],(records,pool),strict=True):
        with (args.output_dir/name).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    with (args.output_dir/FILES[0]).open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='assets_sha256'},indent=2))


if __name__=='__main__':
    main()
