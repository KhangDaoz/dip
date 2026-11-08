"""T3 paired degradation curves with frozen T1/T2 recipes; fail-closed inputs."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import platform
import statistics

import cv2
import numpy as np
import skimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from t1_t2_compat import checked_asset, file_sha
from t1_t2_compat import validate_t1_inputs
from common.io import read_image
from common.paths import REPO_DIR
from cleaning.binarize import METHODS, otsu
from cleaning.metrics import BinaryReference, _binary
from synthetic import degrade
import t1_shadow as t1

GROUPS = {'shadow_planar': range(4), 'shadow_silhouette': range(4),
          'blur_gaussian': range(5), 'blur_motion': range(5), 'stamp': range(4)}
SPLITS = {'train': 60, 'val': 10, 'test': 30}
LOCKS = {
    't1_config': '2719cb5eb7ecc03b5498e910903f1d866608b5ebf1d9d4ebe8bfe2fd59c23e2e',
    't1_selection': '6e4acd656471b6a3318f2d19f7116b291bc76b2cde2a340a988377502d9484c2',
    't2_config': '9688436f57a1ab87d5f1f22f2832274b97208f6a8cc6aa8ba99980643223a1b2',
    't2_selection': '04b58c8b09d83e6f852970d19a173415829f7587e227aabcf0c993460d8a1a98',
}


def seed_for(source_id, group, stream='geometry', seed=42):
    """No level in token: each level restarts its own RNG from this seed."""
    if group not in GROUPS or not source_id:
        raise ValueError('Invalid source/group')
    token = f'{seed}:t3_v1:{source_id}:{group}:{stream}'
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little')


def sources_from_manifest(manifest):
    """Validate metadata, NOT image decoding or pixel identity."""
    if manifest.get('version') != 'v2':
        raise ValueError('Expected v2')
    rows = manifest['samples']
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample')
    groups = {}
    for row in rows:
        groups.setdefault(row['source_id'], []).append(row)
    if len(groups) != 100 or manifest['n_sources'] != 100:
        raise ValueError('Expected 100 sources')
    selected = []
    for sid, values in sorted(groups.items()):
        if len({r['split'] for r in values}) != 1:
            raise ValueError('Source crosses splits')
        base = [r for r in values if r['variant_index'] == 0]
        if len(base) != 1 or base[0]['split'] not in SPLITS:
            raise ValueError('Expected one v00 provenance row')
        row = base[0]
        if row['width'] <= 0 or row['height'] <= 0:
            raise ValueError('Invalid dimensions')
        selected.append({k: row[k] for k in ('source_id', 'split', 'sample_id',
                        'width', 'height', 'rel_clean_path', 'rel_clean_binary_path', 'rel_gt_path')})
    if dict(Counter(r['split'] for r in selected)) != SPLITS:
        raise ValueError('Expected source splits 60/10/30')
    if Counter(r['split'] for r in rows) != Counter(manifest['split_counts']):
        raise ValueError('Sample split counts disagree')
    return selected


def plan_rows(sources):
    """22 curve slots/source; zero slots share one physical clean input."""
    if len({s['source_id'] for s in sources}) != len(sources):
        raise ValueError('Duplicate source')
    return [dict(source_id=s['source_id'], split=s['split'], group=g, level=level,
                 generation_seed=seed_for(s['source_id'], g),
                 input_key=f"{s['source_id']}:{g}:{level}" if level else f"{s['source_id']}:clean",
                 H_clean_to_derived=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            for s in sources for g, levels in GROUPS.items() for level in levels]


def load_frozen(paths):
    """Pin these exact historical files; no silent fallback or retuning."""
    for key, expected in LOCKS.items():
        if file_sha(paths[key]) != expected:
            raise ValueError(f'Frozen artifact changed: {key}')
    data = {key: json.loads(Path(path).read_text(encoding='utf-8')) for key, path in paths.items()}
    recipes = data['t2_selection']['frozen_recipes']
    pairs = {(r['method'], r['branch']) for r in recipes}
    expected = {(m, b) for m in ('otsu', 'niblack', 'sauvola', 'wolf', 'gaussian') for b in ('none', 't1')}
    if len(recipes) != 10 or pairs != expected:
        raise ValueError('Expected ten frozen recipes')
    if data['t2_config']['t1_selection_sha256'] != LOCKS['t1_selection']:
        raise ValueError('T1 binding mismatch')
    return data


def audit_assets(root, sources):
    """Hash reference bytes and check label provenance; does not decode pixels."""
    evidence = []
    for source in sources:
        hashes = {}
        for key in ('rel_clean_path', 'rel_clean_binary_path', 'rel_gt_path'):
            path = checked_asset(root, source[key])
            hashes[key] = file_sha(path)
        label = json.loads(checked_asset(root, source['rel_gt_path']).read_text(encoding='utf-8'))
        if any(label[k] != source[k] for k in ('source_id', 'split', 'sample_id')):
            raise ValueError('Label provenance mismatch')
        if label['image_size'] != [source['width'], source['height']]:
            raise ValueError('Label dimensions mismatch')
        evidence.append(dict(source_id=source['source_id'], hashes=hashes))
    return evidence


def verify_reference_checksums(root, sources, evidence):
    """Compare all 300 consumed references with delivered checksum lists."""
    expected = {}
    for name in ('checksums_v1.sha256', 'checksums_v2.sha256'):
        for line in (Path(root) / name).read_text(encoding='utf-8').splitlines():
            digest, relative = line.split(maxsplit=1)
            relative = relative.lstrip('*')
            if relative in expected and expected[relative] != digest:
                raise ValueError('Conflicting checksum entries')
            expected[relative] = digest
    by_id = {e['source_id']: e for e in evidence}
    for source in sources:
        for key, actual in by_id[source['source_id']]['hashes'].items():
            if expected.get(source[key]) != actual:
                raise ValueError('Missing or changed reference checksum: ' + source[key])
    return sum(len(by_id[s['source_id']]['hashes']) for s in sources)


EXPECTED_MANIFEST = '05dd1a8da9268610d5194e334d7173716865bc14ac6d2da333f9fdb8e4dcd8ad'
EXPECTED_CHECKSUM_LISTS = {
    'checksums_v1.sha256': '0d699fdcefe3edb0df93533bd0d981762ef628f0c56f7634424f04a27b89136c',
    'checksums_v2.sha256': '39dbf3164f0e426beae4b2884bc0a71c137ccc2953651b3307f77179abd503fc',
}
STAMP_POLICY = 'canonical_max_radius_alpha_v1_linear_resize_integer_fixed_anchor'


def array_sha(array):
    """Hash includes dtype/shape so byte streams with different frames differ."""
    header = json.dumps([str(array.dtype), list(array.shape)]).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def check_color(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32):
        raise ValueError('Expected uint8 BGR image, both dimensions >=32')


def stamp_context(image, source_id, region=None):
    """One rotated/textured maximum-radius alpha per source, not per level.

    Reuse the group's renderer and multiply compositing; only the scheduling of
    RNG, placement and template resampling belong to this experiment.
    """
    check_color(image)
    h, w = image.shape[:2]
    seeds = {name: seed_for(source_id, 'stamp', name)
             for name in ('geometry', 'texture', 'color')}
    rng = np.random.default_rng(seeds['geometry'])
    jitter = float(rng.uniform(0.9, 1.1))
    radius = 0.14 * min(h, w) * jitter
    alpha, angle = degrade._stamp_alpha(radius, np.random.default_rng(seeds['texture']), degrade.STAMP_TEXT)
    # One integer anchor; avoid half-pixel center shifts between odd/even sizes.
    margin = int(np.ceil(alpha.shape[0] / 2)) + 1
    center = degrade._pick_center(h, w, rng, region=region, margin=margin)
    center = [int(round(c)) for c in center]
    color_rng = np.random.default_rng(seeds['color'])
    color = [int(np.clip(c + color_rng.integers(-15, 16), 0, 255)) for c in degrade.RED_INK]
    return dict(alpha=alpha, source_id=source_id, image_shape=list(image.shape),
                seeds=seeds, center=center, rotation_deg=float(angle),
                color_bgr=color, radius_jitter=jitter, canonical_radius=radius,
                canonical_alpha_sha256=array_sha(alpha), policy=STAMP_POLICY)


def generate(image, source_id, group, level, context=None):
    """Exactly one perturbation; level zero never calls a degradation function."""
    check_color(image)
    if group not in GROUPS or type(level) is not int or level not in GROUPS[group]:
        raise ValueError('Invalid group/level')
    seed = seed_for(source_id, group)
    meta = dict(group=group, level=level, generation_seed=seed,
                H_clean_to_derived=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if level == 0:
        return image.copy(), dict(meta, applied='none')
    if group.startswith('shadow_'):
        out, gt = degrade.shadow(image, level=level, rng=np.random.default_rng(seed),
                                 shadow_type=group.removeprefix('shadow_'))
    elif group.startswith('blur_'):
        out, gt = degrade.blur(image, level=level, rng=np.random.default_rng(seed),
                               kind=group.removeprefix('blur_'))
    else:
        if context is None:
            context = stamp_context(image, source_id)
        if context['source_id'] != source_id or context['image_shape'] != list(image.shape):
            raise ValueError('Stamp context belongs to another source/frame')
        scale = degrade.STAMP_LEVELS[level]['radius_frac'] / 0.14
        base = context['alpha']
        size = max(1, int(round(base.shape[0] * scale)))
        small = cv2.resize(base, (size, size), interpolation=cv2.INTER_LINEAR)
        alpha = np.zeros(image.shape[:2], np.float32)
        cx, cy = context['center']
        x0, y0 = cx - size // 2, cy - size // 2
        if x0 < 0 or y0 < 0 or x0 + size > image.shape[1] or y0 + size > image.shape[0]:
            raise ValueError('Stamp template would be clipped')
        alpha[y0:y0+size, x0:x0+size] = small * degrade.STAMP_LEVELS[level]['opacity']
        out = degrade._multiply_ink(image, alpha, context['color_bgr'])
        gt = {k: v for k, v in context.items() if k != 'alpha'}
        gt.update(radius=context['canonical_radius'] * scale,
                  opacity=degrade.STAMP_LEVELS[level]['opacity'],
                  radius_frac=degrade.STAMP_LEVELS[level]['radius_frac'],
                  rendered_size=size, interpolation='INTER_LINEAR',
                  mask_sha256=array_sha((alpha > 0.15).astype(np.uint8) * 255))
    if out.shape != image.shape or out.dtype != np.uint8:
        raise ValueError('Generator changed frame/type')
    # Record array hashes instead of writing masks or huge JSON arrays.
    details = {k: (dict(shape=list(v.shape), sha256=array_sha(v)) if isinstance(v, np.ndarray) else v)
               for k, v in gt.items()}
    return out, dict(meta, applied=group, parameters=details)


def read_reference(root, source):
    color = read_image(checked_asset(root, source['rel_clean_path']))
    binary = read_image(checked_asset(root, source['rel_clean_binary_path']), gray=True)
    check_color(color)
    if color.shape[:2] != (source['height'], source['width']) or binary.shape != color.shape[:2]:
        raise ValueError('Reference pixel frame mismatch')
    _binary(binary)
    if not np.array_equal(binary, otsu(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))):
        raise ValueError('Reference not equal to Otsu(gray(clean))')
    return color, binary


def preflight(root, paths, archive):
    """Byte/metadata only; no image decoding and no output writes."""
    root = Path(root).resolve()
    manifest = checked_asset(root, 'manifest_v2.json')
    if file_sha(manifest) != EXPECTED_MANIFEST:
        raise ValueError('T3 manifest differs from locked step-1 version')
    for name, expected in EXPECTED_CHECKSUM_LISTS.items():
        if file_sha(checked_asset(root, name)) != expected:
            raise ValueError('Delivered checksum list changed: ' + name)
    frozen = load_frozen(paths)
    sources = sources_from_manifest(json.loads(manifest.read_text(encoding='utf-8')))
    evidence = audit_assets(root, sources)
    count = verify_reference_checksums(root, sources, evidence)
    versions = dict(python=platform.python_version(), opencv=cv2.__version__,
                    numpy=np.__version__, skimage=skimage.__version__)
    prior = frozen['t1_config']
    samples = t1.load_v2(dict(prior, data_dir=str(root)))
    compat = validate_t1_inputs(paths['t1_selection'], root, samples, versions, archive)
    # Metrics used by T3 must also match the T2 artifact, not merely today's hash.
    metric_path = REPO_DIR / 'code/src/cleaning/metrics.py'
    t2_config = frozen['t2_config']
    expected_metric = next((v for k, v in t2_config['code_sha256'].items()
                            if k.replace('\\', '/') == 'code/src/cleaning/metrics.py'), None)
    if file_sha(metric_path) != expected_metric:
        raise ValueError('Metrics differ from frozen T2')
    files = [Path(__file__), Path(__file__).with_name('t1_t2_compat.py'),
             REPO_DIR / 'code/src/synthetic/degrade.py', metric_path,
             REPO_DIR / 'code/src/cleaning/background.py', REPO_DIR / 'code/src/cleaning/binarize.py',
             REPO_DIR / 'code/src/common/io.py']
    report = dict(manifest_sha256=file_sha(manifest), versions=versions,
                  reference_files_checked=count, reference_evidence=evidence,
                  checksum_lists={n: file_sha(root / n) for n in ('checksums_v1.sha256', 'checksums_v2.sha256')},
                  frozen_artifacts={k: file_sha(p) for k, p in paths.items()},
                  code_sha256={str(p.relative_to(REPO_DIR)): file_sha(p) for p in files},
                  t1_compatibility=compat, stamp_policy=STAMP_POLICY,
                  reference='auto Otsu; not manual GT, OCR, T4 or T5',
                  pixel_checks_run=False, benchmark_run=False)
    return sources, frozen, report


def predict(gray, recipe, shadow_recipe):
    image = gray if recipe['branch'] == 'none' else t1.correct(gray, **shadow_recipe)
    result = METHODS[recipe['method']](image, **recipe['params'])
    _binary(result)
    if result.shape != gray.shape:
        raise ValueError('Binarizer changed frame')
    return result


def train_sanity(root, sources, frozen, contract, output):
    """Exactly two lexicographically first TRAIN sources; never val/test pixels."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Output already exists; choose a new directory')
    selected = sorted((s for s in sources if s['split'] == 'train'), key=lambda s: s['source_id'])[:2]
    if len(selected) != 2:
        raise ValueError('Need two train sources')
    recipes = frozen['t2_selection']['frozen_recipes']
    shadow_recipe = {k: frozen['t1_selection']['winner'][k] for k in ('method', 'kernel')}
    output.mkdir(parents=True)
    contract = dict(contract, phase='train_sanity', selected_sources=selected,
                    frozen_recipes=recipes, shadow_recipe=shadow_recipe,
                    expected_rows=440, complete=False)
    def save(name, obj):
        (output / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    # Freeze before decoding even train images; failure leaves complete=False.
    save('config.json', contract)
    records, samples, checks = [], [], []
    for index, source in enumerate(selected):
        color, binary = read_reference(root, source)
        original_hash = array_sha(color)
        reference = BinaryReference(binary)
        label = json.loads(checked_asset(root, source['rel_gt_path']).read_text(encoding='utf-8'))
        context = stamp_context(color, source['source_id'], label.get('medicine_region_clean'))
        cv2.imwrite(str(output / f'source_{index+1}_original.png'), color)
        for group, levels in GROUPS.items():
            for level in levels:
                out, metadata = generate(color, source['source_id'], group, level, context)
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                if array_sha(color) != original_hash or (level == 0 and not np.array_equal(color, out)):
                    raise ValueError('Mutation or invalid level zero')
                samples.append(dict(source_id=source['source_id'], split='train',
                                    provenance=source, bgr_sha256=array_sha(out), gray_sha256=array_sha(gray),
                                    reference_sha256=array_sha(binary), **metadata))
                for recipe in recipes:
                    predicted = predict(gray, recipe, shadow_recipe)
                    # Reuse validated metric implementation; retain only F1/P/R here.
                    score = reference.score(predicted)
                    records.append(dict(source_id=source['source_id'], split='train', group=group,
                                        level=level, candidate=recipe['candidate'], method=recipe['method'],
                                        branch=recipe['branch'], params_json=recipe['params_json'],
                                        f1=score['f1'], precision=score['precision'], recall=score['recall'],
                                        predicted_sha256=array_sha(predicted), gray_sha256=array_sha(gray)))
                    # First source, maximum level, two predeclared recipes only.
                    if index == 0 and level == levels[-1] and recipe['candidate'] in (
                            't1__otsu__base', 'none__sauvola__k0.3_window31'):
                        cv2.imwrite(str(output / f'{group}_{recipe["candidate"]}.png'), 255-predicted)
                if index == 0 and level == levels[-1]:
                    cv2.imwrite(str(output / f'{group}_degraded.png'), out)
            print(f'train sanity: source {index+1}/2 {group}', flush=True)
        checks.append(dict(source_id=source['source_id'], input_unchanged=True,
                           reference_matches_otsu=True, all_shapes_types_valid=True))
    if len(records) != 440 or len(samples) != 44:
        raise ValueError('Incomplete sanity grid')
    with (output / 'per_image.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    save('samples.json', samples)
    save('checks.json', checks)
    save('completion.json', dict(complete=True, rows=len(records), variants=len(samples),
                                 test_pixels_read=False, val_pixels_read=False,
                                 note='Sanity only; no model/parameter selection or formal curves'))


FORMAL_FILES = ('config.json', 'samples.json', 'generated_samples.json', 'per_image.csv',
                'summary.csv', 'figure.png', 'figure.pdf', 'completion.json')


def write_json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def summarize_records(records, source_ids, recipes):
    """Validate the complete paired grid before computing macro mean/sample SD."""
    ids = set(source_ids)
    if len(ids) != len(source_ids) or not ids:
        raise ValueError('Invalid source cohort')
    candidates = {r['candidate']: r for r in recipes}
    if len(candidates) != len(recipes):
        raise ValueError('Duplicate recipes')
    cells = {}
    for row in records:
        group, level, candidate, sid = row['group'], int(row['level']), row['candidate'], row['source_id']
        if (group not in GROUPS or level not in GROUPS[group] or
                candidate not in candidates or sid not in ids or row['split'] != 'test'):
            raise ValueError('Unexpected observation')
        cell = cells.setdefault((group, level, candidate), {})
        if sid in cell:
            raise ValueError('Duplicate source in cell')
        value = float(row['f1'])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid F1')
        cell[sid] = value
    expected = {(g, lv, c) for g, levels in GROUPS.items() for lv in levels for c in candidates}
    if set(cells) != expected or any(set(cell) != ids for cell in cells.values()):
        raise ValueError('Incomplete paired grid; no dropping failed samples')
    result = []
    for group, levels in GROUPS.items():
        for level in levels:
            for candidate, recipe in candidates.items():
                values = [cells[group, level, candidate][sid] for sid in sorted(ids)]
                result.append(dict(group=group, level=level, candidate=candidate,
                                   method=recipe['method'], branch=recipe['branch'],
                                   n_sources=len(values), mean_f1=statistics.mean(values),
                                   std_f1=statistics.stdev(values) if len(values)>1 else 0.,
                                   min_f1=min(values), max_f1=max(values)))
    # Level zero is shared clean input, never independent observations by group.
    for sid in ids:
        for candidate in candidates:
            if len({cells[g, 0, candidate][sid] for g in GROUPS}) != 1:
                raise ValueError('Level-zero scores differ across groups')
    return result


def write_csv_new(path, records):
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def plot_curves(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(('otsu','niblack','sauvola','wolf','gaussian'), plt.get_cmap('tab10').colors[:5]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, group in zip(axes.flat, GROUPS):
        for method, color in colors.items():
            for branch in ('none', 't1'):
                rows = sorted((r for r in summary if r['group']==group and r['method']==method
                               and r['branch']==branch), key=lambda r:r['level'])
                ax.plot([r['level'] for r in rows], [r['mean_f1'] for r in rows],
                        color=color, linestyle='-' if branch=='none' else '--', marker='o',
                        markersize=3, label=f'{method} / {branch}')
        ax.set_title(group + (' (radius + opacity)' if group=='stamp' else ''))
        ax.set_xticks(list(GROUPS[group])); ax.set_ylim(0,1.02)
        ax.set_xlabel('Degradation level (0 = shared clean)'); ax.set_ylabel('Macro F1')
        ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center',frameon=False)
    axes.flat[-1].text(.05,.05,'none: no shadow correction\nt1: Closing81 (frozen)\n30 paired sources per point\nAuto-Otsu reference; not OCR\nSD in summary.csv; no confidence bands',fontsize=9)
    fig.suptitle('T3: frozen T2 recipes on controlled variants of dataset 6 test sources')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(Path(output)/'figure.png',dpi=180)
    fig.savefig(Path(output)/'figure.pdf')
    plt.close(fig)


def formal_benchmark(root, sources, frozen, contract, output):
    """Lock protocol on disk BEFORE any validation or test pixel is decoded."""
    output = Path(output)
    if any((output/name).exists() for name in FORMAL_FILES):
        raise FileExistsError('Formal artifact exists; choose a fresh output directory')
    test = sorted((s for s in sources if s['split']=='test'),key=lambda s:s['source_id'])
    val = sorted((s for s in sources if s['split']=='val'),key=lambda s:s['source_id'])
    if len(test)!=30 or len(val)!=10 or len({s['source_id'] for s in test+val})!=40:
        raise ValueError('Expected exact 30 test / 10 val sources')
    recipes = frozen['t2_selection']['frozen_recipes']
    shadow_recipe = {k:frozen['t1_selection']['winner'][k] for k in ('method','kernel')}
    planned = plan_rows(test)
    by_id = {s['source_id']:s for s in test}
    for row in planned:
        row['provenance'] = by_id[row['source_id']]
        row['seeds'] = {k:seed_for(row['source_id'],row['group'],k)
                        for k in ('geometry','texture','color')}
    output.mkdir(parents=True,exist_ok=True)
    write_json_new(output/'samples.json',planned)
    config = dict(contract, phase='formal_test', sources=test,
                  samples_sha256=file_sha(output/'samples.json'),
                  groups={k:list(v) for k,v in GROUPS.items()},
                  seed=42, frozen_recipes=recipes, shadow_recipe=shadow_recipe,
                  expected_rows=6600, expected_curve_slots=660,
                  expected_unique_inputs=540, complete=False,
                  aggregation='macro per-source F1; sample SD ddof=1; no CI',
                  val_illustrations=dict(source=val[0], levels='maximum per group',
                                        candidates=['t1__otsu__base','none__sauvola__k0.3_window31']),
                  no_parameter_selection=True)
    write_json_new(output/'config.json',config)
    config_hash=file_sha(output/'config.json')
    # Val illustration selection is frozen above, not chosen after test scores.
    illustration_dir=output/'val_examples'
    illustration_dir.mkdir(exist_ok=False)
    val_source=val[0]
    color,binary=read_reference(root,val_source)
    label=json.loads(checked_asset(root,val_source['rel_gt_path']).read_text(encoding='utf-8'))
    ctx=stamp_context(color,val_source['source_id'],label.get('medicine_region_clean'))
    cv2.imwrite(str(illustration_dir/'clean.png'),color)
    val_evidence=[]
    for group, levels in GROUPS.items():
        image,meta=generate(color,val_source['source_id'],group,levels[-1],ctx)
        cv2.imwrite(str(illustration_dir/f'{group}_degraded.png'),image)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        for recipe in recipes:
            if recipe['candidate'] in config['val_illustrations']['candidates']:
                pred=predict(gray,recipe,shadow_recipe)
                cv2.imwrite(str(illustration_dir/f'{group}_{recipe["candidate"]}.png'),255-pred)
        val_evidence.append(dict(source_id=val_source['source_id'],split='val',
                                 bgr_sha256=array_sha(image),**meta))
    write_json_new(illustration_dir/'provenance.json',val_evidence)
    records, generated=[],[]
    csv_path=output/'per_image.csv'
    for index,source in enumerate(test):
        color,binary=read_reference(root,source)
        original_hash=array_sha(color)
        reference=BinaryReference(binary)
        label=json.loads(checked_asset(root,source['rel_gt_path']).read_text(encoding='utf-8'))
        ctx=stamp_context(color,source['source_id'],label.get('medicine_region_clean'))
        clean_cache={}
        source_records=[]
        for group,levels in GROUPS.items():
            for level in levels:
                image,meta=generate(color,source['source_id'],group,level,ctx)
                if array_sha(color)!=original_hash or (level==0 and not np.array_equal(image,color)):
                    raise ValueError('Input mutated or invalid level zero')
                gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                image_hash=array_sha(gray)
                generated.append(dict(source_id=source['source_id'],split='test',
                                      bgr_sha256=array_sha(image),gray_sha256=image_hash,
                                      reference_sha256=array_sha(binary),**meta))
                if level!=0 or not clean_cache:
                    branch_images={'none':gray, 't1':t1.correct(gray,**shadow_recipe)}
                for recipe in recipes:
                    key=recipe['candidate']
                    if level==0 and key in clean_cache:
                        values=clean_cache[key]
                    else:
                        predicted=METHODS[recipe['method']](branch_images[recipe['branch']],**recipe['params'])
                        _binary(predicted)
                        if predicted.shape!=binary.shape:
                            raise ValueError('Prediction frame mismatch')
                        score=reference.score(predicted)
                        values=dict(f1=score['f1'],precision=score['precision'],recall=score['recall'],
                                    predicted_sha256=array_sha(predicted))
                        if level==0: clean_cache[key]=values
                    source_records.append(dict(source_id=source['source_id'],split='test',group=group,
                                               level=level,candidate=key,method=recipe['method'],
                                               branch=recipe['branch'],params_json=recipe['params_json'],
                                               gray_sha256=image_hash,**values))
        records.extend(source_records)
        with csv_path.open('x' if index==0 else 'a',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(source_records[0]))
            if index==0: writer.writeheader()
            writer.writerows(source_records)
        print(f'formal test: {index+1}/30 {source["source_id"]} ({len(records)} rows)',flush=True)
    summary=summarize_records(records,[s['source_id'] for s in test],recipes)
    write_json_new(output/'generated_samples.json',generated)
    write_csv_new(output/'summary.csv',summary)
    plot_curves(summary,output)
    if len(records)!=6600 or len(generated)!=660 or file_sha(output/'config.json')!=config_hash:
        raise ValueError('Formal run incomplete or config changed')
    write_json_new(output/'completion.json',dict(complete=True,rows=len(records),curve_slots=len(generated),
                                               summary_cells=len(summary),n_sources=30,
                                               config_sha256=config_hash,
                                               artifact_sha256={n:file_sha(output/n) for n in FORMAL_FILES
                                                                if n!='completion.json'},
                                               val_scored=False,parameters_changed=False))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check-inputs-only', action='store_true')
    modes.add_argument('--train-sanity', action='store_true')
    modes.add_argument('--formal-test', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=REPO_DIR / 'data/5_synthetic')
    parser.add_argument('--t1-dir', type=Path, default=REPO_DIR / 'results/t1_v2')
    parser.add_argument('--t2-dir', type=Path, default=REPO_DIR / 'results/t2_v2')
    parser.add_argument('--compatibility-evidence', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_DIR / 'results/t3/train_sanity')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.train_sanity and args.output_dir.exists():
        raise FileExistsError('Output already exists')
    if args.formal_test and any((args.output_dir/name).exists() for name in FORMAL_FILES):
        raise FileExistsError('Formal artifact exists')
    paths = {f'{run}_{kind}': getattr(args, f'{run}_dir') / f'{kind}.json'
             for run in ('t1', 't2') for kind in ('config', 'selection')}
    sources, frozen, report = preflight(args.data_dir, paths, args.compatibility_evidence)
    if args.check_inputs_only:
        print(json.dumps(report, indent=2))
        return
    if args.formal_test:
        formal_benchmark(args.data_dir,sources,frozen,report,args.output_dir)
    else:
        train_sanity(args.data_dir, sources, frozen, report, args.output_dir)


if __name__ == '__main__':
    main()
