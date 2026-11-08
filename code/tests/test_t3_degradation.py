"""Step-1 tests: metadata fixtures and real generator probes are distinct."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import t3_degradation as t3
from synthetic.degrade import blur, shadow, stamp


def metadata():
    rows = []
    for split, count in t3.SPLITS.items():
        for i in range(count):
            rows.append(dict(source_id=f'{split}_{i}', sample_id=f'{split}_{i}_v00',
                             split=split, variant_index=0, width=100, height=120,
                             rel_clean_path='a', rel_clean_binary_path='b', rel_gt_path='c'))
    return dict(version='v2', n_sources=100, samples=rows, split_counts=t3.SPLITS.copy())


def test_metadata_plan_counts_and_pairing():
    sources = t3.sources_from_manifest(metadata())
    rows = t3.plan_rows(sources)
    assert Counter(r['split'] for r in rows) == {'train': 1320, 'val': 220, 'test': 660}
    assert len({r['input_key'] for r in rows}) == 1800
    for s in sources:
        sub = [r for r in rows if r['source_id'] == s['source_id']]
        for group, levels in t3.GROUPS.items():
            group_rows = [r for r in sub if r['group'] == group]
            assert [r['level'] for r in group_rows] == list(levels)
            assert len({r['generation_seed'] for r in group_rows}) == 1
        assert len({r['input_key'] for r in sub if r['level'] == 0}) == 1


@pytest.mark.parametrize('problem', ['duplicate', 'cross_split', 'missing_base', 'bad_split', 'dimensions', 'counts'])
def test_metadata_rejects_invalid(problem):
    m = metadata()
    if problem == 'duplicate': m['samples'].append(copy.deepcopy(m['samples'][0]))
    if problem == 'cross_split':
        r = dict(m['samples'][0], sample_id='extra', variant_index=1, split='test')
        m['samples'].append(r)
    if problem == 'missing_base': m['samples'][0]['variant_index'] = 1
    if problem == 'bad_split': m['samples'][0]['split'] = 'test'
    if problem == 'dimensions': m['samples'][0]['width'] = 0
    if problem == 'counts': m['split_counts']['train'] = 59
    with pytest.raises(ValueError): t3.sources_from_manifest(m)


def test_seed_stable_and_separated():
    assert t3.seed_for('x', 'stamp') == t3.seed_for('x', 'stamp')
    assert len({t3.seed_for(s, g, stream) for s in ('a', 'b')
                for g in t3.GROUPS for stream in ('geometry', 'texture')}) == 20
    with pytest.raises(ValueError): t3.seed_for('x', 'unknown')


def test_lock_rejects_modified_file(tmp_path):
    paths = {k: tmp_path / k for k in t3.LOCKS}
    for path in paths.values(): path.write_text('{}')
    with pytest.raises(ValueError, match='Frozen artifact changed'): t3.load_frozen(paths)


def test_assets_hash_bytes_and_label_provenance(tmp_path):
    source = t3.sources_from_manifest(metadata())[0]
    (tmp_path / 'a').write_bytes(b'not an image')
    (tmp_path / 'b').write_bytes(b'also not an image')
    label = {k: source[k] for k in ('sample_id', 'source_id', 'split')}
    label['image_size'] = [100, 120]
    (tmp_path / 'c').write_text(json.dumps(label))
    # Intentionally succeeds on undecodable bytes: this proves metadata-only scope.
    assert len(t3.audit_assets(tmp_path, [source])[0]['hashes']) == 3
    label['source_id'] = 'wrong'
    (tmp_path / 'c').write_text(json.dumps(label))
    with pytest.raises(ValueError): t3.audit_assets(tmp_path, [source])


@pytest.mark.parametrize('relative', ['../escape', '/missing_absolute'])
def test_assets_reject_outside_or_missing(tmp_path, relative):
    source = dict(t3.sources_from_manifest(metadata())[0], rel_clean_path=relative)
    with pytest.raises(ValueError): t3.audit_assets(tmp_path, [source])


def test_reference_checksum_rejects_missing_and_changed(tmp_path):
    source = {'source_id': 'a', 'rel_clean_path': 'clean.png'}
    evidence = [{'source_id': 'a', 'hashes': {'rel_clean_path': 'abc'}}]
    (tmp_path / 'checksums_v1.sha256').write_text('abc  clean.png\n')
    (tmp_path / 'checksums_v2.sha256').write_text('')
    assert t3.verify_reference_checksums(tmp_path, [source], evidence) == 1
    evidence[0]['hashes']['rel_clean_path'] = 'changed'
    with pytest.raises(ValueError): t3.verify_reference_checksums(tmp_path, [source], evidence)
    (tmp_path / 'checksums_v1.sha256').write_text('')
    with pytest.raises(ValueError): t3.verify_reference_checksums(tmp_path, [source], evidence)


@pytest.mark.parametrize('kind', ['planar', 'silhouette'])
def test_real_shadow_generator_fixed_geometry(kind):
    image = np.full((180, 220, 3), 255, np.uint8)
    seed = t3.seed_for('fixture', 'shadow_' + kind)
    outputs = [shadow(image, level=lv, shadow_type=kind, rng=np.random.default_rng(seed)) for lv in (1, 2, 3)]
    assert len({g['coverage_ratio'] for _, g in outputs}) == 1
    assert [g['darkness'] for _, g in outputs] == [0.2, 0.45, 0.7]
    for out, g in outputs:
        assert out.shape == image.shape and out.dtype == np.uint8
        assert g['shadow_type'] == kind
    assert np.all(outputs[2][0] <= outputs[1][0])
    assert np.all(outputs[1][0] <= outputs[0][0])


@pytest.mark.parametrize('kind', ['gaussian', 'motion'])
def test_real_blur_generator_fixed_direction(kind):
    image = np.full((100, 100, 3), 255, np.uint8)
    image[40:60, 49:51] = 0
    original = image.copy()
    seed = t3.seed_for('fixture', 'blur_' + kind)
    outputs = [blur(image, level=lv, kind=kind, rng=np.random.default_rng(seed)) for lv in (1, 2, 3, 4)]
    assert len({g['angle_deg'] for _, g in outputs}) == 1
    for lv, (out, g) in enumerate(outputs, 1):
        assert g['radius'] == lv and g['kind'] == kind
        assert out.shape == image.shape and out.dtype == np.uint8
    assert np.array_equal(image, original)


def test_real_stamp_same_seed_does_not_freeze_nuisances():
    image = np.full((240, 240, 3), 255, np.uint8)
    outputs = [stamp(image, level=lv, center=(120, 120), rng=np.random.default_rng(42)) for lv in (1, 2, 3)]
    assert len({tuple(g['center']) for _, g in outputs}) == 1
    assert len({g['rotation_deg'] for _, g in outputs}) > 1
    assert [g['opacity'] for _, g in outputs] == [0.55, 0.75, 0.92]
    # Evidence of the blocker, not a claim that the future T3 wrapper is ready.


@pytest.fixture
def page():
    image = np.full((180, 220, 3), 245, np.uint8)
    image[40:150:12, 35:180] = 30
    return image


@pytest.mark.parametrize('group', list(t3.GROUPS))
def test_wrapper_zero_and_determinism_and_no_mutation(page, group):
    original = page.copy()
    zero, meta = t3.generate(page, 's', group, 0)
    assert np.array_equal(zero, page) and zero is not page and meta['applied'] == 'none'
    for level in t3.GROUPS[group]:
        a, ma = t3.generate(page, 's', group, level)
        b, mb = t3.generate(page, 's', group, level)
        assert np.array_equal(a, b) and ma == mb
        assert a.shape == page.shape and a.dtype == np.uint8
        assert ma['H_clean_to_derived'] == [[1,0,0],[0,1,0],[0,0,1]]
    assert np.array_equal(page, original)


def test_controlled_stamp_preserves_nuisances(page):
    ctx = t3.stamp_context(page, 's', [35,40,140,100])
    results = [t3.generate(page, 's', 'stamp', lv, ctx)[1]['parameters'] for lv in (1,2,3)]
    for key in ('center','rotation_deg','color_bgr','radius_jitter','canonical_alpha_sha256','seeds'):
        assert results[0][key] == results[1][key] == results[2][key]
    assert [r['opacity'] for r in results] == [0.55,0.75,0.92]
    assert results[0]['radius'] < results[1]['radius'] < results[2]['radius']
    with pytest.raises(ValueError, match='another source'):
        t3.generate(page, 'different_source', 'stamp', 1, ctx)


@pytest.mark.parametrize('group,level', [('unknown',1),('stamp',4),('blur_motion',True),('shadow_planar',-1)])
def test_wrapper_rejects_invalid(page, group, level):
    with pytest.raises(ValueError): t3.generate(page, 's', group, level)


def test_zero_does_not_call_generator(page, monkeypatch):
    def fail(*a, **kw): raise AssertionError('Generator called at zero')
    monkeypatch.setattr(t3.degrade, 'shadow', fail)
    monkeypatch.setattr(t3.degrade, 'blur', fail)
    monkeypatch.setattr(t3, 'stamp_context', fail)
    for group in t3.GROUPS:
        assert np.array_equal(t3.generate(page, 's', group, 0)[0], page)


@pytest.mark.parametrize('method', ['otsu','niblack','sauvola','wolf','gaussian'])
@pytest.mark.parametrize('branch', ['none','t1'])
def test_predict_binary_conventions(page, method, branch):
    gray = t3.cv2.cvtColor(page, t3.cv2.COLOR_BGR2GRAY)
    params = dict(window=15) if method in ('niblack','sauvola','wolf') else {}
    result = t3.predict(gray, dict(method=method, branch=branch, params=params),
                        dict(method='closing',kernel=81))
    assert result.dtype == np.uint8 and result.shape == gray.shape
    assert set(np.unique(result)) <= {0,255}
    assert np.any(result == 255) and np.any(result == 0)


def test_preflight_cli_does_not_write_or_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(t3, 'preflight', lambda *a: ([],{},dict(pixel_checks_run=False)))
    def fail(*a): raise AssertionError('Runner called')
    monkeypatch.setattr(t3, 'train_sanity', fail)
    out = tmp_path/'must_not_exist'
    t3.main(['--check-inputs-only','--compatibility-evidence','archive.zip','--output-dir',str(out)])
    assert not out.exists() and 'false' in capsys.readouterr().out


def test_sanity_rejects_existing_before_preflight(tmp_path, monkeypatch):
    def fail(*a): raise AssertionError('Preflight called')
    monkeypatch.setattr(t3, 'preflight', fail)
    with pytest.raises(FileExistsError):
        t3.main(['--train-sanity','--compatibility-evidence','archive.zip','--output-dir',str(tmp_path)])


def test_read_reference_checks_actual_pixels(tmp_path, page, monkeypatch):
    from cleaning.binarize import otsu
    gray = t3.cv2.cvtColor(page, t3.cv2.COLOR_BGR2GRAY)
    t3.cv2.imwrite(str(tmp_path/'clean.png'),page)
    t3.cv2.imwrite(str(tmp_path/'binary.png'),otsu(gray))
    s=dict(width=220,height=180,rel_clean_path='clean.png',rel_clean_binary_path='binary.png')
    color,binary=t3.read_reference(tmp_path,s)
    assert np.array_equal(color,page)
    t3.cv2.imwrite(str(tmp_path/'binary.png'),255-binary)
    with pytest.raises(ValueError,match='Otsu'): t3.read_reference(tmp_path,s)


def test_preflight_rejects_manifest_before_loading_pixels(tmp_path):
    (tmp_path/'manifest_v2.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest'): t3.preflight(tmp_path,{},None)


def score_fixture():
    recipes=[dict(candidate='a',method='otsu',branch='none')]
    rows=[dict(source_id=s,split='test',group=g,level=lv,candidate='a',f1=score)
          for s,score in [('large',0.2),('small',0.8)]
          for g,levels in t3.GROUPS.items() for lv in levels]
    return recipes,rows


def test_summary_macro_mean_sd_and_complete_grid():
    recipes,rows=score_fixture()
    summary=t3.summarize_records(rows,['large','small'],recipes)
    assert len(summary)==22
    assert all(r['n_sources']==2 and r['mean_f1']==0.5 for r in summary)
    assert all(r['std_f1']==pytest.approx(0.4242640687) for r in summary)


@pytest.mark.parametrize('error',['missing','duplicate','nonfinite','wrong_split','zero_changed','unknown_level'])
def test_summary_fails_closed(error):
    recipes,rows=score_fixture()
    if error=='missing': rows.pop()
    if error=='duplicate': rows.append(dict(rows[0]))
    if error=='nonfinite': rows[0]['f1']=float('nan')
    if error=='wrong_split': rows[0]['split']='val'
    if error=='zero_changed': rows[0]['f1']=0.3
    if error=='unknown_level': rows[0]['level']=99
    with pytest.raises(ValueError): t3.summarize_records(rows,['large','small'],recipes)


def test_formal_freezes_all_plans_before_any_pixel(tmp_path,monkeypatch):
    sources=t3.sources_from_manifest(metadata())
    output=tmp_path/'formal'
    frozen={'t1_selection':{'winner':{'method':'closing','kernel':81}},
            't2_selection':{'frozen_recipes':[]}}
    def pixel_probe(*args):
        config=json.loads((output/'config.json').read_text())
        planned=json.loads((output/'samples.json').read_text())
        assert len(planned)==660 and {p['split'] for p in planned}=={'test'}
        assert config['samples_sha256']==t3.file_sha(output/'samples.json')
        assert config['expected_rows']==6600 and not config['complete']
        assert config['val_illustrations']['source']['source_id']=='val_0'
        raise RuntimeError('Stopped before decoding first val pixel')
    monkeypatch.setattr(t3,'read_reference',pixel_probe)
    with pytest.raises(RuntimeError,match='Stopped before decoding'):
        t3.formal_benchmark(tmp_path,sources,frozen,{},output)
    assert not (output/'completion.json').exists()
    with pytest.raises(FileExistsError):
        t3.formal_benchmark(tmp_path,sources,frozen,{},output)


def test_formal_refuses_existing_artifacts_before_preflight(tmp_path,monkeypatch):
    (tmp_path/'summary.csv').write_text('history')
    def fail(*args): raise AssertionError('preflight called')
    monkeypatch.setattr(t3,'preflight',fail)
    with pytest.raises(FileExistsError):
        t3.main(['--formal-test','--compatibility-evidence','zip','--output-dir',str(tmp_path)])
    assert (tmp_path/'summary.csv').read_text()=='history'


def acceptance_module():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import verify_t3
    return verify_t3


def test_acceptance_rejects_modified_config(tmp_path):
    verifier=acceptance_module()
    (tmp_path/'completion.json').write_text(json.dumps(dict(complete=True,rows=6600,config_sha256='original',
        artifact_sha256={n:'hash' for n in t3.FORMAL_FILES if n!='completion.json'})))
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(AssertionError): verifier.verify(tmp_path)


def test_acceptance_report_refuses_overwrite(tmp_path,monkeypatch):
    verifier=acceptance_module()
    report=tmp_path/'report.json'; report.write_text('existing evidence')
    monkeypatch.setattr(sys,'argv',['verify_t3','--compatibility-evidence','archive.zip','--report',str(report)])
    with pytest.raises(FileExistsError): verifier.main()
    assert report.read_text()=='existing evidence'


def test_acceptance_checks_baseline_even_without_replay(tmp_path,monkeypatch):
    verifier=acceptance_module()
    monkeypatch.setattr(verifier,'verify',lambda root:{'status':'PASS'})
    (tmp_path/'config.json').write_text(json.dumps({'manifest_sha256':'old'}))
    monkeypatch.setattr(t3,'preflight',lambda *args:([],{},dict(manifest_sha256='changed')))
    with pytest.raises(AssertionError):
        verifier.accept(tmp_path,tmp_path,None,tmp_path,tmp_path,replay=False)


def test_acceptance_disallows_optimized_asserts():
    import subprocess
    path=Path(__file__).resolve().parents[1]/'scripts/verify_t3.py'
    result=subprocess.run([sys.executable,'-O',str(path),'--help'],capture_output=True,text=True)
    assert result.returncode!=0 and 'Run without -O' in result.stderr


@pytest.mark.parametrize('missing',['summary.csv','figure.png','generated_samples.json'])
def test_acceptance_rejects_omitted_artifact_checksum(missing):
    verifier=acceptance_module()
    hashes={n:'hash' for n in t3.FORMAL_FILES if n!='completion.json'}
    hashes.pop(missing)
    with pytest.raises(ValueError,match='Incomplete formal'):
        verifier.check_artifact_schema({'artifact_sha256':hashes})


def test_acceptance_rejects_omitted_figure_checksum():
    verifier=acceptance_module()
    hashes={n:'hash' for n in t3.FORMAL_FILES if n!='completion.json'}
    with pytest.raises(ValueError,match='Incomplete figure'):
        verifier.check_artifact_schema({'artifact_sha256':hashes},
                                       dict(initial_figures={},current_figures={}))


def test_acceptance_rejects_mislabeled_summary():
    verifier=acceptance_module()
    recipe=dict(candidate='otsu_none',method='otsu',branch='none')
    verifier.check_summary_labels([dict(recipe)],[recipe])
    wrong=dict(recipe,method='sauvola')
    with pytest.raises(ValueError,match='labels disagree'):
        verifier.check_summary_labels([wrong],[recipe])
