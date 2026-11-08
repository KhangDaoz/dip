"""T3 acceptance: strict artifact audit and optional fixed-cohort pixel replay."""
import argparse
if not __debug__:
    raise RuntimeError("Run without -O: acceptance checks must remain enabled")
import csv
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'code/experiments'))
import t3_degradation as t3


def check_artifact_schema(completion, render=None):
    """A missing checksum entry is missing evidence, not a successful check."""
    expected=set(t3.FORMAL_FILES)-{'completion.json'}
    if set(completion.get('artifact_sha256',{})) != expected:
        raise ValueError('Incomplete formal artifact checksum manifest')
    if render is not None:
        if (set(render['initial_figures']) != {'figure_initial.png','figure_initial.pdf'}
                or set(render['current_figures']) != {'figure.png','figure.pdf'}):
            raise ValueError('Incomplete figure checksum manifest')


def check_summary_labels(table, recipes):
    by_id={r['candidate']:r for r in recipes}
    for row in table:
        recipe=by_id.get(row['candidate'])
        if recipe is None or any(row[k]!=recipe[k] for k in ('method','branch')):
            raise ValueError('Summary method/branch labels disagree with frozen recipe')


def verify(output):
    output=Path(output)
    load=lambda name: json.loads((output/name).read_text(encoding='utf-8'))
    completion=load('completion.json'); config=load('config.json')
    check_artifact_schema(completion,load('figure_render_update.json')
                          if (output/'figure_render_update.json').exists() else None)
    assert completion['complete'] and completion['rows']==6600
    assert t3.file_sha(output/'config.json')==completion['config_sha256']
    for name,sha in completion['artifact_sha256'].items():
        original=output/name
        if name in ('figure.png','figure.pdf') and (output/'figure_render_update.json').exists():
            original=output/('figure_initial'+Path(name).suffix)
        assert t3.file_sha(original)==sha, name
    if (output/'figure_render_update.json').exists():
        render=load('figure_render_update.json')
        assert render['summary_sha256']==t3.file_sha(output/'summary.csv')
        assert render['frozen_config_sha256']==t3.file_sha(output/'config.json')
        assert render['renderer_sha256']==t3.file_sha(REPO/'code/experiments/t3_plot.py')
        for name,sha in {**render['initial_figures'],**render['current_figures']}.items():
            assert t3.file_sha(output/name)==sha
    assert t3.file_sha(output/'samples.json')==config['samples_sha256']
    for name,sha in config['code_sha256'].items():
        assert t3.file_sha(REPO/name)==sha, name
    plans=load('samples.json'); generated=load('generated_samples.json')
    rows=list(csv.DictReader((output/'per_image.csv').open(encoding='utf-8')))
    table=list(csv.DictReader((output/'summary.csv').open(encoding='utf-8')))
    ids=[s['source_id'] for s in config['sources']]
    assert len(ids)==len(set(ids))==30 and all(s['split']=='test' for s in config['sources'])
    assert len(plans)==len(generated)==660 and len(rows)==6600 and len(table)==220
    assert len({(r['group'], r['level'], r['candidate']) for r in table})==220
    assert Counter(r['source_id'] for r in rows)==Counter({sid:220 for sid in ids})
    key=lambda r:(r['source_id'],r['group'],int(r['level']))
    actual={key(r):r for r in generated}
    assert len(actual)==660 and set(actual)=={key(r) for r in plans}
    for plan in plans:
        r=actual[key(plan)]
        assert r['split']=='test' and r['generation_seed']==plan['generation_seed']
        assert r['H_clean_to_derived']==plan['H_clean_to_derived']==[[1,0,0],[0,1,0],[0,0,1]]
    for sid in ids:
        source=[r for r in generated if r['source_id']==sid]
        assert len({r['bgr_sha256'] for r in source if r['level']==0})==1
        assert len({r['reference_sha256'] for r in source})==1
        for group,levels in t3.GROUPS.items():
            pos=[r['parameters'] for r in source if r['group']==group and r['level']>0]
            keys=('center','rotation_deg','color_bgr','radius_jitter','canonical_alpha_sha256') if group=='stamp' else (
                ('shadow_type','coverage_ratio') if group.startswith('shadow') else ('kind','angle_deg'))
            for name in keys:
                assert all(r[name]==pos[0][name] for r in pos), (sid,group,name)
    recipes={r['candidate']:r for r in config['frozen_recipes']}
    check_summary_labels(table,config['frozen_recipes'])
    for row in rows:
        r=recipes[row['candidate']]
        assert all(row[k]==r[k] for k in ('method','branch','params_json'))
        assert row['gray_sha256']==actual[key(row)]['gray_sha256']
    # Independent numpy arithmetic, rather than trusting the summary function.
    for aggregate in table:
        matching=[r for r in rows if r['group']==aggregate['group'] and r['level']==aggregate['level']
                  and r['candidate']==aggregate['candidate']]
        assert len(matching)==len({r['source_id'] for r in matching})==30
        values=np.array([float(r['f1']) for r in matching])
        assert np.isfinite(values).all() and np.all((values>=0)&(values<=1))
        for name,value in [('mean_f1',values.mean()),('std_f1',values.std(ddof=1)),
                           ('min_f1',values.min()),('max_f1',values.max())]:
            assert np.isclose(float(aggregate[name]),value,rtol=0,atol=1e-12)
        assert int(aggregate['n_sources'])==30
    t3.summarize_records(rows,ids,config['frozen_recipes'])  # includes shared-zero checks
    return dict(status='PASS',rows=6600,curve_slots=660,summary_cells=220,
                sources=30,each_cell_has_all_sources=True,independent_summary_match_atol=1e-12,
                config_samples_code_artifacts_hashes_match=True,
                paired_nuisances_verified=True,recipes_unchanged=True,
                shared_zero_verified=True,pixel_replay_run=False,
                scope='Artifact integrity, metadata invariants and independent CSV aggregation; not full step-4 acceptance')


def accept(root, data, evidence, t1_dir, t2_dir, replay=False):
    root=Path(root)
    report=verify(root)
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    paths={f'{r}_{k}':d/f'{k}.json' for r,d in [('t1',t1_dir),('t2',t2_dir)]
           for k in ('config','selection')}
    sources,frozen,current=t3.preflight(data,paths,evidence)
    for key in ('manifest_sha256','versions','reference_evidence','checksum_lists','frozen_artifacts'):
        assert current[key]==config[key], key
    test=sorted((s for s in sources if s['split']=='test'),key=lambda s:s['source_id'])
    assert config['sources']==test
    assert config['frozen_recipes']==frozen['t2_selection']['frozen_recipes']
    assert config['shadow_recipe']=={k:frozen['t1_selection']['winner'][k] for k in ('method','kernel')}
    plans=json.loads((root/'samples.json').read_text(encoding='utf-8'))
    for old,new in zip(plans,t3.plan_rows(test),strict=True):
        assert all(old[k]==v for k,v in new.items())
    assert config['no_parameter_selection'] is True
    report.update(data_preflight_match=True,all_test_sources_match_manifest=True,
                  locked_recipe_binding_match=True,verifier_sha256=t3.file_sha(Path(__file__)))
    if replay:
        generated=json.loads((root/'generated_samples.json').read_text(encoding='utf-8'))
        by_sample={(r['source_id'],r['group'],r['level']):r for r in generated}
        rows=list(csv.DictReader((root/'per_image.csv').open(encoding='utf-8')))
        by_score={(r['source_id'],r['group'],int(r['level']),r['candidate']):r for r in rows}
        selected=[test[i] for i in (0,15,29)]
        results=[]
        for source in selected:
            color,binary=t3.read_reference(data,source)
            label=json.loads(t3.checked_asset(data,source['rel_gt_path']).read_text(encoding='utf-8'))
            ctx=t3.stamp_context(color,source['source_id'],label.get('medicine_region_clean'))
            reference=t3.BinaryReference(binary)
            n=0
            for group,levels in t3.GROUPS.items():
                for level in levels:
                    image,metadata=t3.generate(color,source['source_id'],group,level,ctx)
                    gray=t3.cv2.cvtColor(image,t3.cv2.COLOR_BGR2GRAY)
                    expected=by_sample[source['source_id'],group,level]
                    assert expected['bgr_sha256']==t3.array_sha(image)
                    assert expected['gray_sha256']==t3.array_sha(gray)
                    assert expected['reference_sha256']==t3.array_sha(binary)
                    assert all(expected[k]==v for k,v in metadata.items())
                    branches={'none':gray,'t1':t3.t1.correct(gray,**config['shadow_recipe'])}
                    for recipe in config['frozen_recipes']:
                        pred=t3.METHODS[recipe['method']](branches[recipe['branch']],**recipe['params'])
                        score=reference.score(pred)
                        saved=by_score[source['source_id'],group,level,recipe['candidate']]
                        assert saved['predicted_sha256']==t3.array_sha(pred)
                        for metric in ('f1','precision','recall'):
                            assert np.isclose(float(saved[metric]),score[metric],rtol=0,atol=1e-12)
                        n+=1
            results.append(dict(source_id=source['source_id'],score_rows=n,curve_slots=22,
                                generated_reference_prediction_hashes_match=True,score_match_atol=1e-12))
            print(f'replay: {source["source_id"]} {n} scores PASS',flush=True)
        report.update(pixel_replay_run=True,replay_selection='sorted test ranks 0,15,29; all levels/groups/recipes',
                      replay=results,replayed_rows=660,replayed_curve_slots=66,
                      scope='Full metadata/CSV integrity plus 660-row pixel replay; not full 6600-row rerun')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=REPO/'results/t3')
    parser.add_argument('--data-dir',type=Path,default=REPO/'data/5_synthetic')
    parser.add_argument('--t1-dir',type=Path,default=REPO/'results/t1_v2')
    parser.add_argument('--t2-dir',type=Path,default=REPO/'results/t2_v2')
    parser.add_argument('--compatibility-evidence',type=Path,required=True)
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError('Report exists; choose a new filename')
    report=accept(args.results_dir,args.data_dir,args.compatibility_evidence,
                  args.t1_dir,args.t2_dir,args.replay)
    if args.report:
        with args.report.open('x',encoding='utf-8') as f:
            json.dump(report,f,indent=2)
    else:
        print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()


