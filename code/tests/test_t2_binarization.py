"""T2 must freeze complete, paired validation comparisons with no test tuning."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    't2_binarization', Path(__file__).resolve().parents[1]/'experiments/t2_binarization.py')
t2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t2)


def validation_rows():
    rows = []
    for source in ('a', 'b'):
        for recipe in t2.defaults():
            rows.append({k:v for k,v in recipe.items() if k != 'params'} | dict(
                source_id=source, split='val', phase='validation', f1=.9, pfm_skeleton=.95,
                drd=1., precision=.9, recall=.9, pseudo_recall=1., psnr_db=20.,
                error_fraction=.01, error_pixels=100))
    frame = pd.DataFrame(rows)
    frame.loc[(frame.method=='otsu') & (frame.branch=='t1'), 'f1'] = .99
    return frame


def test_grid_covers_all_methods_branches_and_signed_niblack_parameters():
    grid = t2.grid(t2.CONFIG)
    assert len(grid) == 80 and len({r['candidate'] for r in grid}) == 80
    assert len([r for r in grid if r['method']=='otsu']) == 2
    assert all(r['params']['k'] < 0 for r in grid if r['method']=='niblack')


def test_selection_freezes_ten_recipes_based_on_validation_f1():
    ranking, recipes = t2.select_parameters(validation_rows())
    assert (ranking.iloc[0]['method'], ranking.iloc[0]['branch']) == ('otsu', 't1')
    assert len(recipes) == 10
    assert {r['candidate'] for r in recipes} == set(validation_rows().candidate)


@pytest.mark.parametrize('bad', ['test', 'train', 'val_clean_control'])
def test_selection_rejects_nonvalidation_data(bad):
    frame = validation_rows()
    frame.loc[0, 'split' if bad != 'val_clean_control' else 'phase'] = bad
    with pytest.raises(ValueError, match='validation rows only'):
        t2.select_parameters(frame)


def test_selection_rejects_missing_sources_or_undefined_drd():
    with pytest.raises(ValueError, match='same val sources'):
        t2.select_parameters(validation_rows().iloc[1:])
    frame = validation_rows(); frame.loc[0, 'drd'] = np.nan
    with pytest.raises(ValueError, match='Undefined selection metrics'):
        t2.select_parameters(frame)


def test_summary_keeps_exact_matches_without_finite_psnr_cap():
    frame = validation_rows().iloc[:2].copy()
    frame.loc[:, ['method','branch','params_json','candidate']] = ['otsu','none','{}','none__otsu__base']
    frame['error_fraction'] = [0., .01]
    frame['error_pixels'] = [0, 100]
    frame['psnr_db'] = [float('inf'), 20.]
    result = t2.summary(frame).iloc[0]
    assert result.exact_matches == 1
    assert result.mean_image_psnr_db == float('inf')
    assert result.psnr_db == pytest.approx(-10*np.log10(.005))
