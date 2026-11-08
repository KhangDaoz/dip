"""Known-answer checks for the read-only acceptance comparison and selection."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import t4_acceptance as t


@pytest.mark.parametrize('value,saved',[(None,''),(True,'True'),(2,'2'),(.5,'.5'),('ref','ref')])
def test_known_equal_scores(value,saved):
    t.assert_equal_scores({'x':value},{'x':saved})


@pytest.mark.parametrize('value,saved',[(None,'0'),(True,'False'),(.5,'.6'),('ref','other')])
def test_changed_score_rejected(value,saved):
    with pytest.raises(ValueError):t.assert_equal_scores({'x':value},{'x':saved})


def test_replay_selection_is_order_independent_metadata_only():
    rows=[dict(source_id=s,sample_id=s,crop_id=s+'box',kind='stamp',pen_color='none',level=l) for s,l in [('b',1),('a',1),('c',2)]]
    assert t.metadata_plan({'reviewed_cohort':rows})==['abox','cbox']
    assert t.metadata_plan({'reviewed_cohort':rows[::-1]})==['abox','cbox']
