"""Bộ 5."""
import numpy as np

from common.testing import todo
from synthetic import degrade


@todo
def test_perspective_returns_homography(page):
    out, gt = degrade.perspective(page, 0.10, np.random.default_rng(0))
    assert np.asarray(gt["H"]).shape == (3, 3)
