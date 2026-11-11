import numpy as np
import pytest

from conftest import make_page, rotate_ccw
from enhance.pipeline import PRESETS, STEPS, enhance


def test_enhance_all_steps(page):
    out, info = enhance(page, "full")
    assert out.dtype == np.uint8 and out.ndim == 3
    assert set(info["timing"]) == set(STEPS)


def test_enhance_none_is_identity(page):
    out, info = enhance(page, "none")
    assert out is page and info["steps"] == []


def test_enhance_rejects_unknown_step(page):
    with pytest.raises(ValueError):
        enhance(page, ("deskew", "sharpen"))


@pytest.mark.parametrize("angle", [-5.0, -2.0, 3.0, 7.0])
def test_deskew_recovers_angle(angle):
    from enhance.deskew import estimate_skew
    assert abs(estimate_skew(rotate_ccw(make_page(), angle)) - angle) <= 0.5


def test_deskew_straightens(page):
    from enhance.deskew import deskew, estimate_skew
    fixed, ang = deskew(rotate_ccw(page, 4.0))
    assert abs(ang - 4.0) <= 0.5 and abs(estimate_skew(fixed)) <= 0.5


def test_shadow_flattens_background(page_gray):
    from enhance.shadow import remove_shadows
    h, w = page_gray.shape
    ramp = np.linspace(0.5, 1.0, w)[None, :]
    shaded = (page_gray * ramp).astype(np.uint8)
    flat = remove_shadows(shaded)
    bg = flat[:40, :]
    assert bg[:, :50].mean() > 200 and abs(bg[:, :50].mean() - bg[:, -50:].mean()) < 15


def test_clahe_and_denoise_keep_shape(page):
    from enhance.clahe import clahe
    from enhance.denoise import denoise
    for f in (clahe, denoise):
        out = f(page)
        assert out.shape == page.shape and out.dtype == np.uint8


def test_presets_valid():
    assert all(set(v) <= set(STEPS) for v in PRESETS.values())
