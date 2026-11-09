"""② Làm sạch & tách lớp mực."""
import numpy as np

from cleaning import background, binarize, metrics
from common.testing import todo
from conftest import f_measure


@todo
def test_remove_shadow_flattens_background(page_gray):
    ramp = np.linspace(0.5, 1.0, page_gray.shape[1])[None, :]
    shaded = (page_gray * ramp).astype(np.uint8)
    flat = background.remove_shadow(shaded, background.background_closing(shaded, kernel=41))
    assert flat[:, -50:].mean() - flat[:, :50].mean() < 15


@todo
def test_otsu_matches_text(page_gray, text_mask):
    assert f_measure(binarize.otsu(page_gray), text_mask) > 0.9


@todo
def test_f_measure_perfect(text_mask):
    assert metrics.f_measure(text_mask, text_mask) == 1.0
