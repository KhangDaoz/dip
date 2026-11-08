"""Edge cases for the implemented cleaning subset."""
import numpy as np
import pytest

from cleaning.background import estimate_background, remove_shadow
from cleaning.binarize import otsu, sauvola


@pytest.mark.parametrize("method", [otsu, sauvola])
def test_white_page_has_no_ink(method):
    assert not method(np.full((40, 60), 255, np.uint8)).any()


def test_shadow_correction_preserves_dark_strokes():
    light = np.tile(np.linspace(80, 230, 200), (80, 1)).astype(np.uint8)
    page = light.copy()
    page[30:33, 20:180] = (light[30:33, 20:180] * .15).astype(np.uint8)
    corrected = remove_shadow(page, estimate_background(page, kernel=21))
    assert corrected[:20].std() < 2
    assert corrected[30:33, 20:180].max() < 60


def test_black_page_does_not_divide_by_zero():
    black = np.zeros((40, 40), np.uint8)
    assert np.array_equal(remove_shadow(black, black), black)


@pytest.mark.parametrize("function", [otsu, sauvola, estimate_background])
def test_reject_color_input(function):
    with pytest.raises(ValueError):
        function(np.zeros((40, 40, 3), np.uint8))


def test_reject_invalid_window_and_mismatched_background():
    gray = np.zeros((40, 40), np.uint8)
    with pytest.raises(ValueError):
        sauvola(gray, window=20)
    with pytest.raises(ValueError):
        remove_shadow(gray, gray[:20])
