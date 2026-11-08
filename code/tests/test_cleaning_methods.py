"""Formula, border and edge-case checks for T1/T2 methods (not benchmarks)."""
import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from cleaning.background import estimate_background, remove_shadow
from cleaning.binarize import METHODS, gaussian_adaptive, niblack, sauvola, wolf


def reference_stats(image, window):
    # Independent small-array oracle: explicitly enumerate reflected windows,
    # rather than using the implementation's box filters or moment formula.
    radius = window // 2
    windows = sliding_window_view(np.pad(image.astype(float), radius, mode="reflect"),
                                  (window, window))
    return windows.mean(axis=(-2, -1)), windows.std(axis=(-2, -1))


@pytest.mark.parametrize("k", [-0.2, 0.0, 0.3])
def test_niblack_signed_k_and_reflected_border_match_formula(k):
    image = np.random.default_rng(51).integers(0, 256, (7, 9), dtype=np.uint8)
    mean, std = reference_stats(image, 3)
    expected = np.where(image < mean + k * std, 255, 0).astype(np.uint8)
    assert np.array_equal(niblack(image, window=3, k=k), expected)


@pytest.mark.parametrize("k", [0.0, 0.5, 1.0])
def test_wolf_uses_global_minimum_and_maximum_local_std(k):
    image = np.random.default_rng(12).integers(30, 221, (7, 9), dtype=np.uint8)
    image[0, 0] = 5
    mean, std = reference_stats(image, 5)
    threshold = (1 - k) * mean + k * float(image.min()) + k * std / std.max() * (mean - image.min())
    expected = np.where(image < threshold, 255, 0).astype(np.uint8)
    assert np.array_equal(wolf(image, window=5, k=k), expected)


@pytest.mark.parametrize("method", [niblack, wolf])
@pytest.mark.parametrize("value", [0, 1, 127, 255])
def test_constant_images_do_not_create_ink_or_divide_by_zero(method, value):
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        out = method(np.full((9, 13), value, np.uint8), window=5)
    assert not out.any()


@pytest.mark.parametrize("method", [niblack, wolf])
def test_contrast_rescaling_preserves_local_decision(method):
    image = np.random.default_rng(17).integers(1, 110, (15, 17), dtype=np.uint8)
    transformed = (image.astype(np.uint16) * 2 + 20).astype(np.uint8)
    assert np.array_equal(method(image, window=5), method(transformed, window=5))


def test_gaussian_separates_dark_stroke_and_respects_threshold_equality():
    page = np.full((15, 17), 220, np.uint8)
    page[5:10, 8] = 30
    out = gaussian_adaptive(page, block=5, c=10)
    assert np.all(out[5:10, 8] == 255)
    assert not out[page == 220].any()
    # OpenCV inverse threshold includes equality when c=0. This is intentional,
    # not a reason to silently substitute strict < or change the supplied c.
    assert np.all(gaussian_adaptive(np.full((5, 7), 128, np.uint8), block=3, c=0) == 255)
    assert not gaussian_adaptive(np.full((5, 7), 128, np.uint8), block=3, c=1).any()


@pytest.mark.parametrize("name", list(METHODS))
@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (9, 13)])
def test_all_methods_support_small_images_without_mutating_input(name, shape):
    # Non-contiguous array, and window larger than the image in the first cases.
    image = np.random.default_rng(4).integers(0, 256, (shape[0], shape[1] * 2), dtype=np.uint8)[:, ::2]
    original = image.copy()
    out = METHODS[name](image)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}
    assert np.array_equal(image, original)


@pytest.mark.parametrize("name", list(METHODS))
def test_all_methods_leave_white_paper_empty(name):
    assert not METHODS[name](np.full((17, 19), 255, np.uint8)).any()


@pytest.mark.parametrize("function", [*METHODS.values(), estimate_background])
@pytest.mark.parametrize("image", [
    np.zeros((4, 5, 3), np.uint8), np.zeros((4, 5), float),
    np.zeros((0, 5), np.uint8), [[0, 255]],
])
def test_reject_incorrect_image_type(function, image):
    with pytest.raises(ValueError, match="2D uint8"):
        function(image)


@pytest.mark.parametrize("function,argument", [
    (niblack, "window"), (sauvola, "window"), (wolf, "window"),
    (gaussian_adaptive, "block"), (estimate_background, "kernel"),
])
@pytest.mark.parametrize("invalid", [0, 1, 4, 3.0, True])
def test_reject_invalid_neighborhood(function, argument, invalid):
    with pytest.raises(ValueError, match="odd integer"):
        function(np.full((5, 7), 200, np.uint8), **{argument: invalid})


@pytest.mark.parametrize("function,argument", [
    (niblack, "k"), (sauvola, "k"), (wolf, "k"), (gaussian_adaptive, "c"),
])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), "0.2", None, True])
def test_reject_invalid_scalar(function, argument, invalid):
    with pytest.raises(ValueError, match="finite real"):
        function(np.full((5, 7), 200, np.uint8), **{argument: invalid})


def reference_neighborhood(image, window, operation):
    radius = window // 2
    windows = sliding_window_view(np.pad(image, radius, mode="edge"), (window, window))
    return operation(windows, axis=(-2, -1)).astype(np.uint8)


@pytest.mark.parametrize("kernel", [3, 5])
@pytest.mark.parametrize("method", ["closing", "closing_median"])
def test_background_closing_order_and_replicated_border(method, kernel):
    image = np.random.default_rng(35).integers(20, 256, (9, 13), dtype=np.uint8)
    original = image.copy()
    dilated = reference_neighborhood(image, kernel, np.max)
    expected = reference_neighborhood(dilated, kernel, np.min)
    if method == "closing_median":
        expected = reference_neighborhood(expected, kernel, np.median)
    assert np.array_equal(estimate_background(image, method, kernel), expected)
    assert np.array_equal(image, original)


@pytest.mark.parametrize("method", ["median", "closing", "closing_median"])
def test_background_preserves_broad_illumination_and_removes_thin_stroke(method):
    light = np.tile(np.linspace(90, 230, 101), (51, 1)).astype(np.uint8)
    page = light.copy()
    page[24:27, 10:90] = 15
    bg = estimate_background(page, method=method, kernel=11)
    corrected = remove_shadow(page, bg)
    assert bg.shape == page.shape and bg.dtype == np.uint8
    assert np.all(bg[25, 15:85] > 70)  # Background should not absorb the thin ink stroke.
    assert corrected[:15, 15:85].std() < 2
    assert corrected[24:27, 15:85].max() < 60


def test_unknown_background_method_is_an_input_error_not_a_skip():
    with pytest.raises(ValueError, match="Unknown background method"):
        estimate_background(np.full((5, 7), 255, np.uint8), method="typo")
