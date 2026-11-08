"""Specification for remove_marks; no datasets, OCR or quality benchmark.

All tests execute normally; NotImplementedError is a failure.
"""
import cv2
import numpy as np
import pytest

from cleaning.inpaint import remove_marks

METHODS = ('telea', 'ns', 'fill_white')


def sample(color=False):
    yy, xx = np.indices((35, 47))
    gray = (130 + xx + yy).astype(np.uint8)
    image = np.stack((gray, gray + 10, gray + 20), axis=2) if color else gray
    mask = np.zeros(gray.shape, np.uint8)
    mask[12:17, 20:25] = 255
    image[mask > 0] = (10, 20, 200) if color else 10
    return image, mask


def checked(image, mask, **kwargs):
    original, original_mask = image.copy(), mask.copy()
    out = remove_marks(image, mask, **kwargs)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8 and out.shape == image.shape
    assert not np.shares_memory(out, image)
    assert not np.shares_memory(out, mask)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(mask, original_mask)
    np.testing.assert_array_equal(out[mask == 0], image[mask == 0])
    return out


@pytest.mark.parametrize('color', [False, True])
@pytest.mark.parametrize('method', METHODS)
def test_output_contract_and_nontrivial_processing(color, method):
    image, mask = sample(color)
    out = checked(image, mask, method=method)
    assert np.any(out[mask > 0] != image[mask > 0])


@pytest.mark.parametrize('color', [False, True])
@pytest.mark.parametrize('method,flag', [('telea', cv2.INPAINT_TELEA), ('ns', cv2.INPAINT_NS)])
@pytest.mark.parametrize('radius', [1, 3, 5])
def test_method_and_radius_match_opencv_inside_mask(color, method, flag, radius):
    image, mask = sample(color)
    expected = cv2.inpaint(image.copy(), mask.copy(), radius, flag)
    out = checked(image, mask, method=method, radius=radius)
    # Confirms public method dispatch, radius, BGR order and native intensity
    # scale; comparison outside mask is handled independently by checked().
    np.testing.assert_array_equal(out[mask > 0], expected[mask > 0])


@pytest.mark.parametrize('color', [False, True])
def test_fill_white_is_exact_white_and_radius_independent(color):
    image, mask = sample(color)
    for radius in [1, 3, 9]:
        out = checked(image, mask, method='fill_white', radius=radius)
        assert np.all(out[mask > 0] == 255)


def test_defaults_are_telea_radius_three():
    image, mask = sample(True)
    np.testing.assert_array_equal(checked(image, mask),
                                  checked(image, mask, method='telea', radius=3))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('color', [False, True])
def test_zero_mask_returns_independent_unchanged_image(method, color):
    image, mask = sample(color)
    mask[:] = 0
    np.testing.assert_array_equal(checked(image, mask, method=method), image)


@pytest.mark.parametrize('color', [False, True])
@pytest.mark.parametrize('method', METHODS)
def test_full_mask_has_explicit_policy(method, color):
    image, mask = sample(color)
    mask[:] = 255
    if method == 'fill_white':
        assert np.all(checked(image, mask, method=method) == 255)
    else:
        before = image.copy()
        with pytest.raises(ValueError):
            remove_marks(image, mask, method=method)
        np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('bad', [None, [], np.zeros((0, 4), np.uint8),
    np.zeros((4, 0), np.uint8), np.zeros((4,), np.uint8),
    np.zeros((4, 5, 1), np.uint8), np.zeros((4, 5, 4), np.uint8),
    np.zeros((4, 5), np.float32), np.zeros((4, 5), np.uint16),
    np.zeros((4, 5), bool)])
def test_invalid_image_rejected(bad):
    with pytest.raises(ValueError):
        remove_marks(bad, np.zeros((4, 5), np.uint8))


@pytest.mark.parametrize('bad', [None, [], np.zeros((0, 5), np.uint8),
    np.zeros((4, 4), np.uint8), np.zeros((4, 5, 1), np.uint8),
    np.zeros((4, 5), bool), np.zeros((4, 5), np.float32),
    np.zeros((4, 5), np.uint16), np.ones((4, 5), np.uint8),
    np.full((4, 5), 128, np.uint8)])
def test_invalid_mask_rejected(bad):
    with pytest.raises(ValueError):
        remove_marks(np.full((4, 5), 220, np.uint8), bad)


@pytest.mark.parametrize('method', [None, '', 'Telea', 'navier_stokes', 'white', 1, [], {}])
def test_invalid_method_rejected_even_for_empty_mask(method):
    image, mask = sample()
    mask[:] = 0
    with pytest.raises(ValueError):
        remove_marks(image, mask, method=method)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('radius', [0, -1, 1.5, 3.0, True, np.bool_(False),
                                   None, '3', float('nan'), float('inf')])
def test_invalid_radius_rejected_even_for_empty_mask(method, radius):
    image, mask = sample()
    mask[:] = 0
    with pytest.raises(ValueError):
        remove_marks(image, mask, method=method, radius=radius)


@pytest.mark.parametrize('method', METHODS)
def test_numpy_integer_radius_supported(method):
    image, mask = sample()
    checked(image, mask, method=method, radius=np.int64(3))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('color', [False, True])
def test_readonly_strided_image_and_mask(method, color):
    image, mask = sample(color)
    image = image[::-1, ::2]
    mask = mask[::-1, ::2]
    assert not image.flags.c_contiguous and not mask.flags.c_contiguous
    image.setflags(write=False)
    mask.setflags(write=False)
    checked(image, mask, method=method)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('kind', ['corner', 'edge', 'multiple', 'single', 'almost_full'])
def test_boundary_and_sparse_masks_preserve_unmasked_pixels(method, kind):
    image, mask = sample(True)
    mask[:] = 0
    if kind == 'corner':
        mask[:3, :3] = 255
    elif kind == 'edge':
        mask[10:15, -2:] = 255
    elif kind == 'multiple':
        mask[5:8, 5:8] = mask[20:23, 30:33] = 255
    elif kind == 'single':
        mask[15, 20] = 255
    else:
        mask[:] = 255
        mask[0, 0] = 0
    checked(image, mask, method=method)


@pytest.mark.parametrize('shape', [(1, 1), (1, 9), (9, 1)])
@pytest.mark.parametrize('method', METHODS)
def test_tiny_zero_masks_return_copy(shape, method):
    image = np.full(shape, 200, np.uint8)
    mask = np.zeros(shape, np.uint8)
    np.testing.assert_array_equal(checked(image, mask, method=method), image)


@pytest.mark.parametrize('shape', [(1, 9), (9, 1)])
@pytest.mark.parametrize('method', METHODS)
def test_thin_images_with_partial_mask(shape, method):
    image = np.full(shape, 200, np.uint8)
    mask = np.zeros(shape, np.uint8)
    mask.reshape(-1)[4] = 255
    checked(image, mask, method=method)


@pytest.mark.parametrize('method', METHODS)
def test_constant_gray_paper_is_reconstructed_without_silent_whitening(method):
    image = np.full((60, 80, 3), 200, np.uint8)
    mask = np.zeros((60, 80), np.uint8)
    mask[25:30, 35:40] = 255
    image[mask > 0] = (20, 35, 210)
    out = checked(image, mask, method=method)
    if method == 'fill_white':
        assert np.all(out[mask > 0] == 255)
    else:
        # Tolerance permits interpolation/rounding, but rejects a no-op, wrong
        # intensity scale, BGR leakage or accidental fill-white dispatch.
        assert np.max(np.abs(out[mask > 0].astype(int) - 200)) <= 5


@pytest.mark.parametrize('method', METHODS)
def test_image_and_mask_may_alias_without_mutating_either(method):
    image = np.zeros((20, 30), np.uint8)
    image[8:12, 13:17] = 255
    # Valid grayscale image and valid mask stored in the exact same array.
    out = checked(image, image, method=method)
    if method == 'fill_white':
        np.testing.assert_array_equal(out, image)
    else:
        assert np.max(out[image > 0]) <= 5


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('empty_mask', [False, True])
def test_unrepresentable_radius_raises_value_error_before_backend(method, empty_mask):
    image, mask = sample()
    if empty_mask:
        mask[:] = 0
    before, before_mask = image.copy(), mask.copy()
    with pytest.raises(ValueError, match='finite float'):
        remove_marks(image, mask, method=method, radius=10**400)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(mask, before_mask)
