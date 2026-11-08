"""Step-1 specification, not real-data accuracy or permission to erase.

All assertions execute normally; NotImplementedError is a failure.
Synthetic expectations cover clear rings and contextual small text, not the
impossible claim that mask geometry can distinguish every logo/letter/seal.
"""
import cv2
import numpy as np
import pytest

from cleaning.ink import split_stamp_vs_text


def canvas():
    return np.zeros((320, 480), np.uint8)


def draw_text(mask, text, origin, scale, thickness):
    cv2.putText(mask, text, origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=scale, color=255, thickness=thickness, lineType=cv2.LINE_8)
    # Some OpenCV builds antialias text despite LINE_8. This fixture represents
    # a binary candidate mask, not a grayscale rendering. Keep every drawn pixel.
    mask[mask > 0] = 255


def partition(mask):
    before = mask.copy()
    result = split_stamp_vs_text(mask)
    assert isinstance(result, tuple) and len(result) == 2
    stamp, keep = result
    for out in result:
        assert isinstance(out, np.ndarray)
        assert out.shape == mask.shape and out.dtype == np.uint8
        assert set(np.unique(out)) <= {0, 255}
        assert not np.shares_memory(out, mask)
    assert not np.shares_memory(stamp, keep)
    assert not np.any((stamp > 0) & (keep > 0))
    np.testing.assert_array_equal(stamp | keep, mask)
    np.testing.assert_array_equal(mask, before)
    return stamp, keep


@pytest.mark.parametrize("bad", [None, [], np.zeros((0, 5), np.uint8),
    np.zeros((5, 0), np.uint8), np.zeros((5,), np.uint8),
    np.zeros((5, 5, 1), np.uint8), np.zeros((5, 5), bool),
    np.zeros((5, 5), np.float32), np.zeros((5, 5), np.uint16),
    np.array([[0, 1, 255]], np.uint8), np.array([[128]], np.uint8)])
def test_invalid_masks_raise_value_error(bad):
    with pytest.raises(ValueError):
        split_stamp_vs_text(bad)


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (17, 1), (320, 480)])
def test_zero_masks(shape):
    stamp, keep = partition(np.zeros(shape, np.uint8))
    assert not stamp.any() and not keep.any()


def test_readonly_noncontiguous_mask_and_independent_outputs():
    backing = np.zeros((320, 960), np.uint8)
    cv2.circle(backing, (250, 160), 70, 255, 4)
    mask = backing[:, ::2]
    original = backing.copy()
    mask.setflags(write=False)
    stamp, keep = partition(mask)
    stamp[:] = 0
    keep[:] = 255
    np.testing.assert_array_equal(backing, original)


@pytest.mark.parametrize("axes,angle", [((65, 65), 0), ((80, 55), 0),
                                         ((80, 55), 35)])
def test_clear_isolated_ring_is_stamp_without_filling_its_interior(axes, angle):
    mask = canvas()
    cv2.ellipse(mask, (160, 160), axes, angle, 0, 360, 255, 4)
    stamp, keep = partition(mask)
    # At least the clear rim must be recognized; zero predictions cannot pass.
    assert np.count_nonzero(stamp) / np.count_nonzero(mask) >= .9
    assert stamp[160, 160] == 0


def test_separate_red_text_is_kept_next_to_clear_stamp():
    ring = canvas()
    cv2.circle(ring, (110, 160), 65, 255, 4)
    text = canvas()
    draw_text(text, "HIGH 10.5", (230, 160), .65, 1)
    stamp, keep = partition(ring | text)
    assert np.count_nonzero(stamp & ring) / np.count_nonzero(ring) >= .9
    assert not np.any(stamp & text)
    np.testing.assert_array_equal(keep & text, text)


@pytest.mark.parametrize("kind", ["text", "small_rings", "frame", "solid", "dot"])
def test_negative_shapes_are_kept(kind):
    mask = canvas()
    if kind == "text":
        draw_text(mask, "O0 HIGH 10.5", (30, 160), .7, 2)
    elif kind == "small_rings":
        for x in range(40, 240, 30):
            cv2.circle(mask, (x, 160), 7, 255, 1)
    elif kind == "frame":
        cv2.rectangle(mask, (40, 60), (430, 260), 255, 3)
    elif kind == "solid":
        cv2.circle(mask, (160, 160), 65, 255, -1)
    else:
        mask[100, 100] = 255
    stamp, keep = partition(mask)
    assert not stamp.any()
    np.testing.assert_array_equal(keep, mask)


@pytest.mark.parametrize("kind", ["broken", "clipped", "touching", "interior_text"])
def test_difficult_candidates_preserve_pixels_and_protected_text(kind):
    ring = canvas()
    center = (0, 160) if kind == "clipped" else (160, 160)
    cv2.circle(ring, center, 65, 255, 4)
    protected = canvas()
    if kind == "broken":
        ring[90:110, 145:175] = 0
    elif kind in ("touching", "interior_text"):
        origin = (210, 165) if kind == "touching" else (125, 165)
        draw_text(protected, "HIGH", origin, .5, 1)
    if kind == "touching":
        assert np.any(ring & protected), "Fixture must really touch the rim"
    elif kind == "interior_text":
        assert protected.any() and not np.any(ring & protected)
    stamp, keep = partition(ring | protected)
    # No demand to guess a difficult stamp. Independent interior/touching text
    # must not be swallowed by an ellipse fill or whole-component assignment.
    assert not np.any(stamp & protected)
    np.testing.assert_array_equal(keep & protected, protected)
    if kind in ("broken", "clipped", "touching"):
        # These are explicitly retained by the closed-isolated-rim baseline.
        # Check actual behavior, rather than only a vacuous zero text mask.
        assert not stamp.any()
        np.testing.assert_array_equal(keep, ring | protected)


def test_arbitrary_binary_content_preserves_partition():
    mask = np.random.default_rng(42).choice(
        np.array([0, 255], np.uint8), size=(73, 91))
    partition(mask)


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_ring_detection_at_multiple_page_resolutions(scale):
    mask = np.zeros((320 * scale, 480 * scale), np.uint8)
    cv2.ellipse(mask, (160 * scale, 160 * scale), (80 * scale, 55 * scale),
                35, 0, 360, 255, 4 * scale)
    stamp, _ = partition(mask)
    assert np.count_nonzero(stamp) / np.count_nonzero(mask) >= .9


def test_multiple_separate_rims_and_inner_text():
    mask = canvas()
    cv2.circle(mask, (100, 160), 60, 255, 4)
    cv2.circle(mask, (350, 160), 60, 255, 4)
    text = canvas()
    draw_text(text, "HIGH", (70, 165), .5, 1)
    stamp, keep = partition(mask | text)
    np.testing.assert_array_equal(stamp, mask)
    np.testing.assert_array_equal(keep, text)


def test_long_narrow_ellipse_is_not_a_stamp_candidate():
    mask = canvas()
    cv2.ellipse(mask, (200, 160), (140, 25), 0, 0, 360, 255, 3)
    stamp, keep = partition(mask)
    assert not stamp.any()
    np.testing.assert_array_equal(keep, mask)


@pytest.mark.parametrize("center", [(0, 160), (479, 160), (160, 0), (160, 319)])
def test_rims_cut_by_each_page_boundary_are_kept(center):
    mask = canvas()
    cv2.circle(mask, center, 65, 255, 4)
    stamp, keep = partition(mask)
    assert not stamp.any()
    np.testing.assert_array_equal(keep, mask)


@pytest.mark.parametrize("gap", [8, 20, 40])
def test_open_rim_at_several_gap_widths_is_not_forced_into_stamp(gap):
    mask = canvas()
    cv2.circle(mask, (160, 160), 65, 255, 4)
    mask[90:110, 160-gap//2:160+gap//2] = 0
    stamp, keep = partition(mask)
    assert not stamp.any()
    np.testing.assert_array_equal(keep, mask)


@pytest.mark.parametrize("scale", [.4, .7, 1.0])
def test_typical_red_O_and_zero_sizes_are_kept(scale):
    mask = canvas()
    draw_text(mask, 'O0 10.0', (30, 160), scale, 2)
    stamp, keep = partition(mask)
    assert not stamp.any()
    np.testing.assert_array_equal(keep, mask)


def test_red_text_connected_across_rim_is_kept_as_a_whole():
    mask = canvas()
    cv2.circle(mask, (160, 160), 65, 255, 4)
    text = canvas()
    draw_text(text, 'HIGH', (210, 165), .5, 1)
    assert np.any(mask & text)
    stamp, keep = partition(mask | text)
    assert not stamp.any()
    np.testing.assert_array_equal(keep, mask | text)


def test_identical_ring_geometry_cannot_encode_seal_vs_logo_semantics():
    # Documents a limitation, not a claim that recognizing logos is correct.
    # A future image/context-aware API is needed to distinguish identical masks.
    mask = canvas()
    cv2.circle(mask, (160, 160), 65, 255, 4)
    stamp, _ = partition(mask)
    np.testing.assert_array_equal(stamp, mask)
