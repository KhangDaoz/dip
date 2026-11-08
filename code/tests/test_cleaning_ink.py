"""Known-color and geometric stroke checks; no dataset or OCR required."""
import cv2
import numpy as np
import pytest

from cleaning.ink import ink_masks


def assert_contract(masks, image):
    assert set(masks) == {"red", "blue", "black"}
    for mask in masks.values():
        assert mask.shape == image.shape[:2]
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 255}
        assert not np.shares_memory(mask, image)
    assert np.all(sum((m > 0).astype(int) for m in masks.values()) <= 1)


@pytest.mark.parametrize("value", [0, 1, 40, 100, 180, 225, 255])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (17, 1), (64, 80)])
def test_uniform_neutral_background_has_no_ink(value, shape):
    image = np.full((*shape, 3), value, np.uint8)
    masks = ink_masks(image)
    assert_contract(masks, image)
    assert not any(mask.any() for mask in masks.values())


@pytest.mark.parametrize("image", [None, [], np.zeros((4, 5), np.uint8),
    np.zeros((4, 5, 1), np.uint8), np.zeros((4, 5, 4), np.uint8),
    np.zeros((0, 5, 3), np.uint8), np.zeros((4, 0, 3), np.uint8),
    np.zeros((4, 5, 3), np.float32), np.zeros((4, 5, 3), np.uint16),
    np.zeros((4, 5, 3), bool), np.zeros((1, 4, 5, 3), np.uint8)])
def test_invalid_images_raise_clear_value_error(image):
    with pytest.raises(ValueError, match="uint8 BGR"):
        ink_masks(image)


def test_known_bgr_colors_separate_without_background_or_cross_talk():
    image = np.full((100, 180, 3), 230, np.uint8)
    expected = {key: np.zeros(image.shape[:2], bool) for key in ("red", "blue", "black")}
    # Thin independent strokes, known BGR ordering; includes an unassigned green.
    for name, column, color in [("red", 20, (0, 0, 220)),
                               ("blue", 60, (220, 0, 0)),
                               ("black", 100, (20, 20, 20))]:
        image[20:80, column:column+2] = color
        expected[name][20:80, column:column+2] = True
    image[20:80, 140:142] = (0, 200, 0)
    original = image.copy()
    masks = ink_masks(image)
    assert_contract(masks, image)
    assert np.array_equal(image, original)
    for name, gt in expected.items():
        assert np.array_equal(masks[name] > 0, gt), name


@pytest.mark.parametrize("hue,expected", [(0,"red"),(10,"red"),(170,"red"),
    (179,"red"),(11,None),(169,None),(100,"blue"),(120,"blue"),
    (130,"blue"),(99,None),(131,None),(60,None)])
def test_hue_ranges_and_red_wraparound(hue, expected):
    image = np.full((45, 45, 3), 240, np.uint8)
    bgr = cv2.cvtColor(np.uint8([[[hue,255,220]]]), cv2.COLOR_HSV2BGR)[0,0]
    image[20,20] = bgr
    masks = ink_masks(image)
    assert_contract(masks, image)
    for name, mask in masks.items():
        assert bool(mask[20,20]) == (name == expected)
    assert sum(np.count_nonzero(mask) for mask in masks.values()) == (expected is not None)


@pytest.mark.parametrize("hue,saturation,expected", [(0,80,None),(0,81,"red"),
    (120,60,None),(120,61,"blue"),(0,0,None)])
def test_saturation_boundary_does_not_turn_pale_color_into_black(hue, saturation, expected):
    image = np.full((45,45,3),255,np.uint8)
    bgr = cv2.cvtColor(np.uint8([[[hue,saturation,255]]]),cv2.COLOR_HSV2BGR)[0,0]
    image[20,20] = bgr
    masks=ink_masks(image)
    for name, mask in masks.items():
        assert bool(mask[20,20]) == (name == expected)


@pytest.mark.parametrize("color,expected", [((0,0,40),"red"),((0,0,39),None),
    ((40,0,0),"blue"),((39,0,0),None)])
def test_color_value_floor_is_explicit(color, expected):
    image=np.full((45,45,3),230,np.uint8)
    image[20,20]=color
    masks=ink_masks(image)
    for name,mask in masks.items():
        assert bool(mask[20,20]) == (name == expected)


def test_dark_pale_blue_is_unassigned_rather_than_black():
    image=np.full((45,45,3),230,np.uint8)
    image[20,20]=cv2.cvtColor(np.uint8([[[120,60,120]]]),cv2.COLOR_HSV2BGR)[0,0]
    masks=ink_masks(image)
    assert not any(mask[20,20] for mask in masks.values())


def test_dark_nearly_neutral_stroke_can_still_be_black():
    image=np.full((45,45,3),230,np.uint8)
    image[20,20]=(30,30,35)
    masks=ink_masks(image)
    assert masks['black'][20,20]==255
    assert masks['red'][20,20]==0 and masks['blue'][20,20]==0


def test_single_pixel_dots_and_thin_black_strokes_survive():
    image=np.full((80,100,3),230,np.uint8)
    gt=np.zeros((80,100),bool)
    gt[20,20]=True
    gt[40,20:80]=True
    gt[20:60,80]=True
    image[gt]=25
    masks=ink_masks(image)
    assert np.array_equal(masks['black']>0,gt)
    assert not masks['red'].any() and not masks['blue'].any()


def test_uneven_illumination_preserves_strokes_without_black_paper():
    paper=np.tile(np.linspace(100,240,180).round().astype(np.uint8),(100,1))
    gt=np.zeros(paper.shape,bool)
    gt[20:80:12,15:165]=True
    image=np.repeat(paper[...,None],3,axis=2)
    image[gt]=(image[gt].astype(float)*0.2).astype(np.uint8)
    masks=ink_masks(image)
    assert np.array_equal(masks['black']>0,gt)
    assert not masks['red'].any() and not masks['blue'].any()


def test_read_only_noncontiguous_input_and_masks_are_independent():
    backing=np.full((40,90,3),225,np.uint8)
    backing[10:30,20:22]=20
    image=backing[:,::2,:]
    before=backing.copy()
    image.setflags(write=False)
    masks=ink_masks(image)
    assert_contract(masks,image)
    assert np.array_equal(backing,before)
    for first,second in [('red','blue'),('red','black'),('blue','black')]:
        assert not np.shares_memory(masks[first],masks[second])


def test_colored_background_is_color_candidate_not_semantic_stamp():
    # Deliberately documents the HSV limitation: this is not a stamp detector.
    image=np.full((40,50,3),(0,0,220),np.uint8)
    masks=ink_masks(image)
    assert np.all(masks['red']==255)
    assert not masks['blue'].any() and not masks['black'].any()


@pytest.mark.parametrize("paper,stroke", [(240, 225), (180, 166), (120, 108)])
def test_faint_gray_dots_and_strokes_keep_their_known_geometry(paper, stroke):
    image = np.full((100, 140, 3), paper, np.uint8)
    gt = np.zeros((100, 140), bool)
    gt[20, 20] = True
    gt[40, 20:120] = True
    gt[60:80, 80] = True
    image[gt] = stroke
    masks = ink_masks(image)
    assert np.array_equal(masks['black'] > 0, gt)


@pytest.mark.parametrize("paper", [(185, 215, 235), (235, 210, 185)])
def test_dark_ink_with_same_tint_as_paper_is_not_discarded(paper):
    image = np.full((100, 140, 3), paper, np.uint8)
    gt = np.zeros((100, 140), bool)
    gt[30:70, 40:42] = True
    gt[70, 30:110] = True
    image[gt] = np.rint(np.array(paper) * .4).astype(np.uint8)
    masks = ink_masks(image)
    assert np.array_equal(masks['black'] > 0, gt)
    assert not masks['red'].any() and not masks['blue'].any()


def test_faint_strokes_over_pastel_columns_do_not_fill_the_columns():
    image = np.full((100, 180, 3), 240, np.uint8)
    image[:, 60:120] = (220, 240, 220)
    image[:, 120:] = (230, 225, 240)
    gt = np.zeros((100, 180), bool)
    gt[40, 15:165] = True
    gt[70, 15:165] = True
    image[gt] = np.maximum(image[gt].astype(int) - 16, 0).astype(np.uint8)
    masks = ink_masks(image)
    assert np.array_equal(masks['black'] > 0, gt)
    assert not masks['red'].any() and not masks['blue'].any()


def test_clean_step_in_paper_brightness_is_not_a_new_weak_stroke():
    image = np.full((80, 160, 3), 240, np.uint8)
    image[:, :80] = 215
    assert not ink_masks(image)['black'].any()


def test_low_amplitude_paper_noise_does_not_become_weak_ink():
    rng = np.random.default_rng(29)
    values = rng.integers(222, 228, (80, 100), dtype=np.uint8)
    image = np.repeat(values[..., None], 3, axis=2)
    assert not ink_masks(image)['black'].any()


@pytest.mark.parametrize("color", [(80, 96, 96), (40, 50, 50)])
def test_nearly_neutral_dark_stroke_with_small_color_cast_is_kept(color):
    image = np.full((80, 100, 3), 230, np.uint8)
    gt = np.zeros((80, 100), bool)
    gt[20:60, 50] = True
    image[gt] = color
    assert cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0, 1] > 40
    assert np.array_equal(ink_masks(image)['black'] > 0, gt)


def test_saturated_purple_graphic_is_not_black_even_when_matching_paper_hue():
    image = np.full((100, 140, 3), (180, 90, 150), np.uint8)
    image[30:70, 40:42] = (108, 54, 90)
    masks = ink_masks(image)
    # Purple is deliberately outside red/blue ranges. Normalizing by a purple
    # background must not silently relabel the purple stroke as black.
    assert not any(mask.any() for mask in masks.values())


def test_near_black_chroma_noise_does_not_destroy_neutral_stroke():
    image = np.full((80, 100, 3), 235, np.uint8)
    expected = np.zeros((80, 100), bool)
    expected[30:50, 50] = True
    image[expected] = (2, 4, 3)
    assert np.array_equal(ink_masks(image)['black'] > 0, expected)


def test_moderate_green_cast_fringe_requires_nearby_neutral_core():
    image = np.full((80, 100, 3), 235, np.uint8)
    image[30:50, 50] = 20
    image[30:50, 51] = (22, 40, 28)
    # Same chromatic stroke, but remote from a neutral core, stays unassigned.
    image[30:50, 75] = (22, 40, 28)
    black = ink_masks(image)['black']
    assert np.all(black[30:50, 50:52] == 255)
    assert not black[30:50, 75].any()


@pytest.mark.parametrize('color', [(120, 92, 92), (92, 92, 120)])
def test_pale_blue_or_red_next_to_black_does_not_become_black(color):
    image = np.full((80, 100, 3), 235, np.uint8)
    image[30:50, 50] = 20
    image[30:50, 51] = color
    black = ink_masks(image)['black']
    assert np.all(black[30:50, 50] == 255)
    assert not black[30:50, 51].any()


def test_weak_edge_has_intensity_evidence_and_neutral_neighbor_support():
    image = np.full((100, 140, 3), 235, np.uint8)
    image[30:70, 50] = 190
    image[30:70, 51] = 225
    image[30:70, 100] = 229  # Isolated weaker line, no existing stroke support.
    black = ink_masks(image)['black']
    assert np.all(black[35:65, 50:52] == 255)
    assert not black[35:65, 100].any()


def test_neighbor_support_does_not_select_paper_as_dilation_would():
    image = np.full((100, 140, 3), 235, np.uint8)
    image[30:70, 50] = 190
    black = ink_masks(image)['black']
    assert np.all(black[35:65, 50] == 255)
    assert not black[:, 49].any() and not black[:, 51].any()


def test_new_color_recovery_does_not_normalize_dark_colored_graphic():
    image = np.full((100, 140, 3), (20, 30, 20), np.uint8)
    image[30:70, 50] = (1, 2, 1)
    assert not ink_masks(image)['black'].any()
