"""② Làm sạch (Trường)."""
import cv2
import numpy as np
import pytest

from cleaning import background, binarize, ink
from common.testing import todo
from conftest import f_measure


@pytest.mark.parametrize("name", list(binarize.METHODS))
@todo
def test_binarize_output_convention(page_gray, text_mask, name):
    out = binarize.METHODS[name](page_gray)
    assert out.shape == page_gray.shape and out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}, "mực = 255, nền = 0"
    recall = np.logical_and(out > 0, text_mask > 0).sum() / (text_mask > 0).sum()
    assert recall > 0.8


@pytest.mark.parametrize("name", ["otsu", "sauvola", "wolf", "gaussian"])
@todo
def test_binarize_f_measure(page_gray, text_mask, name):
    assert f_measure(binarize.METHODS[name](page_gray), text_mask) > 0.8


@todo
def test_remove_shadow_flattens_background(page_gray, text_mask):
    gradient = np.tile(np.linspace(0.4, 1.0, page_gray.shape[1]), (page_gray.shape[0], 1))
    shadowed = (page_gray * gradient).astype(np.uint8)
    bg = background.estimate_background(shadowed)
    flat = background.remove_shadow(shadowed, bg)
    paper = text_mask == 0
    assert flat[paper].std() < shadowed[paper].std() / 2


@todo
def test_ink_masks_finds_red_stamp(page):
    img = page.copy()
    cv2.circle(img, (450, 650), 70, (40, 40, 210), 6)
    gt = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(gt, (450, 650), 70, 255, 6)
    masks = ink.ink_masks(img)
    assert set(masks) >= {"red", "blue", "black"}
    inter = np.logical_and(masks["red"] > 0, gt > 0).sum()
    assert inter / (gt > 0).sum() > 0.7
