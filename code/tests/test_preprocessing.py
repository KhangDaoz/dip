"""① Tiền xử lý — test tự SKIP khi hàm còn NotImplementedError."""
import cv2
import numpy as np

from common.testing import todo
from preprocessing import corners, deskew, enhance, quality
from conftest import rotate_ccw


@todo
def test_blur_laplacian_sharper_is_larger(page_gray):
    blurred = cv2.GaussianBlur(page_gray, (9, 9), 3)
    assert quality.blur_laplacian(page_gray) > quality.blur_laplacian(blurred)


@todo
def test_find_corners_on_photo(photo):
    found, method = corners.find_corners(cv2.cvtColor(photo["img"], cv2.COLOR_BGR2GRAY))
    assert method != "full_image"
    assert np.abs(np.asarray(found) - photo["corners"]).max() < 15


@todo
def test_skew_projection_recovers_angle(text_mask):
    rotated = rotate_ccw(text_mask, 4.0, border=0)
    assert abs(deskew.skew_projection(rotated) - 4.0) < 0.5


@todo
def test_clahe_keeps_shape(page_gray):
    assert enhance.clahe(page_gray).shape == page_gray.shape
