"""① Tiền xử lý (Hiếu). Hàm còn NotImplementedError → test SKIP; viết xong hàm → test tự chạy."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from common.contracts import CORNER_METHODS, PreprocessResult, Quality
from common.io import read_image
from common.paths import REPO_DIR
from common.testing import todo
from conftest import INK, PAPER, f_measure, make_photo, rotate_ccw
from preprocessing import corners, deskew, enhance, glare, pipeline, quality, rectify


@pytest.mark.parametrize("method", ["laplacian", "tenengrad", "fft"])
@todo
def test_blur_score_lower_when_blurred(page, method):
    blurred = cv2.GaussianBlur(page, (15, 15), 0)
    assert quality.blur_score(page, method) > quality.blur_score(blurred, method)


def _add_glare(img):
    out = img.copy()
    cv2.ellipse(out, (300, 400), (90, 60), 0, 0, 360, (255, 255, 255), -1)
    gt = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(gt, (300, 400), (90, 60), 0, 0, 360, 255, -1)
    return out, gt


@todo
def test_glare_mask_finds_hotspot(page):
    img, gt = _add_glare(page)
    mask = quality.glare_mask(img)
    assert mask.shape == gt.shape
    inter = np.logical_and(mask > 0, gt > 0).sum()
    union = np.logical_or(mask > 0, gt > 0).sum()
    assert inter / union > 0.5


@todo
def test_assess_quality_returns_contract(page):
    q = quality.assess_quality(page)
    assert isinstance(q, Quality)
    assert not q.is_blurry and not q.has_glare


@todo
def test_remove_glare_keeps_shape(page):
    img, gt = _add_glare(page)
    out = glare.remove_glare(img, gt, method="telea")
    assert out.shape == img.shape and out.dtype == np.uint8


@todo
def test_order_corners():
    shuffled = np.float32([[10, 300], [400, 20], [0, 0], [420, 310]])
    assert np.allclose(corners.order_corners(shuffled), [[0, 0], [400, 20], [420, 310], [10, 300]])


@pytest.mark.parametrize("finder", ["find_corners_contour", "find_corners_hough", "find_corners"])
@todo
def test_corners_within_2pct_diagonal(photo, finder):
    result = getattr(corners, finder)(photo["img"])
    if finder == "find_corners":
        result, method = result
        assert method in CORNER_METHODS
    assert result is not None
    diag = np.hypot(*photo["img"].shape[:2])
    err = np.linalg.norm(np.asarray(result) - photo["corners"], axis=1).max()
    assert err < 0.02 * diag


@todo
def test_rectify_shape_and_invertible(photo):
    warped, H = rectify.rectify(photo["img"], photo["corners"])
    assert H.shape == (3, 3) and abs(np.linalg.det(H)) > 1e-9
    h, w = warped.shape[:2]
    assert abs(h / w - 800 / 600) < 0.1


@pytest.mark.parametrize("method", ["deskew_projection", "deskew_hough", "deskew_radon"])
@todo
def test_deskew_recovers_angle(page, method):
    tilted = rotate_ccw(page, 5.0)
    _, angle = getattr(deskew, method)(tilted)
    assert abs(angle - 5.0) < 0.5


RED = (0, 0, 255)


def _red_centroid(img):
    b, g, r = cv2.split(img.astype(int))
    ys, xs = np.nonzero((r > 150) & (g < 100) & (b < 100))
    assert len(xs) > 0, "không thấy chấm đỏ"
    return np.array([xs.mean(), ys.mean()])


def _apply(H, pt):
    v = np.asarray(H, float) @ np.array([pt[0], pt[1], 1.0])
    return v[:2] / v[2]


@todo
def test_rotation_homography_matches_rotate(page):
    img = page.copy()
    cv2.circle(img, (150, 200), 8, RED, -1)
    R = deskew.rotation_homography(img.shape, 7.0)
    assert np.linalg.norm(_apply(R, (150, 200)) - _red_centroid(deskew.rotate(img, 7.0))) < 1.5


@pytest.fixture
def tilted_photo(page):
    """Giấy in lệch: chữ nghiêng 4° so với mép giấy, rồi chụp có phối cảnh. Biết vị trí chấm đỏ trên ảnh chụp."""
    img = page.copy()
    cv2.circle(img, (450, 120), 8, RED, -1)
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 4.0, 1.0)
    tilted = rotate_ccw(img, 4.0)
    shot = make_photo(tilted)
    dot_in_tilted = M @ np.array([450, 120, 1.0])
    shot["dot"] = _apply(shot["H"], dot_in_tilted)
    return shot


def _run_preprocess(tilted_photo, tmp_path):
    result = pipeline.preprocess(tilted_photo["img"], "t_contract", out_dir=tmp_path, doc_type="prescription")
    result = PreprocessResult.from_dict(result.to_dict())   # validate + roundtrip JSON
    path = Path(result.rectified_path)
    rect = read_image(path if path.is_absolute() else REPO_DIR / path)
    return result, rect


@todo
def test_preprocess_homography_includes_deskew(tilted_photo, tmp_path):
    """Hợp đồng ①→②: H = toàn bộ ảnh gốc → rectified_path, gồm cả xoay chỉnh nghiêng."""
    result, rect = _run_preprocess(tilted_photo, tmp_path)
    assert rect.ndim == 3, "rectified_path phải là ảnh màu BGR"
    # Góc đo trên ảnh đã nắn lệch khỏi 4° một chút do tỉ lệ khung ước lượng; chỉ cần chắc là đã xoay,
    # để phép kiểm tra H bên dưới thật sự có phần xoay.
    assert abs(result.skew_deg) > 2.0
    dot_rect = _red_centroid(rect)
    assert np.linalg.norm(_apply(result.homography, tilted_photo["dot"]) - dot_rect) < 3
    assert np.linalg.norm(result.to_source(dot_rect)[0] - tilted_photo["dot"]) < 3


@todo
def test_preprocess_saves_image_before_clahe(tilted_photo, tmp_path, monkeypatch):
    """Hợp đồng ①→②: ảnh gửi Trường chưa CLAHE — CLAHE chỉ chạy trên bản sao cho phân loại."""
    def fake_clahe(img, *args, **kwargs):
        return 255 - img        # dễ nhận ra nếu lỡ ghi bản này ra rectified_path
    monkeypatch.setattr(enhance, "apply_clahe", fake_clahe)
    if hasattr(pipeline, "apply_clahe"):
        monkeypatch.setattr(pipeline, "apply_clahe", fake_clahe)
    _, rect = _run_preprocess(tilted_photo, tmp_path)
    assert np.median(rect) > 150, "giấy phải sáng — ảnh đã lưu không được là bản CLAHE"


@todo
def test_clahe_keeps_shape(page, page_gray):
    assert enhance.apply_clahe(page).shape == page.shape
    assert enhance.apply_clahe(page_gray).shape == page_gray.shape


# ───────── Hồi quy cho các lỗi thấy trên ảnh thật (preview 01/11) ─────────

@todo
def test_quad_cutting_text_is_detected(photo):
    """Tứ giác bám nếp gập / khung bảng bên trong trang cắt mất chữ → text_outside_ratio cao."""
    gray = cv2.cvtColor(photo["img"], cv2.COLOR_BGR2GRAY)
    real = photo["corners"]
    assert corners.text_outside_ratio(gray, real) < corners.OUTSIDE_TEXT_MAX
    tl, tr, br, bl = real
    cut = np.float32([tl + 0.5 * (tr - tl), tr, br, bl + 0.5 * (br - bl)])   # bỏ nửa trái trang
    assert corners.text_outside_ratio(gray, cut) > corners.OUTSIDE_TEXT_MAX


@todo
def test_glare_ignores_dark_frame(page):
    """Ảnh chụp màn hình: viền tối chiếm phần lớn ảnh kéo mức sáng giấy xuống — nền trắng của trang
    không được bị coi là lóa khi đo trong tờ giấy (corners)."""
    img = np.full((2000, 1500, 3), 20, np.uint8)   # viền tối ~84% ảnh
    doc = page[:800, :600].copy()
    doc[doc > 200] = 250                     # nền trắng như ảnh chụp màn hình
    img[600:1400, 450:1050] = doc
    c = np.float32([[450, 600], [1049, 600], [1049, 1399], [450, 1399]])
    assert quality.assess_quality(img).has_glare, "dữ liệu test phải tái hiện được lỗi cũ"
    assert not quality.assess_quality(img, corners=c).has_glare


@pytest.mark.parametrize("turns", [0, 1])
@todo
def test_orientation_detects_vertical_text(page, turns):
    rotated = np.ascontiguousarray(np.rot90(page, k=turns))
    assert deskew.orientation_turns(rotated) == turns


@todo
def test_rotate90_homography_matches(page):
    img = page.copy()
    cv2.circle(img, (150, 200), 8, RED, -1)
    out, H = deskew.rotate90(img, 1)
    assert out.shape[:2] == img.shape[1::-1]
    assert np.linalg.norm(_apply(H, (150, 200)) - _red_centroid(out)) < 1.5


# ───────── Khử nhiễu muối tiêu, lóa giả trên bảng (02/11) ─────────
from preprocessing import denoise as dn


def _salt_pepper(img, d, seed=0):
    out = img.copy()
    r = np.random.default_rng(seed).random(img.shape[:2])
    out[r < d / 2] = 0
    out[(r >= d / 2) & (r < d)] = 255
    return out


@todo
def test_denoise_leaves_clean_page_untouched(page):
    out, ratio, done = dn.denoise(page)
    assert not done and ratio < dn.NOISE_RATIO_MIN and np.array_equal(out, page)


@todo
def test_switching_median_restores_text_better_than_global_median(page, text_mask):
    noisy = _salt_pepper(page, 0.03)
    ink = lambda im: np.where(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < (PAPER + INK) // 2, 255, 0)
    f_sw = f_measure(ink(dn.switching_median(noisy)), text_mask)
    assert f_sw > f_measure(ink(noisy), text_mask) + 0.05
    assert f_sw > 0.95
    assert dn.denoise(noisy)[2], "ảnh nhiễu 3% phải được phát hiện"


@todo
def test_glare_ignores_zebra_table_rows(page):
    """Bảng hàng trắng xen hàng xám (như phiếu xét nghiệm bộ 2): hàng trắng có chữ không phải lóa."""
    img = page.copy()
    for i, y in enumerate(range(0, img.shape[0], 40)):
        band = img[y:y + 40]
        band[band > 200] = 255 if i % 2 else 205
    img = _salt_pepper(img, 0.01, seed=1)
    img = dn.switching_median(img)
    assert not quality.assess_quality(img).has_glare


@todo
def test_rotate_expand_keeps_content_and_homography(page):
    img = page.copy()
    cv2.circle(img, (20, 20), 8, RED, -1)                      # sát góc: rotate thường sẽ cắt mất
    out = deskew.rotate(img, 12.0, expand=True)
    R = deskew.rotation_homography(img.shape, 12.0, expand=True)
    assert out.shape[0] > img.shape[0] and out.shape[1] > img.shape[1]
    assert np.linalg.norm(_apply(R, (20, 20)) - _red_centroid(out)) < 1.5


@todo
def test_quad_with_bad_angles_rejected():
    """Tứ giác Hough ghép nhầm cạnh chéo (ảnh thật fzSpB4S3Gw): góc trong lệch xa 90° → loại."""
    rect = np.float32([[0, 0], [600, 0], [600, 800], [0, 800]])
    bad = np.float32([[0, 0], [600, 0], [600, 800], [450, 250]])    # góc BL ≈ 42° ... vô lý cho tờ giấy
    assert corners.max_angle_deviation(rect) < 1
    assert corners.max_angle_deviation(bad) > corners.MAX_ANGLE_DEV
