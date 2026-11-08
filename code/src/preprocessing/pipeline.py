"""Ghép các bước của ① thành một hàm, trả về PreprocessResult (hợp đồng ① → ②)."""
from pathlib import Path

import numpy as np

from common.contracts import PreprocessResult
from common.io import to_gray, write_image
from common.paths import OUT_DIR, REPO_DIR
from . import enhance
from .corners import find_corners
from .denoise import DEFAULT_METHOD as DEFAULT_DENOISE, denoise
from .deskew import deskew_hough, deskew_projection, deskew_radon, orientation_turns, rotate, rotate90, \
    rotation_homography
from .glare import remove_glare
from .quality import GLARE_HEAVY, GLARE_LIGHT, assess_quality, glare_mask, quad_mask
from .rectify import rectify

DESKEW = {"projection": deskew_projection, "hough": deskew_hough, "radon": deskew_radon}
RESIDUAL_MAX = 1.0   # độ: góc đo lại trên ảnh đã xoay vượt ngưỡng → hủy bước chỉnh nghiêng


def _rel_to_repo(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def _classify(img_rect: np.ndarray, classifier):
    """CLAHE trên BẢN SAO → đặc trưng → bộ phân loại. Trả về (doc_type, confidence)."""
    from classification.classify import REJECT, predict_with_reject
    from classification.features import extract_all
    gray = to_gray(enhance.apply_clahe(img_rect.copy()))
    labels, conf = predict_with_reject(classifier, extract_all(gray)[None])
    label = "other" if labels[0] == REJECT else str(labels[0])
    return label, float(conf[0])


def preprocess(img: np.ndarray, image_id: str, out_dir: Path = OUT_DIR,
               classifier=None, doc_type: str = None, deskew_method: str = "projection",
               denoise_method: str = DEFAULT_DENOISE, inpaint_glare: bool = False) -> PreprocessResult:
    """Các bước:
        0. denoise: có nhiễu muối tiêu (> 0,05% điểm nhiễu xung) → trung vị chọn lọc (D11); không đổi hình học
        1. find_corners; assess_quality đo trong tờ giấy; lóa nhẹ (1–8%) và inpaint_glare → remove_glare
           (mặc định TẮT: inpainting không khôi phục được chữ, báo lóa nhầm thì xóa luôn chữ — chỉ bật khi D5
           trên bộ 6 v2 chứng minh CER tốt lên; quality.has_glare vẫn báo cho app)
        2. rectify theo 4 góc                           → (warped, H_persp); full_image thì giữ nguyên ảnh
        3. orientation_turns: dòng chữ chạy dọc → xoay 90° → (oriented, H_turn)
        4. deskew trên oriented                         → (rect, skew_deg); đo lại trên rect, còn lệch > RESIDUAL_MAX → không xoay
           rect = rotate(oriented, -skew_deg, expand=True) nên R = rotation_homography(..., expand=True)
        5. Lưu rect vào out_dir/<image_id>_rect.png     ← BGR, CHƯA CLAHE, đây là ảnh gửi ②
           homography = R @ H_turn @ H_persp            ← toàn bộ ảnh gốc → rect
        6. apply_clahe trên BẢN SAO của rect → features → classifier → doc_type, doc_confidence

    rectified_path ghi tương đối với REPO_DIR nếu out_dir nằm trong repo, ngược lại ghi đường dẫn tuyệt đối.
    classifier: mô hình đã train của classification.classify; None thì dùng doc_type truyền vào
    (để chạy tích hợp khi chưa train xong).
    deskew_method: "projection" | "hough" | "radon" (chọn theo D4).
    denoise_method: khóa của denoise.METHODS hoặc None để bỏ bước 0.
    """
    if classifier is None and doc_type is None:
        raise NotImplementedError("phân loại chưa có — truyền doc_type")

    if denoise_method is not None:
        img, _, _ = denoise(img, denoise_method)
    corners, corner_method = find_corners(img)
    quality = assess_quality(img, corners=corners)
    work = img
    if inpaint_glare and GLARE_LIGHT <= quality.glare_ratio <= GLARE_HEAVY:
        work = remove_glare(img, glare_mask(img, region=quad_mask(img.shape, corners)))

    if corner_method == "full_image":
        warped, H_persp = work, np.eye(3)
    else:
        warped, H_persp = rectify(work, corners)

    oriented, H_turn = rotate90(warped, orientation_turns(warped))
    _, skew_deg = DESKEW[deskew_method](oriented)
    rect = rotate(oriented, -skew_deg, expand=True)                 # nới khung: không cắt mất góc nội dung
    if skew_deg and abs(DESKEW[deskew_method](rect)[1]) > RESIDUAL_MAX:
        # Tự kiểm tra: xoay đúng thì đo lại phải ≈ 0°. Còn lệch → phép đo bị nội dung lạ đánh lừa
        # (tờ giấy phía sau, ít chữ, nắn méo) → không xoay, an toàn hơn xoay sai (review 02/11).
        skew_deg, rect = 0.0, oriented.copy()
    R = rotation_homography(oriented.shape, -skew_deg, expand=True)

    rect_path = write_image(Path(out_dir) / f"{image_id}_rect.png", rect)
    if classifier is not None:
        doc_type, doc_confidence = _classify(rect, classifier)
    else:
        doc_confidence = 1.0

    return PreprocessResult(
        image_id=image_id,
        rectified_path=_rel_to_repo(rect_path),
        homography=(R @ H_turn @ H_persp).tolist(),
        corners_src=np.asarray(corners, float).tolist(),
        corner_method=corner_method,
        skew_deg=float(skew_deg),
        doc_type=doc_type,
        doc_confidence=doc_confidence,
        quality=quality,
    ).validate()
