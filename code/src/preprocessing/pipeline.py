"""① Ghép H1–H11 theo hợp đồng PreprocessResult (common/contracts.py)."""
import os
from pathlib import Path

import cv2
import numpy as np

from common.contracts import PreprocessResult, Quality
from common.io import write_image
from common.paths import REPO_DIR
from . import corners as C
from . import deskew, denoise, enhance, glare, quality, rectify

PARAMS = {
    "long_side": 1000,          # H1
    "median_ksize": 3,          # H2
    "blur_threshold": 448.6,    # H3 — F1 lớn nhất trên valid bộ 5 (D1)
    "tau_v": 240, "tau_s": 60,  # H4 — IoU mask lóa lớn nhất trên valid (D1)
    "glare_fix_range": (0.01, 0.08),  # H5: chỉ khử lóa nhẹ
    "canny_low": 30, "canny_high": 60,    # H6: chọn trên valid (D2), đúng T1 = 2·T2 (C5 · S47)
    "min_skew": 0.3,            # H8: bỏ qua góc nhỏ hơn (độ)
    "clahe_clip": 2.0, "clahe_tile": 8,   # H9
}


def resize_long_side(img: np.ndarray, long_side: int = 1000) -> tuple:
    """H1: đưa cạnh dài về long_side để dò; trả về (ảnh nhỏ, hệ số tỉ lệ). Nắn vẫn chạy trên ảnh gốc."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return img.copy(), 1.0
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA), s


def ink_binary(gray: np.ndarray) -> np.ndarray:
    """Ảnh nhị phân nhanh (Otsu, mực = 255) chỉ để dò góc nghiêng."""
    _, b = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def _rel(path: Path) -> str:
    try:
        return Path(os.path.relpath(path, REPO_DIR)).as_posix()
    except ValueError:
        return Path(path).as_posix()


def classify_document(rectified_bgr: np.ndarray, model=None) -> tuple:
    """H10–H11: chuẩn hóa ảnh (khử bóng + nhị phân, giảm học thuộc nguồn) → đặc trưng bố cục → kNN có từ chối.
    Trả về (doc_type, confidence). Mô hình: code/models/doc_knn.pkl do experiments/d10_classify.py tạo."""
    from classification import classify, features
    model = model or classify.load_model()
    label, conf = classify.predict_with_reject(model, features.extract_normalized(rectified_bgr)[None, :])
    return str(label[0]), float(conf[0])


def preprocess(img_bgr: np.ndarray, image_id: str, out_dir: Path, doc_type: str = None,
               params: dict = None, model=None) -> PreprocessResult:
    """Ảnh chụp → ảnh đã nắn + chỉnh nghiêng (chưa CLAHE), ma trận H tổng, 4 góc, loại tài liệu, điểm chất lượng.

    doc_type khác None thì bỏ qua bước phân loại (dùng khi classification chưa xong).
    Ghi ảnh đã nắn vào out_dir/<image_id>_rect.png.
    """
    p = {**PARAMS, **(params or {})}
    img = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    # H1–H2: ảnh nhỏ để dò, khử nhiễu trung vị
    small, s = resize_long_side(img, p["long_side"])
    gray_s = denoise.median(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), p["median_ksize"])

    # H3–H4: chất lượng
    blur = quality.blur_laplacian(gray_s)
    gmask_s = quality.glare_mask(small, p["tau_v"], p["tau_s"])
    g_ratio = quality.glare_ratio(gmask_s)

    # H5: khử lóa nhẹ trên ảnh gốc
    glare_removed = False
    lo, hi = p["glare_fix_range"]
    if lo <= g_ratio <= hi:
        gmask = cv2.resize(gmask_s, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img = glare.remove_glare(img, gmask)
        glare_removed = True

    # H6–H7: 4 góc trên ảnh nhỏ → đổi về tọa độ gốc → nắn
    corners_s, method = C.find_corners(gray_s, p["canny_low"], p["canny_high"])
    corners_src = corners_s / s
    if method == "full_image":
        corners_src = C.full_image_corners(img.shape)
        H_p, size = np.eye(3), (img.shape[1], img.shape[0])
        warped = img.copy()
    else:
        H_p, size = rectify.homography_from_corners(corners_src)
        warped = rectify.warp(img, H_p, size)

    # H8: chỉnh nghiêng phần còn sót
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    skew = deskew.skew_projection(ink_binary(gray_w))
    H = H_p
    if abs(skew) >= p["min_skew"]:
        R = deskew.rotation_matrix(skew, size)
        warped = cv2.warpAffine(warped, R[:2], size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        H = R @ H_p
        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        skew = 0.0

    rect_path = write_image(Path(out_dir) / f"{image_id}_rect.png", warped)

    # H9–H11: phân loại
    if doc_type is None:
        doc_type, conf = classify_document(warped, model)
    else:
        conf = 1.0

    q = Quality(blur_var=round(blur, 2), glare_ratio=round(g_ratio, 4),
                is_blurry=bool(blur < p["blur_threshold"]), has_glare=bool(g_ratio > lo),
                glare_removed=glare_removed)
    return PreprocessResult(image_id=image_id, rectified_path=_rel(rect_path),
                            homography=np.asarray(H, dtype=float).round(6).tolist(),
                            corners_src=np.asarray(corners_src, dtype=float).round(2).tolist(),
                            corner_method=method, skew_deg=round(float(skew), 2),
                            doc_type=doc_type, doc_confidence=round(conf, 3), quality=q).validate()
