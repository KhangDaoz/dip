"""Đánh giá chất lượng ảnh (IQA): đo mờ, đo lóa. Thí nghiệm D1."""
import cv2
import numpy as np

from common.contracts import Quality
from common.io import to_gray

BLUR_THRESHOLD = 100.0                  # chọn lại bằng ROC (D1)
GLARE_V, GLARE_S = 230, 30              # pixel lóa: V > GLARE_V và S < GLARE_S — D1 (bộ 6 v2.1): V 230 IoU 0,62 / AUC 0,93 (240: 0,50 / 0,86); S gần như không ảnh hưởng
GLARE_DELTA = 15                        # và sáng hơn mức giấy trung vị ít nhất GLARE_DELTA
GLARE_LIGHT, GLARE_HEAVY = 0.01, 0.08   # < 1%: không lóa · 1–8%: lóa nhẹ (khử) · > 8%: chụp lại
GLARE_MIN_AREA_RATIO = 0.001            # một vùng lóa phải ≥ 0,1% diện tích giấy (chỉnh lại trên bộ 6 v2, D1)
GLARE_PAPER_PCT = 70                    # mức giấy = phân vị 70 của V trong tờ giấy (trung vị bị hàng nền xám của bảng kéo xuống)


def resize_long_side(img: np.ndarray, long_side: int):
    """Resize để cạnh dài = long_side. Trả về (ảnh, scale) với tọa độ_nhỏ = tọa độ_gốc × scale."""
    scale = long_side / max(img.shape[:2])
    if abs(scale - 1) < 1e-6:
        return img, 1.0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp), scale


def blur_score(img: np.ndarray, method: str = "laplacian") -> float:
    """Điểm độ nét, càng lớn càng nét. Phụ thuộc độ phân giải → so sánh trên ảnh đã resize cùng cỡ.

    method: "laplacian" (phương sai Laplacian) | "tenengrad" (Sobel) | "fft" (năng lượng tần số cao).
    """
    gray = to_gray(img).astype(np.float64)
    if method == "laplacian":
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if method == "tenengrad":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx ** 2 + gy ** 2))
    if method == "fft":
        # Tỉ lệ năng lượng phổ nằm ngoài vùng tần số thấp (bán kính 10% cạnh ngắn), bỏ thành phần DC
        h, w = gray.shape
        power = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2
        yy, xx = np.ogrid[:h, :w]
        r = 0.1 * min(h, w)
        low = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= r ** 2
        return float(power[~low].sum() / max(power.sum(), 1e-12) * 1000)
    raise ValueError(f"method không hợp lệ: {method}")


def quad_mask(shape, corners) -> np.ndarray:
    """Mask (255) của tứ giác 4 góc trên ảnh có shape này."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(np.asarray(corners)).astype(np.int32), 255)
    return mask


def glare_mask(img: np.ndarray, v_thresh: int = GLARE_V, s_thresh: int = GLARE_S,
               min_area: int = 50, delta: int = GLARE_DELTA, region: np.ndarray = None,
               min_area_ratio: float = GLARE_MIN_AREA_RATIO) -> np.ndarray:
    """Mask vùng lóa (255 = lóa) từ ảnh BGR: V cao, S thấp trong HSV; bỏ thành phần liên thông nhỏ hơn
    max(min_area, min_area_ratio × diện tích giấy).

    Ngưỡng V thực tế = max(v_thresh, mức giấy + delta), mức giấy = phân vị GLARE_PAPER_PCT của V: giấy trắng của ảnh scan / ảnh phẳng
    (V ≈ 255) không bị tính là lóa, chỉ vùng sáng vượt hẳn mặt giấy mới là lóa.
    region (255 = tờ giấy, ví dụ quad_mask của 4 góc): mức giấy chỉ lấy trong region và lóa chỉ tính
    trong region — tránh nền tối quanh giấy (viền màn hình, mặt bàn) kéo mức giấy xuống làm báo lóa giả.
    Ngưỡng diện tích tương đối: lóa thật là một khối liền; bảng có hàng nền trắng xen hàng nền xám cho hàng
    trăm mảnh trắng nhỏ giữa các chữ — trước đây bị cộng dồn thành "lóa nhẹ" (ảnh bộ 3 report1, 02/11).
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    inside = np.ones(v.shape, bool) if region is None else region > 0
    vals = v[inside] if inside.any() else v.ravel()
    paper_v = float(np.percentile(vals, GLARE_PAPER_PCT))
    v_eff = max(v_thresh, paper_v + delta)
    mask = ((v >= v_eff) & (s <= s_thresh) & inside).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(min_area, min_area_ratio * inside.sum())
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def assess_quality(img: np.ndarray, long_side: int = 1000, corners=None) -> Quality:
    """Resize cạnh dài về long_side rồi tính blur_score + glare_ratio, trả về Quality.

    corners (4×2, tọa độ ảnh gốc): nếu có, đo cả hai chỉ trong tờ giấy; glare_ratio = diện tích lóa /
    diện tích giấy.
    """
    small, scale = resize_long_side(img, long_side)
    region = None if corners is None else quad_mask(small.shape, np.asarray(corners) * scale)
    if region is not None and not region.any():
        region = None
    gray = to_gray(small)
    if region is None:
        blur_var = blur_score(small, "laplacian")
    else:
        lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
        blur_var = float(lap[cv2.erode(region, np.ones((5, 5), np.uint8)) > 0].var())   # bỏ mép giấy
    glare = glare_mask(small, region=region) > 0
    glare_ratio = float(glare.mean() if region is None else glare[region > 0].mean())
    return Quality(blur_var=blur_var, glare_ratio=glare_ratio,
                   is_blurry=blur_var < BLUR_THRESHOLD, has_glare=glare_ratio >= GLARE_LIGHT)
