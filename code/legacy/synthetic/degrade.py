"""Các phép làm hỏng có kiểm soát của bộ 5 (KeHoach.md mục 8.2).

Mỗi hàm nhận ảnh sạch BGR + mức (level) + rng, trả về (ảnh hỏng, ground truth dạng dict).
Bộ 5 v2 chính thức (1.500 mẫu) đã sinh và đưa lên Drive; module này để tái lập và giải thích mô hình suy biến.
"""
import cv2
import numpy as np

PAPER_BORDER = 255


def _paper_color(img):
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return tuple(int(v) for v in np.median(edge.reshape(-1, img.shape[2] if img.ndim == 3 else 1), axis=0))


def perspective(img: np.ndarray, level: float, rng: np.random.Generator) -> tuple:
    """Dịch ngẫu nhiên 4 góc level × cạnh; H 3×3; nội suy cubic cho ảnh, nearest cho mask. GT: H, 4 góc (📘 QĐ · S6).

    Ảnh đặt trên nền bàn tối, khung ảnh rộng hơn để giữ đủ 4 góc giấy.
    """
    h, w = img.shape[:2]
    m = int(level * max(h, w)) + 20
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    jitter = rng.uniform(-level, level, (4, 2)) * np.array([w, h])
    dst = (src + jitter + m).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    size = (w + 2 * m, h + 2 * m)
    bg = np.full((size[1], size[0], 3), 60, np.uint8)
    warped = cv2.warpPerspective(img, H, size, flags=cv2.INTER_CUBIC)
    page = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, size, flags=cv2.INTER_NEAREST)
    bg[page > 0] = warped[page > 0]
    return bg, {"H": H.tolist(), "corners": dst.tolist(), "page_mask": page}


def rotate(img: np.ndarray, angle_deg: float) -> tuple:
    """Ma trận xoay quanh tâm, giữ kích thước, vùng trống tô màu giấy. GT: góc thật (dương = ngược chiều kim đồng hồ)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=_paper_color(img))
    return out, {"angle_deg": float(angle_deg), "H": np.vstack([M, [0, 0, 1]]).tolist()}


def gaussian_blur(img: np.ndarray, level: int) -> tuple:
    """Tích chập nhân Gaussian, 4 mức (C3 · S47–53): σ = 0,8·level, cửa sổ lẻ ≈ 6σ."""
    sigma = 0.8 * level
    k = int(6 * sigma) | 1
    return cv2.GaussianBlur(img, (k, k), sigma), {"blur_kind": "gaussian", "sigma": sigma}


def motion_blur(img: np.ndarray, level: int, rng: np.random.Generator) -> tuple:
    """Tích chập nhân đường thẳng, 4 mức (C3 · S47–48): g(x,y) = Σ w(s,t) f(x+s, y+t), w là đoạn thẳng dài 4·level+1."""
    length = 4 * level + 1
    angle = float(rng.uniform(0, 180))
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1), (length, length))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k), {"blur_kind": "motion", "length": length, "angle": angle}


def shadow(img: np.ndarray, level: int, rng: np.random.Generator) -> tuple:
    """I' = I · M, M là mặt nạ chiếu sáng (gradient, bàn tay làm mờ), 3 mức (C3 · S10–11). GT: mask bóng."""
    h, w = img.shape[:2]
    strength = (0.25, 0.45, 0.65)[level - 1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
    grad = (grad - grad.min()) / max(np.ptp(grad), 1e-6)
    blob = np.zeros((h, w), np.float32)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    cv2.ellipse(blob, (int(cx), int(cy)), (int(w * rng.uniform(0.15, 0.3)), int(h * rng.uniform(0.2, 0.4))),
                float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    blob = cv2.GaussianBlur(blob, (0, 0), max(w, h) * 0.05)
    dark = np.clip(0.6 * grad + blob, 0, 1)
    M = 1 - strength * dark
    out = np.clip(img.astype(np.float32) * M[..., None], 0, 255).astype(np.uint8)
    return out, {"shadow_level": level, "shadow_mask": ((1 - M) > 0.1 * strength).astype(np.uint8) * 255}


def glare(img: np.ndarray, level: int, rng: np.random.Generator, exposure_range=(0.80, 0.92)) -> tuple:
    """Giảm phơi sáng trước, rồi tăng V / giảm S trong elip Gaussian, 3 mức (C4 · S39–41). GT: mask lóa, ảnh sạch đã giảm phơi sáng."""
    gain = float(rng.uniform(*exposure_range))
    exposed = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)   # toán tử điểm (C3 · S10)
    h, w = img.shape[:2]
    area = (0.01, 0.04, 0.12)[level - 1]
    blob = np.zeros((h, w), np.float32)
    ax = int(np.sqrt(area * h * w / np.pi) * rng.uniform(0.8, 1.3))
    ay = int(area * h * w / np.pi / max(ax, 1))
    cv2.ellipse(blob, (int(rng.uniform(0.2, 0.8) * w), int(rng.uniform(0.2, 0.8) * h)), (ax, ay),
                float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    blob = cv2.GaussianBlur(blob, (0, 0), max(ax, ay) * 0.3 + 1)
    hsv = cv2.cvtColor(exposed, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + blob * 255, 0, 255)      # V tăng: sáng hơn (C4 · S40)
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - blob)                        # S giảm: trộn ánh sáng trắng (C4 · S39)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out, {"glare_level": level, "exposure_gain": gain, "glare_mask": (blob > 0.5).astype(np.uint8) * 255,
                 "clean_exposed": exposed}


def stamp_pen(img: np.ndarray, level: int, rng: np.random.Generator, kind: str = "stamp") -> tuple:
    """Dấu đỏ / bút xanh / bút đen trộn trong suốt, 3 mức che (C4 · S29–31, S38). GT: mask dấu / bút."""
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.uint8)
    if kind == "stamp":
        r = int(min(h, w) * (0.08, 0.11, 0.14)[level - 1])
        cx, cy = int(rng.uniform(r, w - r)), int(rng.uniform(r, h - r))
        cv2.circle(layer, (cx, cy), r, 255, max(2, r // 12))
        cv2.circle(layer, (cx, cy), int(r * 0.7), 255, max(1, r // 20))
        cv2.putText(layer, "DA KHAM", (cx - int(r * 0.55), cy + 5), cv2.FONT_HERSHEY_SIMPLEX, r / 90, 255, 2)
        color = (40, 40, 200)
    else:
        pts = np.cumsum(rng.normal(0, w * 0.03, (12, 2)), axis=0) + rng.uniform(0.3, 0.7, 2) * [w, h]
        cv2.polylines(layer, [pts.astype(np.int32)], False, 255, 2 + level, cv2.LINE_AA)
        color = (170, 60, 20) if kind == "pen_blue" else (30, 30, 30)
    alpha = (layer.astype(np.float32) / 255 * 0.85)[..., None]
    out = (img.astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)
    key = "stamp_mask" if kind == "stamp" else "pen_mask"
    return out, {key: (layer > 0).astype(np.uint8) * 255, "kind": kind, "level": level}


def jpeg(img: np.ndarray, quality: int) -> tuple:
    """Nén JPEG chất lượng 30–90, 3 mức (C4 · S15, S18; C6 · S40–54): lượng tử hóa DCT khối 8×8 làm mất tần số cao."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), {"jpeg_quality": int(quality)}


def gaussian_noise(img: np.ndarray, level: int, rng: np.random.Generator) -> tuple:
    """g = f + η, η ~ N(0, σ²), σ = 4·level (C3 · S65–66)."""
    sigma = 4.0 * level
    out = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return out, {"noise_sigma": sigma}


def clean_binary(img: np.ndarray) -> np.ndarray:
    """Ảnh nhị phân chuẩn = Otsu trên ảnh sạch (C5 · S65–67). Mực = 255."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def ignore_band(binary_gt: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Dải viền bỏ qua khi chấm: (A ⊕ B) − (A ⊖ B), B = ksize × ksize (mở rộng tách biên C7 · S40). Vùng bỏ qua = 255.

    Trên ảnh sạch các phương pháp nhị phân chỉ khác nhau ở viền nét (biên dốc khi scan), nên không chấm dải này để
    đáp án tạo bằng Otsu không ưu ái Otsu (KeHoach.md mục 6.4, 8.2)."""
    a = (binary_gt > 0).astype(np.uint8)
    se = np.ones((ksize, ksize), np.uint8)
    return ((cv2.dilate(a, se) - cv2.erode(a, se)) > 0).astype(np.uint8) * 255
