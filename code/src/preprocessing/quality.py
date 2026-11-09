"""H3 đo mờ, H4 đo lóa (KeHoach.md mục 5.1)."""
import cv2
import numpy as np

# Nhân Laplacian 4-lân cận đúng như C5 · S29: ∇²f = f(x+1,y)+f(x−1,y)+f(x,y+1)+f(x,y−1) − 4f(x,y)
LAPLACIAN_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blur_laplacian(gray: np.ndarray) -> float:
    """H3 — phương án chính: B = Var(∇²f), nhân Laplacian 4-lân cận (C5 · S28–29). Càng lớn càng nét."""
    lap = cv2.filter2D(_gray(gray).astype(np.float64), -1, LAPLACIAN_4)
    return float(lap.var())


def blur_tenengrad(gray: np.ndarray) -> float:
    """H3 — đối chứng: mean(Gx² + Gy²) với Sobel (C5 · S15, S19)."""
    g = _gray(gray).astype(np.float64)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def blur_fft(gray: np.ndarray, radius_ratio: float = 0.1) -> float:
    """H3 — đối chứng: tỉ lệ năng lượng tần số cao (C3 · S8): phần phổ nằm ngoài bán kính radius_ratio·min(H, W)."""
    g = _gray(gray).astype(np.float64)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    total = spec.sum()
    return float(spec[r > radius_ratio * min(h, w)].sum() / total) if total > 0 else 0.0


def glare_mask(img_bgr: np.ndarray, tau_v: int = 240, tau_s: int = 30, min_area: int = 50) -> np.ndarray:
    """H4: mask {V > tau_v} ∩ {S < tau_s} trong HSV, bỏ thành phần liên thông nhỏ (C4 · S39–41). Lóa = 255.

    S thấp = nhiều ánh sáng trắng trộn vào màu (C4 · S39); V cao = rất sáng (C4 · S40).
    """
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > tau_v) & (hsv[:, :, 1] < tau_s)).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def glare_ratio(mask: np.ndarray) -> float:
    """Tỉ lệ diện tích lóa trong ảnh, 0..1."""
    return float((mask > 0).mean()) if mask.size else 0.0
