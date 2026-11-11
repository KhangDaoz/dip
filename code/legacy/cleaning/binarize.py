"""R7 nhị phân hóa (KeHoach.md mục 6.1). Mọi hàm trả về ảnh uint8, mực = 255, nền = 0.

Phương án chính: Sauvola (ngưỡng cục bộ). Đối chứng: Otsu, ISODATA, Tam giác, Đối xứng nền (C5 · S65–78),
Niblack, Wolf.
"""
import cv2
import numpy as np

METHODS = ("sauvola", "otsu", "isodata", "triangle", "symmetry", "niblack", "wolf")


def _ink(gray: np.ndarray, T) -> np.ndarray:
    """Mực tối hơn ngưỡng → 255."""
    return np.where(gray.astype(np.float64) <= T, 255, 0).astype(np.uint8)


def _hist(gray):
    return np.bincount(gray.ravel(), minlength=256).astype(np.float64)


def otsu_threshold(gray: np.ndarray) -> int:
    """k* = argmax σ_B²(k), σ_B²(k) = [m_G P₁(k) − m(k)]² / (P₁(k)[1 − P₁(k)]) (C5 · S65–67)."""
    p = _hist(gray) / gray.size
    P1 = np.cumsum(p)
    m = np.cumsum(np.arange(256) * p)
    mG = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mG * P1 - m) ** 2 / (P1 * (1 - P1))
    sb = np.nan_to_num(sb)
    best = np.flatnonzero(sb == sb.max())
    return int(round(best.mean()))   # cực đại không duy nhất → lấy trung bình (C5 · S67 bước 6)


def otsu(gray: np.ndarray) -> np.ndarray:
    """Đối chứng chính: ngưỡng toàn cục tối ưu Otsu (C5 · S65–67)."""
    return _ink(gray, otsu_threshold(gray))


def isodata(gray: np.ndarray, delta: float = 0.5) -> np.ndarray:
    """Đối chứng: t_{k+1} = (μ1(t_k) + μ2(t_k)) / 2, lặp tới |t_{k+1} − t_k| < ΔT (C5 · S72–73)."""
    p = _hist(gray)
    z = np.arange(256)
    t = float(gray.mean())
    for _ in range(100):
        lo, hi = z <= t, z > t
        mu1 = (z[lo] * p[lo]).sum() / max(p[lo].sum(), 1)
        mu2 = (z[hi] * p[hi]).sum() / max(p[hi].sum(), 1)
        t_new = (mu1 + mu2) / 2
        if abs(t_new - t) < delta:
            t = t_new
            break
        t = t_new
    return _ink(gray, t)


def triangle(gray: np.ndarray) -> np.ndarray:
    """Đối chứng: tam giác Zack (C5 · S78): nối đỉnh lược đồ với đầu kia, ngưỡng tại b₀ có khoảng cách d lớn nhất.

    Tài liệu: đỉnh là nền sáng, đuôi dài về phía tối (mực) → tìm b₀ phía tối của đỉnh.
    """
    h = _hist(gray)
    b_max = int(np.argmax(h))
    nz = np.flatnonzero(h)
    b_min = int(nz[0])
    if b_max <= b_min:
        return otsu(gray)
    b = np.arange(b_min, b_max + 1)
    x1, y1, x2, y2 = b_min, h[b_min], b_max, h[b_max]
    d = np.abs((y2 - y1) * b - (x2 - x1) * h[b] + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)
    return _ink(gray, int(b[np.argmax(d)]))


def symmetry(gray: np.ndarray, p: float = 0.95) -> np.ndarray:
    """Đối chứng: đối xứng nền (C5 · S76). Nền là đỉnh maxp, đối xứng quanh đỉnh; tìm điểm a phía sáng sao cho
    P(a) = p, rồi lấy ngưỡng đối xứng T = maxp − (a − maxp)."""
    h = _hist(gray)
    maxp = int(np.argmax(h))
    cdf = np.cumsum(h) / h.sum()
    a = int(np.searchsorted(cdf, p))
    return _ink(gray, maxp - max(a - maxp, 1))


def _local_stats(gray: np.ndarray, window: int):
    """Trung bình m và độ lệch chuẩn s trong cửa sổ w × w (lọc hộp, C3 · S50)."""
    g = gray.astype(np.float64)
    w = (int(window) | 1, int(window) | 1)
    m = cv2.boxFilter(g, -1, w, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(g * g, -1, w, borderType=cv2.BORDER_REFLECT)
    return m, np.sqrt(np.maximum(m2 - m * m, 0))


def niblack(gray: np.ndarray, window: int = 31, k: float = -0.2) -> np.ndarray:
    """Đối chứng ngưỡng cục bộ: T = m + k·s (➕ mở rộng C5 · S64)."""
    m, s = _local_stats(gray, window)
    return _ink(gray, m + k * s)


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2, r: float = 128) -> np.ndarray:
    """PHƯƠNG ÁN CHÍNH: T(x, y) = m · [1 + k (s / R − 1)] (➕ mở rộng ngưỡng toàn cục C5 · S64 sang từng cửa sổ).

    Vùng nền trơn (s nhỏ) → T thấp hơn m nhiều, nền giữ trắng; vùng có chữ (s lớn) → T gần m.
    """
    m, s = _local_stats(gray, window)
    return _ink(gray, m * (1 + k * (s / r - 1)))


def wolf(gray: np.ndarray, window: int = 31, k: float = 0.5) -> np.ndarray:
    """Đối chứng ngưỡng cục bộ Wolf–Jolion: T = (1 − k) m + k M + k (s / R)(m − M), M = min ảnh, R = max s."""
    m, s = _local_stats(gray, window)
    M, R = float(gray.min()), max(float(s.max()), 1e-6)
    return _ink(gray, (1 - k) * m + k * M + k * (s / R) * (m - M))


def binarize(gray: np.ndarray, method: str = "sauvola", **params) -> np.ndarray:
    """Gọi một trong METHODS theo tên."""
    fn = {"sauvola": sauvola, "otsu": otsu, "isodata": isodata, "triangle": triangle,
          "symmetry": symmetry, "niblack": niblack, "wolf": wolf}[method]
    return fn(gray, **params)
