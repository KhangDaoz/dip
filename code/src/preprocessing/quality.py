"""H3 đo mờ, H4 đo lóa (KeHoach.md mục 5.1)."""
import numpy as np


def blur_laplacian(gray: np.ndarray) -> float:
    """H3 — phương án chính: B = Var(∇²f), nhân Laplacian 4-lân cận (C5 · S28–29). Càng lớn càng nét."""
    raise NotImplementedError("blur_laplacian")


def blur_tenengrad(gray: np.ndarray) -> float:
    """H3 — đối chứng: mean(Gx² + Gy²) với Sobel (C5 · S15, S19)."""
    raise NotImplementedError("blur_tenengrad")


def blur_fft(gray: np.ndarray, radius_ratio: float = 0.1) -> float:
    """H3 — đối chứng: tỉ lệ năng lượng tần số cao (C3 · S8)."""
    raise NotImplementedError("blur_fft")


def glare_mask(img_bgr: np.ndarray, tau_v: int = 240, tau_s: int = 30, min_area: int = 50) -> np.ndarray:
    """H4: mask {V > tau_v} ∩ {S < tau_s} trong HSV, bỏ thành phần liên thông nhỏ (C4 · S39–41). Lóa = 255."""
    raise NotImplementedError("glare_mask")


def glare_ratio(mask: np.ndarray) -> float:
    """Tỉ lệ diện tích lóa trong ảnh, 0..1."""
    raise NotImplementedError("glare_ratio")
