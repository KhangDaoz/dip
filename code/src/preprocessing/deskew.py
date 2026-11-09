"""H8 chỉnh nghiêng (KeHoach.md mục 5.1). Góc trả về theo độ, dương = ngược chiều kim đồng hồ."""
import numpy as np


def skew_projection(binary: np.ndarray, max_angle: float = 15.0, step: float = 0.25) -> float:
    """Phương án chính: θ* = argmax_θ Var(P_θ), P_θ(y) = Σ_x B_θ(x, y) (➕ C3 · S23)."""
    raise NotImplementedError("skew_projection")


def skew_hough(binary: np.ndarray, max_angle: float = 15.0) -> float:
    """Đối chứng: góc trung vị của các đoạn thẳng Hough."""
    raise NotImplementedError("skew_hough")


def skew_radon(binary: np.ndarray, max_angle: float = 15.0, step: float = 0.25) -> float:
    """Đối chứng: biến đổi Radon."""
    raise NotImplementedError("skew_radon")


def rotation_matrix(angle_deg: float, size: tuple) -> np.ndarray:
    """Ma trận xoay 3×3 (để ghép vào H tổng: H = R · H_persp)."""
    raise NotImplementedError("rotation_matrix")
