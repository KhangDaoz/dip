"""R7 nhị phân hóa (KeHoach.md mục 6.1). Mọi hàm trả về ảnh uint8, mực = 255, nền = 0."""
import numpy as np

METHODS = ("otsu", "isodata", "triangle", "symmetry", "niblack", "sauvola", "wolf")


def otsu(gray: np.ndarray) -> np.ndarray:
    """Phương án chính sau khử bóng: k* = argmax σ_B²(k) (C5 · S65–67)."""
    raise NotImplementedError("otsu")


def isodata(gray: np.ndarray, delta: float = 0.5) -> np.ndarray:
    """Đối chứng: t_{k+1} = (μ1 + μ2) / 2 (C5 · S72–73)."""
    raise NotImplementedError("isodata")


def triangle(gray: np.ndarray) -> np.ndarray:
    """Đối chứng: phương pháp tam giác Zack (C5 · S78)."""
    raise NotImplementedError("triangle")


def symmetry(gray: np.ndarray, p: float = 0.95) -> np.ndarray:
    """Đối chứng: đối xứng nền (C5 · S76)."""
    raise NotImplementedError("symmetry")


def niblack(gray: np.ndarray, window: int = 31, k: float = -0.2) -> np.ndarray:
    """Đối chứng ngưỡng cục bộ (➕ mở rộng C5 · S64)."""
    raise NotImplementedError("niblack")


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2, r: float = 128) -> np.ndarray:
    """Đối chứng ngưỡng cục bộ (➕ mở rộng C5 · S64)."""
    raise NotImplementedError("sauvola")


def wolf(gray: np.ndarray, window: int = 31, k: float = 0.5) -> np.ndarray:
    """Đối chứng ngưỡng cục bộ (➕ mở rộng C5 · S64)."""
    raise NotImplementedError("wolf")


def binarize(gray: np.ndarray, method: str = "otsu", **params) -> np.ndarray:
    """Gọi một trong METHODS theo tên."""
    raise NotImplementedError("binarize")
