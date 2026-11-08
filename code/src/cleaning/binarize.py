"""5 cách nhị phân hóa. Thí nghiệm T2, T3.

Mọi hàm nhận ảnh xám uint8, trả về ảnh nhị phân uint8 cùng shape với **mực = 255, nền = 0**.
"""
import numpy as np
import cv2
from numbers import Real
from skimage.filters import threshold_sauvola


def _check_gray(gray):
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or gray.size == 0:
        raise ValueError("Expected a nonempty 2D uint8 grayscale image")


def _check_window(window, name="window"):
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 3 or window % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= 3")


def _check_finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


def _local_mean_std(gray, window):
    """Population statistics, square window, reflected border without edge repeat.

    Float64 moments prevent uint8 squaring overflow. Unnormalized sums followed
    by division keep constant integer-valued windows exact (variance zero).
    """
    values = gray.astype(np.float64)
    size = (int(window), int(window))
    area = int(window) ** 2
    mean = cv2.boxFilter(values, cv2.CV_64F, size, normalize=False, borderType=cv2.BORDER_REFLECT_101) / area
    square_mean = cv2.boxFilter(values * values, cv2.CV_64F, size, normalize=False,
                               borderType=cv2.BORDER_REFLECT_101) / area
    std = np.sqrt(np.maximum(square_mean - mean * mean, 0.0))
    return mean, std


def otsu(gray: np.ndarray) -> np.ndarray:
    _check_gray(gray)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def niblack(gray: np.ndarray, window: int = 31, k: float = -0.2) -> np.ndarray:
    """T = mean + k*std; mực khi gray < T, k mặc định âm (-0.2).

    k có dấu ngược với skimage.threshold_niblack (T = mean - k*std).
    Thống kê dùng cửa sổ vuông, biên phản chiếu BORDER_REFLECT_101.
    """
    _check_gray(gray)
    _check_window(window)
    _check_finite(k, "k")
    mean, std = _local_mean_std(gray, window)
    return np.where(gray < mean + k * std, 255, 0).astype(np.uint8)


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2) -> np.ndarray:
    _check_gray(gray)
    _check_window(window)
    _check_finite(k, "k")
    # R=128 is explicit for 8-bit images; never infer it from this image's contrast.
    threshold = threshold_sauvola(gray, window_size=window, k=k, r=128)
    return np.where(gray < threshold, 255, 0).astype(np.uint8)


def wolf(gray: np.ndarray, window: int = 31, k: float = 0.5) -> np.ndarray:
    """Wolf-Jolion: T = mean + k*(std/R - 1)*(mean - M).

    R=max(std) và M=min(gray) trên toàn ảnh, không cố định R=128 như Sauvola.
    Mực khi gray < T; biên phản chiếu như Niblack. Nếu R=0, ảnh không có
    tương phản: dùng T=mean (không tạo mực trên ảnh hằng, không chia cho 0).
    Nguồn: https://chriswolfvision.github.io/www/papers/tr-rfv-2002-01.pdf, Eq. 29.
    """
    _check_gray(gray)
    _check_window(window)
    _check_finite(k, "k")
    mean, std = _local_mean_std(gray, window)
    maximum_std = float(std.max())
    if maximum_std == 0:
        threshold = mean
    else:
        threshold = mean + k * (std / maximum_std - 1.0) * (mean - float(gray.min()))
    return np.where(gray < threshold, 255, 0).astype(np.uint8)


def gaussian_adaptive(gray: np.ndarray, block: int = 31, c: int = 10) -> np.ndarray:
    """OpenCV Gaussian adaptive, THRESH_BINARY_INV (mực=255, nền=0).

    T là trung bình Gaussian trừ c; OpenCV dùng biên lặp và làm tròn ngưỡng.
    Giữ nguyên quy tắc bằng ngưỡng của OpenCV, không ép giống Niblack/Wolf.
    block lẻ >=3; c hữu hạn, có thể bằng 0 hoặc âm theo API OpenCV.
    """
    _check_gray(gray)
    _check_window(block, "block")
    _check_finite(c, "c")
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, int(block), float(c))


METHODS = {"otsu": otsu, "niblack": niblack, "sauvola": sauvola, "wolf": wolf, "gaussian": gaussian_adaptive}
