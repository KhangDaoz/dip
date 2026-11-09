"""R1 ước lượng nền, R2 khử bóng (KeHoach.md mục 6.1)."""
import numpy as np


def background_closing(gray: np.ndarray, kernel: int = 81, median_ksize: int = 0) -> np.ndarray:
    """Phương án chính: Closing ảnh xám = lọc Max rồi Min (C3 · S74; C7 · S35, S38), sau đó trung vị nếu median_ksize > 0."""
    raise NotImplementedError("background_closing")


def background_median(gray: np.ndarray, kernel: int = 81) -> np.ndarray:
    """Đối chứng: trung vị cỡ lớn trực tiếp (C3 · S73)."""
    raise NotImplementedError("background_median")


def remove_shadow(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """R2: toán tử điểm I_clean = min(255, I / B̂ · 255) (C3 · S10–11)."""
    raise NotImplementedError("remove_shadow")
