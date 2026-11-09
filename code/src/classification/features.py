"""H10 trích đặc trưng bố cục (KeHoach.md mục 5.1). Đầu vào: ảnh xám đã tiền xử lý."""
import numpy as np


def hog_features(gray: np.ndarray, size: int = 256) -> np.ndarray:
    """HOG (C8 · S12, S17)."""
    raise NotImplementedError("hog_features")


def glcm_haralick(gray: np.ndarray, distances=(1,), angles=(0, 45, 90, 135)) -> np.ndarray:
    """GLCM + năng lượng, entropy, tương phản, đồng nhất (C8 · S14–15)."""
    raise NotImplementedError("glcm_haralick")


def ink_density_grid(binary: np.ndarray, grids=(3, 6)) -> np.ndarray:
    """Mật độ mực trên lưới 3×3 và 6×6."""
    raise NotImplementedError("ink_density_grid")


def line_ratio(binary: np.ndarray, length: int = 40) -> np.ndarray:
    """Tỉ lệ đường kẻ ngang / dọc sau Opening SE 1×40 và 40×1 (C7 · S28)."""
    raise NotImplementedError("line_ratio")


def color_histogram(img_bgr: np.ndarray, bins: int = 16) -> np.ndarray:
    """Đối chứng: lược đồ màu chuẩn hóa (C8 · S13)."""
    raise NotImplementedError("color_histogram")


def extract_all(gray: np.ndarray) -> np.ndarray:
    """Ghép các đặc trưng chính thành một vector."""
    raise NotImplementedError("extract_all")
