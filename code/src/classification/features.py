"""Đặc trưng thủ công cho phân loại tài liệu (KeHoach_Hieu.md mục 3.5). Đầu vào: ảnh xám đã tiền xử lý."""
import numpy as np


def hog_features(gray: np.ndarray, size: int = 256, orientations: int = 9,
                 cell: int = 16, block: int = 2) -> np.ndarray:
    """HOG trên ảnh resize size×size."""
    raise NotImplementedError("hog_features")


def lbp_hist(gray: np.ndarray, P: int = 8, R: int = 1, grid: int = 2) -> np.ndarray:
    """LBP nri_uniform (59 bin): histogram toàn ảnh + lưới grid×grid."""
    raise NotImplementedError("lbp_hist")


def layout_density(binary: np.ndarray, grids=(3, 6)) -> np.ndarray:
    """Tỉ lệ pixel mực trên lưới 3×3 và 6×6."""
    raise NotImplementedError("layout_density")


def line_ratio(binary: np.ndarray, length: int = 40) -> np.ndarray:
    """Tỉ lệ pixel đường kẻ ngang / dọc (opening với SE 1×length và length×1)."""
    raise NotImplementedError("line_ratio")


def text_block_stats(binary: np.ndarray) -> np.ndarray:
    """Số dòng (projection), độ rộng dòng trung bình, tỉ lệ vùng trắng, ..."""
    raise NotImplementedError("text_block_stats")


def extract_all(gray: np.ndarray, groups=("hog", "lbp", "layout", "lines", "text")) -> np.ndarray:
    """Nối các nhóm đặc trưng được chọn (dùng cho ablation)."""
    raise NotImplementedError("extract_all")
