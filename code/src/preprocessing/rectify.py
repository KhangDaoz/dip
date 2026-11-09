"""H7 nắn phối cảnh (KeHoach.md mục 5.1; 📘 QĐ · S6, nội suy C2 · S23–24)."""
import cv2
import numpy as np

INTERPOLATION = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}


def homography_from_corners(corners: np.ndarray) -> tuple:
    """Tính H 3×3 từ 4 góc (TL, TR, BR, BL) sang hình chữ nhật đích. Trả về (H, (width, height)).

    Cạnh đích = cạnh dài hơn trong mỗi cặp cạnh đối (giữ độ phân giải, không làm mất chi tiết).
    H giải từ 4 cặp điểm: [x' y' w]ᵀ = H [x y 1]ᵀ (Gonzalez & Woods).
    """
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    tl, tr, br, bl = c
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width, height = max(width, 1), max(height, 1)
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    return cv2.getPerspectiveTransform(c, dst), (width, height)


def warp(img: np.ndarray, H: np.ndarray, size: tuple, interpolation: str = "bicubic") -> np.ndarray:
    """Ánh xạ ngược theo H: mỗi pixel đích lấy giá trị tại H⁻¹·(x, y) trên ảnh nguồn, nội suy giữa các mẫu
    (C2 · S23–24: ảnh số chỉ là các mẫu rời rạc). interpolation ∈ {"bicubic", "bilinear"} (đối chứng)."""
    return cv2.warpPerspective(img, np.asarray(H, dtype=np.float64), tuple(int(v) for v in size),
                               flags=INTERPOLATION[interpolation], borderMode=cv2.BORDER_REPLICATE)
