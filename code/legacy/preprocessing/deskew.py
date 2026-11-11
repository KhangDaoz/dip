"""H8 chỉnh nghiêng (KeHoach.md mục 5.1). Góc trả về theo độ, dương = nội dung bị xoay ngược chiều kim đồng hồ.

Muốn làm thẳng: xoay ảnh một góc −θ (hàm rotation_matrix làm việc này).
"""
import cv2
import numpy as np

WORK_SIDE = 800   # thu nhỏ ảnh nhị phân khi dò góc để chạy nhanh; góc không đổi khi co giãn đều


def _shrink(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    s = WORK_SIDE / max(h, w)
    if s >= 1:
        return binary
    return cv2.resize(binary, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _rotate(binary: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)   # góc dương = xoay ngược chiều kim đồng hồ
    return cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def _profile_score(binary: np.ndarray, theta: float) -> float:
    """Var(P_θ) với P_θ(y) = Σ_x B_θ(x, y): ảnh xoay −θ, rồi đếm pixel mực theo hàng (➕ lược đồ C3 · S23)."""
    return float(_rotate(binary, -theta).sum(axis=1, dtype=np.float64).var())


def skew_projection(binary: np.ndarray, max_angle: float = 15.0, step: float = 0.25) -> float:
    """Phương án chính: θ* = argmax_θ Var(P_θ), P_θ(y) = Σ_x B_θ(x, y) (➕ C3 · S23).

    Khi dòng chữ nằm ngang, mỗi hàng hoặc đầy mực hoặc trống, nên profile có đỉnh / thung lũng rõ nhất.
    Tìm thô bước 1°, rồi tinh bước `step` quanh góc tốt nhất.
    """
    b = (_shrink(binary) > 0).astype(np.uint8)
    coarse = np.arange(-max_angle, max_angle + 1e-9, 1.0)
    best = max(coarse, key=lambda t: _profile_score(b, t))
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, step)
    return float(max(fine, key=lambda t: _profile_score(b, t)))


def skew_hough(binary: np.ndarray, max_angle: float = 15.0) -> float:
    """Đối chứng: góc trung vị của các đoạn thẳng Hough trên ảnh đã Dilation ngang (chữ cùng dòng dính thành vệt)."""
    b = _shrink(binary)
    h, w = b.shape
    smear = cv2.dilate(b, np.ones((1, max(3, w // 60)), np.uint8))
    edges = cv2.Canny(smear, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=60, minLineLength=w // 6, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):   # OpenCV 4: (N,1,4), OpenCV 5: (N,4)
        a = -np.degrees(np.arctan2(y2 - y1, x2 - x1))   # trục y ảnh hướng xuống → đổi dấu để dương = ngược chiều kim đồng hồ
        if abs(a) <= max_angle:
            angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def skew_radon(binary: np.ndarray, max_angle: float = 15.0, step: float = 0.25) -> float:
    """Đối chứng: biến đổi Radon — tích phân ảnh theo các hướng θ, chọn hướng có phương sai hình chiếu lớn nhất."""
    from skimage.transform import radon
    b = _shrink(binary).astype(np.float64) / 255.0
    h, w = b.shape
    side = int(np.ceil(np.hypot(h, w)))
    pad = np.zeros((side, side))
    y0, x0 = (side - h) // 2, (side - w) // 2
    pad[y0:y0 + h, x0:x0 + w] = b
    thetas = np.arange(-max_angle, max_angle + 1e-9, step)
    # theta = 90° trong skimage chiếu dọc theo hàng (tổng theo x) → profile ngang của ảnh thẳng
    sino = radon(pad, theta=90.0 + thetas, circle=False)
    return float(thetas[int(np.argmax(sino.var(axis=0)))])


def rotation_matrix(angle_deg: float, size: tuple) -> np.ndarray:
    """Ma trận 3×3 LÀM THẲNG ảnh nghiêng angle_deg (xoay −angle_deg quanh tâm), size = (width, height).

    Ghép vào H tổng: H = R · H_persp.
    """
    w, h = size
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle_deg, 1.0)
    return np.vstack([M, [0.0, 0.0, 1.0]])
