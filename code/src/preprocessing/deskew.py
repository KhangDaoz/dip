"""Chỉnh nghiêng, 3 phương án. Thí nghiệm D4.

Mọi hàm deskew_* trả về (ảnh đã chỉnh, góc_độ) với góc_độ là góc nghiêng phát hiện được,
cùng quy ước dấu với rotate(): rotate(ảnh_thẳng, a) cho ảnh nghiêng a độ → deskew_* trả về ≈ a,
và ảnh đã chỉnh = rotate(ảnh, -a).

Cả 3 cách đo trên ảnh mực chỉ gồm thành phần cỡ ký tự (corners.char_mask) — Otsu toàn ảnh nhận
nhầm bóng, nền, nếp gập thành mực và kéo góc đo lệch hẳn — đã thu nhỏ cạnh dài về WORK_SIDE để chạy
nhanh; góc không phụ thuộc tỉ lệ thu nhỏ. Góc đo được chạm biên ±angle_range coi là không tin cậy → 0.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.transform import radon

from common.io import to_gray
from .corners import char_centroids, char_mask
from .quality import resize_long_side

WORK_SIDE = 1000      # cạnh dài khi đo góc (projection, Hough)
RADON_SIDE = 400      # Radon chậm hơn nhiều → đo trên ảnh nhỏ hơn
VERTICAL_MAX = 0.45   # < 45% ký tự có láng giềng gần nhất theo phương ngang → dòng chữ chạy dọc
ANGLE_RANGE = 20.0    # khoảng dò ±20°: góc chạm biên bị coi là không tin cậy, nên phải rộng hơn góc cần sửa (≤ 15°, D4)


def border_color(img: np.ndarray):
    """Màu trung vị của viền ảnh (dải 2% mỗi cạnh) — dùng tô vùng trống khi xoay, gần màu giấy."""
    h, w = img.shape[:2]
    b = max(1, int(0.02 * min(h, w)))
    strips = [img[:b].reshape(-1, *img.shape[2:]), img[-b:].reshape(-1, *img.shape[2:]),
              img[:, :b].reshape(-1, *img.shape[2:]), img[:, -b:].reshape(-1, *img.shape[2:])]
    med = np.median(np.concatenate(strips), axis=0)
    return tuple(float(v) for v in np.atleast_1d(med))


def _rotation_matrix(shape, angle_deg: float, expand: bool = False):
    """Ma trận 2×3 xoay quanh tâm; expand=True dịch thêm để ảnh xoay nằm trọn trong khung mới.
    Trả về (M, (rộng, cao) ảnh ra)."""
    h, w = shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    if not expand:
        return M, (w, h)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(np.ceil(w * c + h * s)), int(np.ceil(w * s + h * c))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return M, (nw, nh)


def rotate(img: np.ndarray, angle_deg: float, border=None, expand: bool = False) -> np.ndarray:
    """Xoay quanh tâm ngược chiều kim đồng hồ angle_deg độ.

    expand=False giữ nguyên kích thước (4 góc nội dung bị cắt khi góc lớn); expand=True nới khung để giữ
    trọn nội dung — pipeline dùng expand=True (ảnh thật nghiêng 9° từng mất ~1/3 số chữ, review 02/11).
    Vùng trống tô bằng border (mặc định: màu trung vị viền ảnh, tức màu giấy) — không viền đen.
    """
    if angle_deg == 0:
        return img.copy()
    value = border_color(img) if border is None else border
    M, size = _rotation_matrix(img.shape, angle_deg, expand)
    return cv2.warpAffine(img, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=value)


def rotation_homography(shape, angle_deg: float, expand: bool = False) -> np.ndarray:
    """Ma trận 3×3 khớp đúng với rotate(img, angle_deg, expand=expand) trên ảnh có shape này.

    Điểm p trên ảnh trước khi xoay → R @ [x, y, 1] là vị trí của nó trên ảnh sau rotate().
    pipeline dùng nó để gộp phép chỉnh nghiêng vào homography: H = R @ H_persp.
    """
    return np.vstack([_rotation_matrix(shape, angle_deg, expand)[0], [0.0, 0.0, 1.0]])


def horizontal_neighbor_ratio(img: np.ndarray) -> float:
    """Tỉ lệ ký tự có láng giềng gần nhất lệch < 45° so với phương ngang.

    Chữ cùng dòng sát nhau hơn khoảng cách giữa hai dòng → trang thẳng cho tỉ lệ cao (≈ 0,6–1),
    trang xoay 90° cho tỉ lệ thấp. Chỉ dùng thành phần cỡ ký tự nên không bị bóng, nền, nếp gập chi phối.
    """
    pts = char_centroids(to_gray(resize_long_side(img, WORK_SIDE)[0]))
    if len(pts) < 10:
        return 1.0                       # quá ít chữ để kết luận → coi như thẳng
    _, idx = cKDTree(pts).query(pts, k=2)
    v = pts[idx[:, 1]] - pts
    return float((np.abs(v[:, 0]) > np.abs(v[:, 1])).mean())


def orientation_turns(img: np.ndarray) -> int:
    """Số lần xoay 90° ngược chiều kim đồng hồ để dòng chữ nằm ngang: 0 hoặc 1.

    Chưa phân biệt 0° với 180° (trang lộn ngược) — xem KeHoach_Hieu, bước 6.
    """
    return 1 if horizontal_neighbor_ratio(img) < VERTICAL_MAX else 0


def rotate90(img: np.ndarray, turns: int):
    """Xoay turns × 90° ngược chiều kim đồng hồ. Trả về (ảnh, ma trận 3×3 ảnh cũ → ảnh mới)."""
    H = np.eye(3)
    for _ in range(turns % 4):
        w = img.shape[1]
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        H = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, w - 1.0], [0.0, 0.0, 1.0]]) @ H   # (x, y) → (y, w-1-x)
    return img, H


def ink_mask(img: np.ndarray, long_side: int = WORK_SIDE) -> np.ndarray:
    """Ảnh mực đã thu nhỏ: chỉ thành phần cỡ ký tự (mực = 255)."""
    small, _ = resize_long_side(to_gray(img), long_side)
    return char_mask(small, min_area=8 if long_side >= WORK_SIDE else 3)


def _trusted(angle: float, angle_range: float, step: float = 0.25) -> float:
    """Góc chạm biên dò thì cực đại thật có thể nằm ngoài khoảng (hoặc không có) → không xoay."""
    return 0.0 if abs(angle) >= angle_range - step / 2 else angle


def _profile_score(ink: np.ndarray, angle_deg: float) -> float:
    """Phương sai profile ngang sau khi xoay ảnh mực về -angle_deg: dòng chữ càng nằm ngang càng lớn."""
    rotated = cv2.warpAffine(ink, _rotation_matrix(ink.shape, -angle_deg)[0], (ink.shape[1], ink.shape[0]),
                             flags=cv2.INTER_NEAREST, borderValue=0)
    return float(rotated.sum(axis=1, dtype=np.float64).var())


def deskew_projection(img: np.ndarray, angle_range: float = ANGLE_RANGE, step: float = 0.25):
    """Xoay thử trong ±angle_range, chọn góc cho phương sai profile ngang lớn nhất.

    Dò thô bước 1° rồi dò mịn bước `step` quanh góc tốt nhất.
    """
    ink = ink_mask(img)
    coarse = np.arange(-angle_range, angle_range + 1e-9, 1.0)
    best = coarse[np.argmax([_profile_score(ink, a) for a in coarse])]
    fine = np.arange(max(-angle_range, best - 1), min(angle_range, best + 1) + 1e-9, step)
    angle = _trusted(float(fine[np.argmax([_profile_score(ink, a) for a in fine])]), angle_range, step)
    return rotate(img, -angle), angle


def deskew_hough(img: np.ndarray, angle_range: float = ANGLE_RANGE):
    """Hough lines trên ảnh nhị phân, lấy góc trung vị.

    Giãn ngang ảnh mực để mỗi dòng chữ thành một vệt liền, HoughLinesP trên biên các vệt,
    giữ đoạn lệch ≤ angle_range so với phương ngang, lấy trung vị có trọng số theo độ dài.
    """
    ink = ink_mask(img)
    h, w = ink.shape
    blobs = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 40), 3)))
    edges = cv2.Canny(blobs, 50, 150)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=80, minLineLength=w // 5, maxLineGap=10)
    if segs is None:
        return img.copy(), 0.0
    x1, y1, x2, y2 = segs.reshape(-1, 4).T.astype(float)
    angles = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))       # y hướng xuống → đổi dấu để CCW dương
    angles = (angles + 90) % 180 - 90                            # đoạn vẽ ngược chiều → cùng một góc
    lengths = np.hypot(x2 - x1, y2 - y1)
    keep = np.abs(angles) <= angle_range
    if not keep.any():
        return img.copy(), 0.0
    order = np.argsort(angles[keep])
    a, wts = angles[keep][order], lengths[keep][order]
    angle = _trusted(float(a[np.searchsorted(np.cumsum(wts), wts.sum() / 2)]), angle_range)
    return rotate(img, -angle), angle


def deskew_radon(img: np.ndarray, angle_range: float = ANGLE_RANGE, step: float = 0.25):
    """Radon transform (skimage.transform.radon), chọn góc có phương sai lớn nhất.

    Phép chiếu ở theta = 90° cộng dọc theo hàng (profile ngang); dòng chữ nghiêng a độ cho
    phương sai lớn nhất tại theta = 90° + a. Dò thô bước 1° rồi dò mịn bước `step`.
    """
    ink = ink_mask(img, RADON_SIDE).astype(np.float64) / 255.0

    def best(angles):
        sinogram = radon(ink, theta=90.0 + angles, circle=False)
        return float(angles[np.argmax(sinogram.var(axis=0))])

    coarse = best(np.arange(-angle_range, angle_range + 1e-9, 1.0))
    angle = _trusted(best(np.arange(max(-angle_range, coarse - 1), min(angle_range, coarse + 1) + 1e-9, step)),
                     angle_range, step)
    return rotate(img, -angle), angle
