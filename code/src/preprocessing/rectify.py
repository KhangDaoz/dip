"""Nắn phối cảnh. Thí nghiệm D3."""
import cv2
import numpy as np

from .corners import order_corners

INTERPOLATIONS = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR}


def aspect_ratio_zhang(corners: np.ndarray, shape):
    """Tỉ lệ rộng / cao thật của tờ giấy từ 4 góc trên ảnh (Zhang & He, "Whiteboard scanning", 2007).

    Giả định camera pinhole, điểm chính ở tâm ảnh, pixel vuông; ước lượng tiêu cự f từ 4 góc rồi suy ra
    tỉ lệ. Khi 2 cặp cạnh gần song song (f² ≤ 0, phép chiếu gần affine) dùng tỉ lệ độ dài cạnh.
    Trả về None nếu kết quả vô lý.
    """
    h, w = shape[:2]
    u0, v0 = w / 2.0, h / 2.0
    tl, tr, br, bl = (np.array([x, y, 1.0]) for x, y in np.asarray(corners, float))
    m1, m2, m3, m4 = tl, tr, bl, br
    den2, den3 = np.dot(np.cross(m2, m4), m3), np.dot(np.cross(m3, m4), m2)
    if abs(den2) < 1e-9 or abs(den3) < 1e-9:
        return None
    k2 = np.dot(np.cross(m1, m4), m3) / den2
    k3 = np.dot(np.cross(m1, m4), m2) / den3
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    f2 = None
    if abs(n2[2] * n3[2]) > 1e-12:
        f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 ** 2)
               + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 ** 2)) \
            / (n2[2] * n3[2])
    if f2 is not None and f2 > 0:
        a_inv = np.linalg.inv(np.array([[np.sqrt(f2), 0, u0], [0, np.sqrt(f2), v0], [0, 0, 1]]))
        b = a_inv.T @ a_inv
        ratio = np.sqrt((n2 @ b @ n2) / (n3 @ b @ n3))
    else:
        ratio = np.sqrt((n2[0] ** 2 + n2[1] ** 2) / (n3[0] ** 2 + n3[1] ** 2))
    return float(ratio) if np.isfinite(ratio) and 0.2 < ratio < 5 else None


def target_size(corners: np.ndarray, shape=None):
    """Kích thước ảnh đích (rộng, cao): chiều rộng = cạnh ngang dài nhất, chiều cao theo tỉ lệ Zhang–He;
    không ước lượng được tỉ lệ thì lấy cạnh dọc dài nhất."""
    tl, tr, br, bl = np.asarray(corners, float)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    ratio = aspect_ratio_zhang(corners, shape) if shape is not None else None
    if ratio is not None:
        height = width / ratio
    return max(int(round(width)), 2), max(int(round(height)), 2)


def rectify(img: np.ndarray, corners: np.ndarray, interpolation: str = "bicubic"):
    """getPerspectiveTransform + warpPerspective.

    Kích thước đích: xem target_size(). interpolation: "bicubic" | "bilinear".
    Trả về (warped, H) với H (3×3, float64) biến ảnh gốc → ảnh đã nắn.
    """
    src = order_corners(corners)
    w, h = target_size(src, img.shape)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (w, h), flags=INTERPOLATIONS[interpolation],
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, H.astype(np.float64)
