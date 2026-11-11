"""A1 nắn phối cảnh (ch. 5: Canny; ch. 2: nội suy khi ánh xạ ngược).

Tìm 4 góc tờ giấy: contour → Hough → hình chữ nhật bao nhỏ nhất → cả ảnh; rồi warpPerspective.
Góc trả về theo thứ tự TL, TR, BR, BL (mảng 4×2 float).
"""
from __future__ import annotations

import cv2
import numpy as np

from common.io import to_gray

MIN_AREA_RATIO = 0.2   # tứ giác phải chiếm >= 20 % ảnh mới coi là tờ giấy
INTERPOLATION = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}


def order_corners(pts) -> np.ndarray:
    """Sắp 4 điểm theo TL, TR, BR, BL: TL có x+y nhỏ nhất, BR lớn nhất; TR có y−x nhỏ nhất, BL lớn nhất."""
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s, d = p.sum(1), p[:, 1] - p[:, 0]
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]])


def _valid_quad(q, shape) -> bool:
    if q is None:
        return False
    q = np.asarray(q, dtype=np.float32)
    area = cv2.contourArea(q.reshape(-1, 1, 2))
    return cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.int32)) and area >= MIN_AREA_RATIO * shape[0] * shape[1]


def _edges(gray, canny_low, canny_high):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)          # Canny bước 1: làm mịn Gaussian (C5 · S40)
    edges = cv2.Canny(blur, canny_low, canny_high)    # bước 2–4: gradient, NMS, ngưỡng kép (C5 · S41–47)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8))  # nối chỗ biên đứt (C7 · S24)


def corners_contour(gray: np.ndarray, canny_low: int = 50, canny_high: int = 150, approx_eps: float = 0.02):
    """Phương án chính: Canny → đường bao lớn nhất → xấp xỉ tứ giác."""
    edges = _edges(gray, canny_low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        hull = cv2.convexHull(cnt)
        approx = cv2.approxPolyDP(hull, approx_eps * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            q = order_corners(approx.reshape(4, 2))
            if _valid_quad(q, gray.shape):
                return q
    return None


def _intersect(l1, l2):
    (x1, y1, x2, y2), (x3, y3, x4, y4) = l1, l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return px, py


def corners_hough(gray: np.ndarray, canny_low: int = 50, canny_high: int = 150):
    """Dự phòng 1: đường thẳng Hough → 4 cạnh ngoài cùng (trên, dưới, trái, phải) → giao điểm."""
    h, w = gray.shape
    edges = _edges(gray, canny_low, canny_high)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min(h, w) // 4, maxLineGap=20)
    if lines is None:
        return None
    horiz, vert = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):   # OpenCV 4: (N,1,4), OpenCV 5: (N,4)
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        (horiz if ang < 30 or ang > 150 else vert if 60 < ang < 120 else []).append((x1, y1, x2, y2))
    if len(horiz) < 2 or len(vert) < 2:
        return None
    top = min(horiz, key=lambda l: (l[1] + l[3]) / 2)
    bottom = max(horiz, key=lambda l: (l[1] + l[3]) / 2)
    left = min(vert, key=lambda l: (l[0] + l[2]) / 2)
    right = max(vert, key=lambda l: (l[0] + l[2]) / 2)
    pts = [_intersect(top, left), _intersect(top, right), _intersect(bottom, right), _intersect(bottom, left)]
    if any(p is None for p in pts):
        return None
    q = order_corners(pts)
    return q if _valid_quad(q, gray.shape) else None


def corners_minrect(gray: np.ndarray):
    """Dự phòng 2: hình chữ nhật bao nhỏ nhất của vùng giấy (giấy sáng hơn nền: tách bằng Otsu, C5 · S65)."""
    _, paper = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))  # lấp chữ trên giấy (C7 · S38)
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    q = order_corners(cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea))))
    q[:, 0] = np.clip(q[:, 0], 0, gray.shape[1] - 1)
    q[:, 1] = np.clip(q[:, 1], 0, gray.shape[0] - 1)
    return q if _valid_quad(q, gray.shape) else None


def full_image_corners(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)


def is_full_frame(q, shape, tol: float = 0.02) -> bool:
    """Tứ giác trùng gần hết khung ảnh (ảnh scan / chụp màn hình) → không cần nắn."""
    h, w = shape[:2]
    return np.abs(np.asarray(q) - full_image_corners(shape)).max() < tol * max(h, w)


def find_corners(gray: np.ndarray, canny_low: int = 50, canny_high: int = 150) -> tuple:
    """Chạy lần lượt contour → hough → minrect → cả ảnh. Trả về (corners, method)."""
    for method, fn in (("contour", lambda g: corners_contour(g, canny_low, canny_high)),
                       ("hough", lambda g: corners_hough(g, canny_low, canny_high)),
                       ("minrect", corners_minrect)):
        q = fn(gray)
        if q is not None and not is_full_frame(q, gray.shape):
            return q, method
    return full_image_corners(gray.shape), "full_image"


def homography_from_corners(corners: np.ndarray) -> tuple:
    """H 3×3 từ 4 góc (TL, TR, BR, BL) sang hình chữ nhật đích. Trả về (H, (width, height)).

    Cạnh đích = cạnh dài hơn trong mỗi cặp cạnh đối (giữ độ phân giải).
    """
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    tl, tr, br, bl = c
    width = max(int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))), 1)
    height = max(int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))), 1)
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    return cv2.getPerspectiveTransform(c, dst), (width, height)


def warp(img: np.ndarray, H: np.ndarray, size: tuple, interpolation: str = "bicubic") -> np.ndarray:
    """Ánh xạ ngược theo H, nội suy giữa các mẫu (C2 · S23–24)."""
    return cv2.warpPerspective(img, np.asarray(H, dtype=np.float64), tuple(int(v) for v in size),
                               flags=INTERPOLATION[interpolation], borderMode=cv2.BORDER_REPLICATE)


def rectify(img: np.ndarray, canny_low: int = 50, canny_high: int = 150) -> tuple:
    """A1: Canny → 4 góc → warpPerspective. Không tìm được tờ giấy thì trả ảnh gốc.

    Trả về (ảnh, method) với method ∈ {contour, hough, minrect, full_image}.
    """
    corners, method = find_corners(to_gray(img), canny_low, canny_high)
    if method == "full_image":
        return img, method
    H, size = homography_from_corners(corners)
    return warp(img, H, size), method
