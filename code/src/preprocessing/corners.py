"""Tìm 4 góc tờ giấy, có 2 lớp dự phòng. Thí nghiệm D2.

Mọi hàm trả về mảng float32 4×2 theo thứ tự TL, TR, BR, BL (tọa độ ảnh đưa vào), hoặc None nếu thất bại.
"""
import cv2
import numpy as np

from common.io import to_gray
from .quality import resize_long_side

MIN_AREA_RATIO = 0.2     # tứ giác phải chiếm ≥ 20% ảnh
FULL_AREA_RATIO = 0.98   # tứ giác ≥ 98% ảnh → coi như giấy phủ kín khung (ảnh scan / ảnh phẳng)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sắp 4 điểm bất kỳ về thứ tự TL, TR, BR, BL.

    Sắp theo góc quanh trọng tâm (trục y hướng xuống nên tăng dần = chiều kim đồng hồ),
    rồi xoay vòng để điểm có x + y nhỏ nhất đứng đầu. Đúng cả khi tờ giấy xoay gần 45°,
    trường hợp cách "tổng / hiệu tọa độ" quen thuộc bị sai.
    """
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def _area_ratio(quad: np.ndarray, shape) -> float:
    return cv2.contourArea(quad.astype(np.float32)) / float(shape[0] * shape[1])


def _valid_quad(quad, shape, min_area_ratio: float) -> bool:
    return (quad is not None and cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(np.float32))
            and _area_ratio(quad, shape) >= min_area_ratio)


def _edges(gray: np.ndarray, canny) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, *canny)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8))   # nối các đoạn mép bị đứt


def _paper_mask(gray: np.ndarray) -> np.ndarray:
    """Giấy sáng hơn nền: Otsu + closing để lấp chữ bên trong."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, min(gray.shape) // 50) | 1
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))


def find_corners_contour(img: np.ndarray, canny=(50, 150), eps_ratio: float = 0.02,
                         min_area_ratio: float = MIN_AREA_RATIO):
    """Canny → findContours → approxPolyDP, lấy tứ giác lồi lớn nhất (diện tích ≥ min_area_ratio × ảnh).

    Thử trên 2 bản đồ: biên Canny và mask giấy (Otsu) — bản nào cho tứ giác lớn hơn thì lấy.
    """
    gray = to_gray(img)
    best, best_area = None, 0.0
    for binary in (_edges(gray, canny), _paper_mask(gray)):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            approx = cv2.approxPolyDP(c, eps_ratio * cv2.arcLength(c, True), True)
            if len(approx) != 4:
                continue
            quad = order_corners(approx)
            if _valid_quad(quad, gray.shape, min_area_ratio) and cv2.contourArea(quad) > best_area:
                best, best_area = quad, cv2.contourArea(quad)
            break   # chỉ xét contour lớn nhất có dạng tứ giác của mỗi bản đồ
    return best


def _intersect(l1, l2):
    """Giao điểm 2 đường (rho, theta) dạng chuẩn Hough."""
    (r1, t1), (r2, t2) = l1, l2
    A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, [r1, r2])


def find_corners_hough(img: np.ndarray, canny=(50, 150), min_area_ratio: float = MIN_AREA_RATIO):
    """Hough lines, chọn 2 cạnh ngang + 2 cạnh dọc ngoài cùng, lấy 4 giao điểm.

    Đường gần ngang: theta ∈ [45°, 135°], xếp theo rho ≈ vị trí y; đường gần dọc: còn lại, chuẩn hóa
    theta về [-45°, 45°] để rho ≈ vị trí x. Chữ bên trong trang chỉ sinh đường nằm giữa nên không ảnh hưởng.
    """
    gray = to_gray(img)
    h, w = gray.shape
    lines = cv2.HoughLines(_edges(gray, canny), 1, np.pi / 180, int(0.25 * min(h, w)))
    if lines is None:
        return None
    horiz, vert = [], []
    for rho, theta in lines[:200, 0]:
        if np.pi / 4 <= theta <= 3 * np.pi / 4:
            horiz.append((rho, theta))
        else:
            vert.append((-rho, theta - np.pi) if theta > np.pi / 2 else (rho, theta))
    # Nét chữ sinh nhiều đường xiên giả → chỉ giữ đường lệch ≤ 15° so với hướng chủ đạo của nhóm
    # (trung vị theta của 5 đường nhiều phiếu nhất; HoughLines đã sắp theo số phiếu giảm dần).
    def dominant(group):
        if len(group) < 2:
            return []
        ref = np.median([t for _, t in group[:5]])
        return [l for l in group if abs(l[1] - ref) <= np.radians(15)]

    horiz, vert = dominant(horiz), dominant(vert)
    if len(horiz) < 2 or len(vert) < 2:
        return None
    top, bottom = min(horiz, key=lambda l: l[0]), max(horiz, key=lambda l: l[0])
    left, right = min(vert, key=lambda l: l[0]), max(vert, key=lambda l: l[0])
    if bottom[0] - top[0] < 0.3 * h or right[0] - left[0] < 0.3 * w:
        return None
    pts = [_intersect(a, b) for a, b in ((top, left), (top, right), (bottom, right), (bottom, left))]
    if any(p is None for p in pts):
        return None
    pts = np.array(pts, np.float32)
    margin = 0.05 * max(h, w)
    if (pts[:, 0] < -margin).any() or (pts[:, 0] > w + margin).any() \
            or (pts[:, 1] < -margin).any() or (pts[:, 1] > h + margin).any():
        return None
    quad = order_corners(pts)
    return quad if _valid_quad(quad, gray.shape, min_area_ratio) else None


def find_corners_minrect(img: np.ndarray, canny=(50, 150), min_area_ratio: float = MIN_AREA_RATIO):
    """minAreaRect của vùng giấy lớn nhất (mask Otsu). Luôn ra hình chữ nhật → không sửa được phối cảnh."""
    gray = to_gray(img)
    contours, _ = cv2.findContours(_paper_mask(gray), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    quad = order_corners(cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea))))
    return quad if _area_ratio(quad, gray.shape) >= min_area_ratio else None


def full_image_corners(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])


FINDERS = (("contour", find_corners_contour), ("hough", find_corners_hough), ("minrect", find_corners_minrect))

BORDER_TOL = 0.015        # cạnh cách mép ảnh ≤ 1,5% cạnh dài → coi như nằm trên khung
OUTSIDE_TEXT_MAX = 0.05   # > 5% ký tự nằm ngoài tứ giác → cắt mất nội dung (mép giả; 10% còn cắt dải đầu / cuối trang, review 02/11)
OUTSIDE_MARGIN = 0.005    # ký tự phải cách tứ giác > 0,5% đường chéo mới tính là "ngoài" (1,5% bỏ lọt dải đầu / cuối trang sát khung)
MAX_ANGLE_DEV = 40.0      # góc trong lệch > 40° so với 90° → không phải tờ giấy (bộ 5, phối cảnh 20%: tối đa 31,6°)


def max_angle_deviation(quad: np.ndarray) -> float:
    """Độ lệch lớn nhất (độ) giữa 4 góc trong của tứ giác và 90°."""
    q = np.asarray(quad, float).reshape(4, 2)
    dev = 0.0
    for i in range(4):
        u, v = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        cos = u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)
        dev = max(dev, abs(np.degrees(np.arccos(np.clip(cos, -1, 1))) - 90))
    return dev


def edges_on_border(quad: np.ndarray, shape, tol: float = BORDER_TOL) -> int:
    """Số cạnh của tứ giác nằm trùng một mép khung ảnh (cả 2 đầu cạnh sát cùng một mép)."""
    h, w = shape[:2]
    t = tol * max(h, w)
    n = 0
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        for axis, val in ((0, 0), (0, w - 1), (1, 0), (1, h - 1)):
            if abs(p[axis] - val) <= t and abs(q[axis] - val) <= t:
                n += 1
                break
    return n


def _char_components(gray: np.ndarray, min_area: int = 8):
    """Thành phần cỡ ký tự (adaptive threshold): cao 0,6–5% ảnh, rộng ≤ 10% ảnh.

    Bỏ được vùng tối lớn (bóng, nền, nếp gập) mà Otsu toàn ảnh hay nhận nhầm là mực.
    Trả về (labels, mask thành phần được giữ theo nhãn, centroids).
    """
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    ch, cw = stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_WIDTH]
    keep = (ch >= 0.006 * h) & (ch <= 0.05 * h) & (cw <= 0.1 * w) & (stats[:, cv2.CC_STAT_AREA] >= min_area)
    keep[0] = False
    return labels, keep, centroids


def char_centroids(gray: np.ndarray) -> np.ndarray:
    """Tâm các thành phần cỡ ký tự."""
    _, keep, centroids = _char_components(gray)
    return centroids[keep]


def char_mask(gray: np.ndarray, min_area: int = 8) -> np.ndarray:
    """Ảnh mực chỉ gồm thành phần cỡ ký tự (255)."""
    labels, keep, _ = _char_components(gray, min_area)
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def text_outside_ratio(gray: np.ndarray, quad: np.ndarray, margin: float = OUTSIDE_MARGIN) -> float:
    """Tỉ lệ ký tự nằm ngoài tứ giác. Mép giấy thật: bên ngoài là nền, tỉ lệ ≈ 0; tứ giác bám nếp gập,
    khung bảng hay tờ giấy phía sau thì cắt mất chữ, tỉ lệ cao."""
    pts = char_centroids(gray)
    if len(pts) == 0:
        return 0.0
    poly = np.asarray(quad, np.float32).reshape(-1, 1, 2)
    m = margin * np.hypot(*gray.shape[:2])
    dist = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in pts])
    return float((dist < -m).mean())


def find_corners(img: np.ndarray, long_side: int = 1000):
    """Thử contour → hough → minrect; không được thì trả về 4 góc của cả ảnh.

    Dò trên ảnh đã resize cạnh dài = long_side, nhưng trả tọa độ theo ảnh gốc.
    Trả về (corners 4×2, method) với method thuộc common.contracts.CORNER_METHODS.
    Tứ giác phủ ≥ 98% khung hoặc có ≥ 3 cạnh nằm trên khung cũng trả "full_image" (ảnh scan, giấy
    tràn khung). Tứ giác cắt mất chữ (text_outside_ratio > OUTSIDE_TEXT_MAX) hoặc méo vô lý (góc trong lệch
    > MAX_ANGLE_DEV so với 90°) bị loại, thử phương án sau.
    """
    small, scale = resize_long_side(img, long_side)
    gray = to_gray(small)
    for name, finder in FINDERS:
        quad = finder(small)
        if quad is None:
            continue
        if _area_ratio(quad, small.shape) >= FULL_AREA_RATIO or edges_on_border(quad, small.shape) >= 3:
            break
        if max_angle_deviation(quad) > MAX_ANGLE_DEV or text_outside_ratio(gray, quad) > OUTSIDE_TEXT_MAX:
            continue
        return (quad / scale).astype(np.float32), name
    return full_image_corners(img.shape), "full_image"
