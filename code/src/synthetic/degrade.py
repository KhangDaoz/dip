"""Các phép làm hỏng có kiểm soát.

Mọi hàm nhận ảnh BGR sạch + mức độ + rng (np.random.Generator), trả về (ảnh_hỏng, gt: dict).
gt chứa đủ thông tin để chấm điểm (ma trận H, góc, mask...). Mask: 255 = vùng bị tác động.
"""
import cv2
import numpy as np


def _make_table_background(h: int, w: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    """Tạo nền bàn chụp tối tự nhiên (độ sáng 55..75) có vân nhẹ để biên tờ giấy nổi bật."""
    base_val = float(rng.uniform(55.0, 75.0))
    y_grad = np.linspace(-8.0, 8.0, h, dtype=np.float32)[:, np.newaxis]
    x_grad = np.linspace(-8.0, 8.0, w, dtype=np.float32)[np.newaxis, :]
    bg = np.full((h, w), base_val, dtype=np.float32) + y_grad + x_grad
    noise = rng.uniform(-5.0, 5.0, size=(h, w)).astype(np.float32)
    bg = np.clip(bg + noise, 30.0, 95.0).astype(np.uint8)
    if channels == 3:
        b = np.clip(bg.astype(np.float32) * 0.90, 0, 255).astype(np.uint8)
        g = np.clip(bg.astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
        r = np.clip(bg.astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])
    return bg


def perspective(img, strength: float = 0.10, rng=None, background=None, level=None, **kwargs):
    """Dịch ngẫu nhiên 4 góc đặt lên nền bàn chụp, mô phỏng góc chụp nghiêng phối cảnh.

    Thứ tự 4 góc chuẩn: [TL, TR, BR, BL].
    gt chứa:
        - "H": ma trận 3x3 biến đổi clean -> degraded (khớp cv2.perspectiveTransform(src, H) == corners).
        - "H_clean_to_degraded": ma trận H dạng list.
        - "H_degraded_to_clean": ma trận nghịch đảo H^-1 (tương đương PreprocessResult.homography của Hiếu).
        - "corners": 4 góc [TL, TR, BR, BL] trên ảnh degraded.
        - "size": [w, h], "shape": [h, w], "strength": float, "page_mask": uint8 (h, w).
    """
    if level is not None:
        strength = level
    if "strength" in kwargs and kwargs["strength"] is not None:
        strength = kwargs["strength"]

    if rng is None:
        rng = np.random.default_rng()

    if isinstance(strength, (int, np.integer)) and strength in (1, 2, 3):
        level_map = {1: 0.05, 2: 0.10, 3: 0.20}
        s_val = level_map[int(strength)]
    else:
        s_val = float(strength)

    h, w = img.shape[:2]
    src_corners = np.float32([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])

    if s_val <= 0.0:
        dst_corners = src_corners.copy()
        H = np.eye(3, dtype=np.float64)
        page_mask = np.full((h, w), 255, dtype=np.uint8)
        out = img.copy()
    else:
        mw = (w - 1.0) * s_val * 0.6
        mh = (h - 1.0) * s_val * 0.6

        tl = [mw + float(rng.uniform(-0.4, 0.6)) * s_val * w, mh + float(rng.uniform(-0.4, 0.6)) * s_val * h]
        tr = [w - 1.0 - mw + float(rng.uniform(-0.6, 0.4)) * s_val * w, mh + float(rng.uniform(-0.4, 0.6)) * s_val * h]
        br = [w - 1.0 - mw + float(rng.uniform(-0.6, 0.4)) * s_val * w, h - 1.0 - mh + float(rng.uniform(-0.6, 0.4)) * s_val * h]
        bl = [mw + float(rng.uniform(-0.4, 0.6)) * s_val * w, h - 1.0 - mh + float(rng.uniform(-0.4, 0.6)) * s_val * h]

        dst_corners = np.float32([tl, tr, br, bl])
        dst_corners[:, 0] = np.clip(dst_corners[:, 0], 0.0, w - 1.0)
        dst_corners[:, 1] = np.clip(dst_corners[:, 1], 0.0, h - 1.0)

        H = cv2.getPerspectiveTransform(src_corners, dst_corners).astype(np.float64)

        if background is None:
            canvas = _make_table_background(h, w, 3 if img.ndim == 3 else 1, rng)
        elif isinstance(background, np.ndarray):
            if background.shape[:2] != (h, w):
                canvas = cv2.resize(background, (w, h))
            else:
                canvas = background.copy()
        else:
            c_val = int(background)
            canvas = np.full((h, w, 3) if img.ndim == 3 else (h, w), c_val, dtype=np.uint8)

        warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_full = np.full((h, w), 255, dtype=np.uint8)
        page_mask = cv2.warpPerspective(mask_full, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        out = canvas.copy()
        out[page_mask > 0] = warped[page_mask > 0]

    gt = {
        "H": H,
        "H_clean_to_degraded": H.tolist(),
        "H_degraded_to_clean": np.linalg.inv(H).tolist(),
        "corners": dst_corners.astype(np.float32),
        "size": [int(w), int(h)],
        "shape": [int(h), int(w)],
        "strength": float(s_val),
        "page_mask": page_mask,
    }
    return out, gt


def rotate(img, angle_deg: float = 0.0, rng=None, border_value=None, random_sign: bool = False, **kwargs):
    """Xoay ảnh quanh tâm ngược chiều kim đồng hồ angle_deg độ (cùng quy ước với deskew.rotate).

    gt chứa:
        - "angle_deg": góc xoay thực tế.
        - "M": ma trận affine 2x3.
        - "H": ma trận đồng nhất 3x3 tương đương (H_rot).
        - "H_clean_to_degraded": H_rot dạng list.
        - "H_degraded_to_clean": H_rot^-1 dạng list.
        - "corners": 4 góc tờ giấy gốc sau khi xoay.
        - "page_mask": mặt nạ vùng tờ giấy gốc còn nằm trong khung (h, w).
    """
    if "angle" in kwargs and kwargs["angle"] is not None:
        angle_deg = kwargs["angle"]

    if random_sign and rng is not None:
        ang = float(angle_deg) * (1.0 if rng.random() >= 0.5 else -1.0)
    else:
        ang = float(angle_deg)

    # Giữ kiểu int nếu người gọi truyền int để khớp chính xác assert gt["angle_deg"] == 10
    record_angle = int(ang) if (isinstance(angle_deg, (int, np.integer)) or int(ang) == ang) else float(ang)

    h, w = img.shape[:2]
    src_corners = np.float32([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])

    cx, cy = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    H_rot = np.eye(3, dtype=np.float64)
    H_rot[:2, :] = M

    if border_value is None:
        b_val = (245, 245, 245) if img.ndim == 3 else 245
    elif isinstance(border_value, (int, float)):
        b_val = (int(border_value),) * 3 if img.ndim == 3 else int(border_value)
    else:
        b_val = border_value

    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=b_val)
    mask_full = np.full((h, w), 255, dtype=np.uint8)
    page_mask = cv2.warpAffine(mask_full, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    corners_rot = cv2.perspectiveTransform(src_corners.reshape(-1, 1, 2), H_rot).reshape(4, 2)

    gt = {
        "angle_deg": record_angle,
        "M": M,
        "H": H_rot,
        "H_clean_to_degraded": H_rot.tolist(),
        "H_degraded_to_clean": np.linalg.inv(H_rot).tolist(),
        "corners": corners_rot.astype(np.float32),
        "size": [int(w), int(h)],
        "shape": [int(h), int(w)],
        "page_mask": page_mask,
    }
    return out, gt


def transform_box(box, H: np.ndarray, img_shape: tuple = None) -> dict:
    """Biến đổi một bounding box [x, y, w, h] qua ma trận Homography H (3x3).

    Trả về dict:
        - "bbox": [x, y, w, h] đã cắt biên vào khung ảnh nếu có img_shape.
        - "bbox_unclipped": [x, y, w, h] bao ngoài chưa cắt biên.
        - "polygon": 4 đỉnh tứ giác sau biến đổi [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        - "visible_ratio": tỷ lệ diện tích đa giác còn nằm trong khung ảnh [0.0 .. 1.0].
        - "clipped": bool (True nếu bị cắt > 5% diện tích).
        - "valid": bool (True nếu hộp có diện tích > 0 và visible_ratio > 0.1).
    """
    if isinstance(box, dict):
        b = box.get("bbox", box.get("box", [0, 0, 0, 0]))
    else:
        b = box

    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    poly = cv2.perspectiveTransform(pts, H.astype(np.float64)).reshape(4, 2)

    x_min_raw = float(poly[:, 0].min())
    y_min_raw = float(poly[:, 1].min())
    x_max_raw = float(poly[:, 0].max())
    y_max_raw = float(poly[:, 1].max())

    w_raw = max(0.0, x_max_raw - x_min_raw)
    h_raw = max(0.0, y_max_raw - y_min_raw)
    bbox_unclipped = [int(round(x_min_raw)), int(round(y_min_raw)), int(round(w_raw)), int(round(h_raw))]

    if img_shape is not None:
        h_out, w_out = img_shape[:2]
        x_min_clip = int(np.clip(np.floor(x_min_raw), 0, w_out))
        y_min_clip = int(np.clip(np.floor(y_min_raw), 0, h_out))
        x_max_clip = int(np.clip(np.ceil(x_max_raw), 0, w_out))
        y_max_clip = int(np.clip(np.ceil(y_max_raw), 0, h_out))
        w_clip = max(0, x_max_clip - x_min_clip)
        h_clip = max(0, y_max_clip - y_min_clip)
        bbox_clipped = [x_min_clip, y_min_clip, w_clip, h_clip]

        canvas_poly = np.float32([[0, 0], [w_out, 0], [w_out, h_out], [0, h_out]])
        poly_float = poly.astype(np.float32)
        orig_area = float(cv2.contourArea(poly_float))

        if orig_area <= 1e-4:
            visible_ratio = 1.0 if (0 <= x_min_raw < w_out and 0 <= y_min_raw < h_out) else 0.0
        else:
            ret, inter = cv2.intersectConvexConvex(poly_float, canvas_poly)
            inter_area = float(cv2.contourArea(inter)) if ret > 0 else 0.0
            visible_ratio = float(np.clip(inter_area / orig_area, 0.0, 1.0))

        clipped = bool(visible_ratio < 0.95)
        valid = bool(w_clip > 1 and h_clip > 1 and visible_ratio > 0.1)
    else:
        bbox_clipped = bbox_unclipped
        visible_ratio = 1.0
        clipped = False
        valid = bool(w_raw > 1 and h_raw > 1)

    return {
        "bbox": bbox_clipped,
        "bbox_unclipped": bbox_unclipped,
        "polygon": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in poly],
        "visible_ratio": round(visible_ratio, 4),
        "clipped": clipped,
        "valid": valid,
    }


def transform_boxes(boxes: list, H: np.ndarray, img_shape: tuple = None) -> list[dict]:
    """Biến đổi danh sách bounding box / annotation qua ma trận Homography H (3x3)."""
    results = []
    for item in boxes:
        if isinstance(item, dict):
            b_val = item.get("bbox", item.get("box", None))
            if b_val is None:
                continue
            t_res = transform_box(b_val, H, img_shape=img_shape)
            merged = item.copy()
            merged.update(t_res)
            results.append(merged)
        else:
            results.append(transform_box(item, H, img_shape=img_shape))
    return results


def _make_motion_kernel(radius: float, angle_deg: float) -> tuple[np.ndarray, int]:
    """Tạo kernel motion blur tuyến tính đối xứng, chuẩn hóa tổng bằng 1.0."""
    ksize = max(3, int(2 * round(2.5 * radius) + 1))
    if ksize % 2 == 0:
        ksize += 1

    scale = 5
    super_size = ksize * scale
    canvas = np.zeros((super_size, super_size), dtype=np.float32)
    center = (super_size - 1) / 2.0
    half_len = (ksize * scale) / 2.0 * 0.9

    rad = np.deg2rad(angle_deg)
    dx = half_len * np.cos(rad)
    dy = half_len * np.sin(rad)
    pt1 = (int(round(center - dx)), int(round(center - dy)))
    pt2 = (int(round(center + dx)), int(round(center + dy)))

    cv2.line(canvas, pt1, pt2, 1.0, thickness=scale, lineType=cv2.LINE_AA)
    kernel = cv2.resize(canvas, (ksize, ksize), interpolation=cv2.INTER_AREA)

    ksum = float(kernel.sum())
    if ksum > 0:
        kernel /= ksum
    else:
        kernel[ksize // 2, ksize // 2] = 1.0
    return kernel, ksize


def blur(img, level=1, rng=None, kind: str = "gaussian", radius=None, **kwargs):
    """kind: "gaussian" | "motion". gt: {"kind", "level", "radius", "ksize", "angle_deg", "sigma"}."""
    if "radius" in kwargs and kwargs["radius"] is not None:
        radius = kwargs["radius"]
    if radius is not None:
        r_val = float(radius)
    else:
        r_val = float(level)

    if rng is None:
        rng = np.random.default_rng()

    if kind not in ("gaussian", "motion", "random"):
        raise ValueError(f"kind không hợp lệ: {kind}. Chỉ chấp nhận 'gaussian' hoặc 'motion'.")
    if kind == "random":
        kind = str(rng.choice(["gaussian", "motion"]))

    if r_val <= 0:
        return img.copy(), {
            "radius": 0.0,
            "level": 0,
            "kind": kind,
            "ksize": 1,
            "angle_deg": 0.0,
            "sigma": 0.0,
        }

    if kind == "gaussian":
        sigma = r_val
        ksize = int(2 * np.ceil(3 * sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, ksize)
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        gt = {
            "radius": r_val,
            "level": r_val,
            "kind": "gaussian",
            "ksize": int(ksize),
            "angle_deg": 0.0,
            "sigma": float(sigma),
        }
        return blurred, gt
    else:  # motion
        angle_deg = float(rng.uniform(0.0, 180.0))
        kernel, ksize = _make_motion_kernel(r_val, angle_deg)
        blurred = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        gt = {
            "radius": r_val,
            "level": r_val,
            "kind": "motion",
            "ksize": int(ksize),
            "angle_deg": float(angle_deg),
            "sigma": 0.0,
        }
        return blurred, gt


def _generate_shadow_map(h: int, w: int, rng: np.random.Generator, shadow_type: str = None) -> tuple[np.ndarray, str]:
    """Sinh bản đồ hệ số bóng chuẩn hóa S(x, y) trong khoảng [0.0, 1.0]."""
    if shadow_type is None:
        shadow_type = "planar" if rng.random() < 0.5 else "silhouette"

    if shadow_type == "planar":
        cx = float(rng.uniform(0.35 * w, 0.65 * w))
        cy = float(rng.uniform(0.35 * h, 0.65 * h))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        nx = np.cos(angle)
        ny = np.sin(angle)

        y_coords, x_coords = np.ogrid[:h, :w]
        dist = (x_coords - cx) * nx + (y_coords - cy) * ny

        sigma = max(15.0, 0.08 * min(h, w))
        s_map = 1.0 / (1.0 + np.exp(-dist / (sigma * 0.4)))
        s_map = (s_map - s_map.min()) / (s_map.max() - s_map.min() + 1e-7)
    else:
        canvas = np.zeros((h, w), dtype=np.float32)
        edge = rng.choice(["corner_tl", "corner_tr", "corner_bl", "corner_br", "top", "bottom", "left", "right"])

        if edge == "corner_tl":
            center = (int(rng.uniform(0.1 * w, 0.35 * w)), int(rng.uniform(0.1 * h, 0.35 * h)))
            axes = (int(rng.uniform(0.25 * w, 0.4 * w)), int(rng.uniform(0.25 * h, 0.4 * h)))
            cv2.ellipse(canvas, center, axes, rng.uniform(20, 70), 0, 360, 1.0, -1)
            cv2.rectangle(canvas, (0, 0), (center[0], center[1]), 1.0, -1)
        elif edge == "corner_tr":
            center = (int(rng.uniform(0.65 * w, 0.9 * w)), int(rng.uniform(0.1 * h, 0.35 * h)))
            axes = (int(rng.uniform(0.25 * w, 0.4 * w)), int(rng.uniform(0.25 * h, 0.4 * h)))
            cv2.ellipse(canvas, center, axes, rng.uniform(-70, -20), 0, 360, 1.0, -1)
            cv2.rectangle(canvas, (center[0], 0), (w, center[1]), 1.0, -1)
        elif edge == "corner_bl":
            center = (int(rng.uniform(0.1 * w, 0.35 * w)), int(rng.uniform(0.65 * h, 0.9 * h)))
            axes = (int(rng.uniform(0.25 * w, 0.4 * w)), int(rng.uniform(0.25 * h, 0.4 * h)))
            cv2.ellipse(canvas, center, axes, rng.uniform(-70, -20), 0, 360, 1.0, -1)
            cv2.rectangle(canvas, (0, center[1]), (center[0], h), 1.0, -1)
        elif edge == "corner_br":
            center = (int(rng.uniform(0.65 * w, 0.9 * w)), int(rng.uniform(0.65 * h, 0.9 * h)))
            axes = (int(rng.uniform(0.25 * w, 0.4 * w)), int(rng.uniform(0.25 * h, 0.4 * h)))
            cv2.ellipse(canvas, center, axes, rng.uniform(20, 70), 0, 360, 1.0, -1)
            cv2.rectangle(canvas, (center[0], center[1]), (w, h), 1.0, -1)
        elif edge == "top":
            center = (int(rng.uniform(0.3 * w, 0.7 * w)), int(rng.uniform(0.15 * h, 0.35 * h)))
            axes = (int(rng.uniform(0.2 * w, 0.35 * w)), int(rng.uniform(0.2 * h, 0.35 * h)))
            cv2.ellipse(canvas, center, axes, rng.uniform(-15, 15), 0, 360, 1.0, -1)
            cv2.rectangle(canvas, (center[0] - axes[0], 0), (center[0] + axes[0], center[1]), 1.0, -1)
        elif edge == "bottom":
            center = (int(rng.uniform(0.3 * w, 0.7 * w)), int(rng.uniform(0.65 * h, 0.85 * h)))
            axes = (int(rng.uniform(0.2 * w, 0.35 * w)), int(rng.uniform(0.2 * h, 0.35 * h)))
            cv2.ellipse(canvas, center, axes, rng.uniform(-15, 15), 0, 360, 1.0, -1)
            cv2.rectangle(canvas, (center[0] - axes[0], center[1]), (center[0] + axes[0], h), 1.0, -1)
        elif edge == "left":
            center = (int(rng.uniform(0.15 * w, 0.35 * w)), int(rng.uniform(0.3 * h, 0.7 * h)))
            axes = (int(rng.uniform(0.2 * w, 0.35 * w)), int(rng.uniform(0.2 * h, 0.35 * h)))
            cv2.ellipse(canvas, center, axes, rng.uniform(-15, 15), 0, 360, 1.0, -1)
            cv2.rectangle(canvas, (0, center[1] - axes[1]), (center[0], center[1] + axes[1]), 1.0, -1)
        else:
            center = (int(rng.uniform(0.65 * w, 0.85 * w)), int(rng.uniform(0.3 * h, 0.7 * h)))
            axes = (int(rng.uniform(0.2 * w, 0.35 * w)), int(rng.uniform(0.2 * h, 0.35 * h)))
            cv2.ellipse(canvas, center, axes, rng.uniform(-15, 15), 0, 360, 1.0, -1)
            cv2.rectangle(canvas, (center[0], center[1] - axes[1]), (w, center[1] + axes[1]), 1.0, -1)

        ksize = int(2 * np.ceil(0.08 * min(h, w)) + 1)
        if ksize % 2 == 0:
            ksize += 1
        s_map = cv2.GaussianBlur(canvas, (ksize, ksize), sigmaX=0.06 * min(h, w))
        s_max = float(s_map.max())
        if s_max > 0:
            s_map /= s_max

    return s_map.astype(np.float32), shadow_type


def shadow(img, level=2, rng=None, darkness=None, shadow_type=None, **kwargs):
    """Nhân mặt nạ gradient phẳng hoặc hình bàn tay/điện thoại đã làm mờ.

    Hệ số truyền sáng alpha = 1.0 - darkness * S(x, y).
    gt: {"level", "darkness", "shadow_type", "mask", "shadow_mask", "coverage_ratio"}.
    - mask / shadow_mask: np.ndarray (H, W), dtype uint8 biểu diễn mức độ che bóng [0..255].
    """
    if "darkness" in kwargs and kwargs["darkness"] is not None:
        darkness = kwargs["darkness"]
    if darkness is not None:
        val = darkness
    else:
        val = level

    if rng is None:
        rng = np.random.default_rng()

    if isinstance(val, (int, np.integer)) and val in (1, 2, 3):
        level_map = {1: 0.2, 2: 0.45, 3: 0.7}
        d_val = level_map[int(val)]
        level_idx = int(val)
    elif val > 1.0:
        level_idx = int(round(val))
        level_map = {1: 0.2, 2: 0.45, 3: 0.7}
        d_val = level_map.get(level_idx, 0.45)
    else:
        d_val = float(np.clip(val, 0.0, 1.0))
        level_idx = 1 if d_val <= 0.3 else (2 if d_val <= 0.55 else 3)

    h, w = img.shape[:2]
    s_map, s_type = _generate_shadow_map(h, w, rng, shadow_type=shadow_type)

    alpha = 1.0 - d_val * s_map
    if img.ndim == 3:
        out = np.clip(np.round(img.astype(np.float32) * alpha[:, :, np.newaxis]), 0, 255).astype(np.uint8)
    else:
        out = np.clip(np.round(img.astype(np.float32) * alpha), 0, 255).astype(np.uint8)

    mask_uint8 = np.clip(np.round(255.0 * d_val * s_map), 0, 255).astype(np.uint8)
    coverage_ratio = float(np.mean(s_map > 0.1))

    gt = {
        "level": level_idx,
        "darkness": float(d_val),
        "shadow_type": s_type,
        "mask": mask_uint8,
        "shadow_mask": mask_uint8,
        "coverage_ratio": coverage_ratio,
    }
    return out, gt


# ─────────────────────────────── v2: lóa, dấu, bút, nén ───────────────────────────────

# Tham số theo mức — ghi vào bảng tham số của báo cáo (Phụ lục D)
GLARE_LEVELS = {1: {"sigma_frac": 0.06, "gain": 0.6},    # sigma elip = sigma_frac × cạnh ngắn; gain: hệ số pha trắng ở tâm
                2: {"sigma_frac": 0.10, "gain": 1.2},    # gain > 1 → lõi elip cháy trắng hoàn toàn
                3: {"sigma_frac": 0.15, "gain": 2.0}}
STAMP_LEVELS = {1: {"radius_frac": 0.08, "opacity": 0.55},   # bán kính dấu = radius_frac × cạnh ngắn
                2: {"radius_frac": 0.11, "opacity": 0.75},
                3: {"radius_frac": 0.14, "opacity": 0.92}}
PEN_LEVELS = {1: {"n_signatures": 1, "n_strokes": 0, "over_text": False},   # chữ ký ngắn ở vùng trống
              2: {"n_signatures": 2, "n_strokes": 3, "over_text": True}}    # nhiều nét cắt ngang dòng chữ
JPEG_NOISE_LEVELS = {1: {"jpeg_quality": 90, "noise_sigma": 3.0},
                     2: {"jpeg_quality": 60, "noise_sigma": 6.0},
                     3: {"jpeg_quality": 30, "noise_sigma": 10.0}}
EXPOSURE_GAIN_RANGE = (0.80, 0.92)   # v10, v14: nhân ảnh trước khi thêm lóa → giấy 255 về ≈ 204–235 (góp ý Hiếu 03/11)
STAMP_TEXT = "MEDTRACK DEMO * MAU DAU GIA *"   # chữ giả, không dùng tên cơ sở y tế thật
PEN_COLORS = {"blue": (150, 60, 25), "black": (35, 35, 40)}   # BGR
RED_INK = (45, 40, 200)                                        # BGR


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _ink_map(img: np.ndarray) -> np.ndarray:
    """Mực chữ trên ảnh (Otsu ngược): True = mực."""
    _, b = cv2.threshold(_to_gray(img), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b > 0


def _pick_center(h: int, w: int, rng: np.random.Generator, region=None, margin: float = 0.0):
    """Tâm ngẫu nhiên trong region [x, y, w, h] (nếu có) hoặc trong 70% giữa ảnh; cách mép ảnh ≥ margin px."""
    if region is not None:
        x0, y0, rw, rh = [float(v) for v in region]
        x1, y1 = x0 + rw, y0 + rh
    else:
        x0, y0, x1, y1 = 0.15 * w, 0.15 * h, 0.85 * w, 0.85 * h
    x0, x1 = max(x0, margin), min(x1, w - 1 - margin)
    y0, y1 = max(y0, margin), min(y1, h - 1 - margin)
    if x1 <= x0:
        x0 = x1 = w / 2.0
    if y1 <= y0:
        y0 = y1 = h / 2.0
    return float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))


def _multiply_ink(img: np.ndarray, alpha: np.ndarray, color_bgr) -> np.ndarray:
    """Phủ mực kiểu in chồng (multiply): chữ đen bên dưới vẫn đen, giấy trắng chuyển sang màu mực."""
    a = alpha.astype(np.float32)
    if img.ndim == 3:
        tint = np.array(color_bgr, np.float32) / 255.0
        out = img.astype(np.float32) * (1.0 - a[..., None] * (1.0 - tint[None, None, :]))
    else:
        lum = (0.114 * color_bgr[0] + 0.587 * color_bgr[1] + 0.299 * color_bgr[2]) / 255.0
        out = img.astype(np.float32) * (1.0 - a * (1.0 - lum))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _ink_texture(shape, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Độ đậm mực không đều (0.55..1) + lỗ chấm nhỏ, mô phỏng mực dấu / bút thật."""
    noise = rng.random(shape).astype(np.float32)
    s = max(0.8, scale)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=s)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-7)
    tex = 0.55 + 0.45 * noise
    tex[rng.random(shape) < 0.03] = 0.0
    return tex


def glare(img, level: int = 2, rng=None, center=None, region=None, **kwargs):
    """Lóa flash: cộng vùng sáng elip Gaussian vào kênh V và giảm kênh S (HSV).

    Hệ số pha trắng B(x, y) = min(1, gain × G(x, y)), G là elip Gaussian xoay ngẫu nhiên, đỉnh = 1 tại tâm.
    V' = V + (255 − V)·B, S' = S·(1 − B) → chữ trong lõi lóa nhạt dần rồi mất hẳn khi B → 1.
    glare_mask (lóa = 255): **vùng tác động** B ≥ 0,5 (chữ đã mất ít nhất một nửa độ tương phản).
    gt: {"level", "glare_mask", "center", "sigma_xy", "angle_deg", "gain", "glare_area_ratio", "saturated_ratio"}.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]
    lv = int(level)
    if lv <= 0:
        return img.copy(), {"level": 0, "glare_mask": np.zeros((h, w), np.uint8), "glare_area_ratio": 0.0,
                            "saturated_ratio": 0.0}
    p = GLARE_LEVELS[lv]
    sx = p["sigma_frac"] * min(h, w) * float(rng.uniform(0.8, 1.3))
    sy = sx * float(rng.uniform(0.6, 1.0))
    ang = float(rng.uniform(0.0, 180.0))
    cx, cy = center if center is not None else _pick_center(h, w, rng, region)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.deg2rad(ang)
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    g = np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    b = np.clip(p["gain"] * g, 0.0, 1.0).astype(np.float32)

    if img.ndim == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 2] = hsv[..., 2] + (255.0 - hsv[..., 2]) * b
        hsv[..., 1] = hsv[..., 1] * (1.0 - b)
        out = cv2.cvtColor(np.clip(np.round(hsv), 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        v_out = hsv[..., 2]
    else:
        v_out = img.astype(np.float32) + (255.0 - img.astype(np.float32)) * b
        out = np.clip(np.round(v_out), 0, 255).astype(np.uint8)

    mask = (b >= 0.5).astype(np.uint8) * 255
    area = int((mask > 0).sum())
    gt = {
        "level": lv,
        "glare_mask": mask,
        "center": [round(float(cx), 1), round(float(cy), 1)],
        "sigma_xy": [round(float(sx), 1), round(float(sy), 1)],
        "angle_deg": round(ang, 2),
        "gain": p["gain"],
        "glare_area_ratio": round(area / float(h * w), 5),
        "saturated_ratio": round(float((v_out[mask > 0] >= 250).mean()) if area else 0.0, 4),
    }
    return out, gt


def _stamp_alpha(radius: float, rng: np.random.Generator, text: str) -> np.ndarray:
    """Vẽ dấu tròn (vành kép + chữ chạy quanh vành + ngôi sao) trên canvas vuông, giá trị 0..1."""
    size = int(2 * radius + 6)
    c = size / 2.0
    canvas = np.zeros((size, size), np.uint8)   # OpenCV chỉ vẽ chữ trên uint8 → vẽ 0/255 rồi đổi về 0..1
    ctr = (int(round(c)), int(round(c)))
    cv2.circle(canvas, ctr, int(radius * 0.96), 255, max(2, int(round(radius * 0.06))), cv2.LINE_AA)
    cv2.circle(canvas, ctr, int(radius * 0.68), 255, max(1, int(round(radius * 0.03))), cv2.LINE_AA)

    # chữ chạy theo chiều kim đồng hồ quanh vành, đọc được từ phía ngoài
    font = cv2.FONT_HERSHEY_DUPLEX
    char_h = 0.17 * radius
    scale = char_h / 22.0
    thick = max(1, int(round(scale * 1.6)))
    ring_r = radius * 0.82
    step = 2 * np.pi / len(text)
    patch = int(char_h * 2.2) + 4
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        theta = i * step
        p = np.zeros((patch, patch), np.uint8)
        (tw, th), _ = cv2.getTextSize(ch, font, scale, thick)
        cv2.putText(p, ch, (int((patch - tw) / 2), int((patch + th) / 2)), font, scale, 255, thick, cv2.LINE_AA)
        rot = cv2.warpAffine(p, cv2.getRotationMatrix2D((patch / 2, patch / 2), -np.degrees(theta), 1.0), (patch, patch))
        px = c + ring_r * np.sin(theta) - patch / 2
        py = c - ring_r * np.cos(theta) - patch / 2
        x0, y0 = int(round(px)), int(round(py))
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(size, x0 + patch), min(size, y0 + patch)
        if xs1 > xs0 and ys1 > ys0:
            canvas[ys0:ys1, xs0:xs1] = np.maximum(canvas[ys0:ys1, xs0:xs1],
                                                  rot[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0])

    # ngôi sao năm cánh ở giữa
    r_out, r_in = radius * 0.30, radius * 0.12
    pts = []
    for k in range(10):
        r = r_out if k % 2 == 0 else r_in
        a = -np.pi / 2 + k * np.pi / 5
        pts.append([c + r * np.cos(a), c + r * np.sin(a)])
    cv2.fillPoly(canvas, [np.round(np.array(pts)).astype(np.int32)], 255, cv2.LINE_AA)

    canvas = canvas.astype(np.float32) / 255.0
    canvas *= _ink_texture(canvas.shape, rng, scale=radius * 0.02)
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=0.6)
    rot_deg = float(rng.uniform(-35.0, 35.0))
    canvas = cv2.warpAffine(canvas, cv2.getRotationMatrix2D((c, c), rot_deg, 1.0), (size, size))
    return np.clip(canvas, 0.0, 1.0), rot_deg


def stamp(img, level: int = 2, rng=None, center=None, region=None, text: str = STAMP_TEXT, **kwargs):
    """Dấu tròn đỏ có chữ, độ trong suốt theo mức, đè lên vùng chữ (region = vùng thuốc nếu có).

    Phủ mực kiểu multiply nên chữ đen bị dấu đè vẫn còn thấy, đúng như dấu mực thật.
    stamp_mask (dấu = 255): pixel có độ đậm mực dấu > 0,15.
    gt: {"level", "stamp_mask", "center", "radius", "opacity", "rotation_deg", "color_bgr", "text",
         "text_covered_ratio"} — text_covered_ratio = tỉ lệ pixel chữ (Otsu trên ảnh đầu vào) nằm dưới dấu.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]
    lv = int(level)
    if lv <= 0:
        return img.copy(), {"level": 0, "stamp_mask": np.zeros((h, w), np.uint8), "text_covered_ratio": 0.0}
    p = STAMP_LEVELS[lv]
    radius = p["radius_frac"] * min(h, w) * float(rng.uniform(0.9, 1.1))
    cx, cy = center if center is not None else _pick_center(h, w, rng, region, margin=radius * 0.5)
    small, rot_deg = _stamp_alpha(radius, rng, text)

    alpha = np.zeros((h, w), np.float32)
    s = small.shape[0]
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    xs0, ys0, xs1, ys1 = max(0, x0), max(0, y0), min(w, x0 + s), min(h, y0 + s)
    alpha[ys0:ys1, xs0:xs1] = small[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    alpha *= p["opacity"]

    color = tuple(int(np.clip(c + rng.integers(-15, 16), 0, 255)) for c in RED_INK)
    ink_before = _ink_map(img)
    out = _multiply_ink(img, alpha, color)
    mask = (alpha > 0.15).astype(np.uint8) * 255
    n_ink = int(ink_before.sum())
    gt = {
        "level": lv,
        "stamp_mask": mask,
        "center": [round(float(cx), 1), round(float(cy), 1)],
        "radius": round(float(radius), 1),
        "opacity": p["opacity"],
        "rotation_deg": round(rot_deg, 2),
        "color_bgr": list(color),
        "text": text,
        "text_covered_ratio": round(float((ink_before & (mask > 0)).sum()) / n_ink, 4) if n_ink else 0.0,
    }
    return out, gt


def _bezier(p0, p1, p2, p3, n: int = 30) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def _signature_points(box, rng: np.random.Generator) -> list:
    """Chuỗi 4–7 đoạn Bézier bậc 3 nối liền, đi từ trái sang phải trong box [x0, y0, x1, y1] — dáng chữ ký."""
    x0, y0, x1, y1 = box
    n_seg = int(rng.integers(4, 8))
    xs = np.sort(rng.uniform(x0, x1, n_seg + 1))
    xs[0], xs[-1] = x0, x1
    pts, prev = [], np.array([xs[0], rng.uniform(y0, y1)])
    for k in range(n_seg):
        end = np.array([xs[k + 1], rng.uniform(y0, y1)])
        c1 = np.array([rng.uniform(prev[0], end[0] + 1), rng.uniform(y0 - 0.4 * (y1 - y0), y1 + 0.4 * (y1 - y0))])
        c2 = np.array([rng.uniform(prev[0], end[0] + 1), rng.uniform(y0 - 0.4 * (y1 - y0), y1 + 0.4 * (y1 - y0))])
        pts.append(_bezier(prev, c1, c2, end))
        prev = end
    return pts


def pen(img, level: int = 1, rng=None, color: str = None, region=None, **kwargs):
    """Nét bút bi xanh / đen giả chữ ký và gạch tay bằng các đường Bézier bậc 3.

    Mức 1: một chữ ký ngắn đặt ở vùng ít mực nhất của nửa dưới trang (vùng trống).
    Mức 2: hai chữ ký + 3 nét gạch dài cắt ngang vùng chữ (region = vùng thuốc nếu có).
    pen_mask (nét bút = 255): pixel có độ đậm mực bút > 0,15.
    gt: {"level", "pen_mask", "pen_color", "color_bgr", "n_strokes", "thickness"}.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]
    lv = int(level)
    if lv <= 0:
        return img.copy(), {"level": 0, "pen_mask": np.zeros((h, w), np.uint8), "pen_color": "none", "n_strokes": 0}
    p = PEN_LEVELS[lv]
    if color is None:
        color = "blue" if rng.random() < 0.7 else "black"
    color_bgr = tuple(int(np.clip(c + rng.integers(-12, 13), 0, 255)) for c in PEN_COLORS[color])
    thick = max(2, int(round(0.003 * w)))
    sig_w, sig_h = 0.25 * w, 0.07 * h

    ink = _ink_map(img).astype(np.float32)
    polylines = []
    for _ in range(p["n_signatures"]):
        if p["over_text"]:
            cx, cy = _pick_center(h, w, rng, region, margin=sig_w / 2)
        else:   # chọn trong 8 vị trí ngẫu nhiên ở nửa dưới trang vị trí ít mực nhất
            best = None
            for _ in range(8):
                cx_c = float(rng.uniform(sig_w / 2, w - sig_w / 2))
                cy_c = float(rng.uniform(0.6 * h, h - sig_h))
                bx0, by0 = int(cx_c - sig_w / 2), int(cy_c - sig_h / 2)
                score = float(ink[max(0, by0):by0 + int(sig_h), max(0, bx0):bx0 + int(sig_w)].mean())
                if best is None or score < best[0]:
                    best = (score, cx_c, cy_c)
            _, cx, cy = best
        box = (cx - sig_w / 2, cy - sig_h / 2, cx + sig_w / 2, cy + sig_h / 2)
        polylines.extend(_signature_points(box, rng))
    for _ in range(p["n_strokes"]):   # nét gạch dài hơi cong cắt ngang chữ
        cx, cy = _pick_center(h, w, rng, region)
        half = float(rng.uniform(0.15, 0.3)) * w
        tilt = float(rng.uniform(-0.15, 0.15))
        p0 = np.array([cx - half, cy - tilt * half])
        p3 = np.array([cx + half, cy + tilt * half])
        bend = float(rng.uniform(-0.03, 0.03)) * h
        polylines.append(_bezier(p0, p0 + (p3 - p0) / 3 + [0, bend], p0 + 2 * (p3 - p0) / 3 - [0, bend], p3, n=60))

    canvas = np.zeros((h, w), np.uint8)
    for pts in polylines:
        t = max(1, thick + int(rng.integers(-1, 2)))
        cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, 255, t, cv2.LINE_AA)
    alpha = canvas.astype(np.float32) / 255.0
    alpha *= 0.85 + 0.15 * _ink_texture(alpha.shape, rng, scale=1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    out = _multiply_ink(img, alpha, color_bgr)
    mask = (alpha > 0.15).astype(np.uint8) * 255
    gt = {
        "level": lv,
        "pen_mask": mask,
        "pen_color": color,
        "color_bgr": list(color_bgr),
        "n_strokes": len(polylines),
        "thickness": thick,
    }
    return out, gt


def jpeg_noise(img, level: int = 2, rng=None, **kwargs):
    """Cộng nhiễu Gaussian rồi nén / giải nén JPEG (mô phỏng cảm biến + lưu ảnh qua ứng dụng nhắn tin).

    gt: {"level", "jpeg_quality", "noise_sigma"}.
    """
    if rng is None:
        rng = np.random.default_rng()
    lv = int(level)
    if lv <= 0:
        return img.copy(), {"level": 0, "jpeg_quality": 100, "noise_sigma": 0.0}
    p = JPEG_NOISE_LEVELS[lv]
    noisy = img.astype(np.float32) + rng.normal(0.0, p["noise_sigma"], img.shape).astype(np.float32)
    noisy = np.clip(np.round(noisy), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_quality"]])
    if not ok:
        raise ValueError("Không nén được JPEG")
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR if img.ndim == 3 else cv2.IMREAD_GRAYSCALE)
    return out, {"level": lv, "jpeg_quality": p["jpeg_quality"], "noise_sigma": p["noise_sigma"]}


def exposure(img, gain: float) -> np.ndarray:
    """Giảm phơi sáng: nhân cả ảnh với gain (∈ EXPOSURE_GAIN_RANGE), mô phỏng mức giấy của ảnh chụp.

    Ảnh nguồn VAIPE là bản scan, giấy V = 255 → lóa (V' = V + (255 − V)·B) không làm giấy sáng thêm được,
    bộ phát hiện tìm vùng sáng hơn mặt giấy không thấy lóa. Ảnh chụp thật: giấy ≈ 170–235 (bộ 2, phân vị 70 của V).
    Không bốc số ngẫu nhiên, không đọc nội dung ảnh → không đổi vị trí lóa / dấu / bút khi chèn vào công thức.
    """
    return np.clip(np.round(img.astype(np.float32) * float(gain)), 0, 255).astype(np.uint8)


DEGRADATIONS = {"perspective": perspective, "rotate": rotate, "blur": blur, "shadow": shadow,
                "glare": glare, "stamp": stamp, "pen": pen, "jpeg_noise": jpeg_noise}
