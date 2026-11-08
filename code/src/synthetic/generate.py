"""Bộ sinh dữ liệu tổng hợp (Synthetic Benchmark Bộ 5).

Hỗ trợ:
- v1 (02/11): [perspective, rotate, blur, shadow]
- v2 (04/11): [perspective, rotate, blur, shadow, glare, stamp, pen, jpeg_noise]
  cập nhật 03/11 (góp ý Hiếu): v10, v14 giảm phơi sáng trước khi thêm lóa + lưu clean_exposed/

Chạy dòng lệnh (từ thư mục code/):
    python -m synthetic.generate --version v1      # 1.000 mẫu v00–v09
    python -m synthetic.generate --version v2      # thêm 500 mẫu v10–v14, cần v1 có sẵn
    python -m synthetic.generate --config code/src/synthetic/config.yaml
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = REPO_ROOT / "code"
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from synthetic import degrade
except ImportError:
    import degrade


RECIPE_NAMES_V1 = [
    "single_rotate_low_mid",      # v00: rotate 5..10 deg
    "single_rotate_high",         # v01: rotate 20..30 deg
    "single_perspective",         # v02: perspective 0.05..0.20
    "single_blur_gaussian",       # v03: gaussian blur level 1..4
    "single_blur_motion",         # v04: motion blur level 1..4
    "single_shadow",              # v05: planar or silhouette shadow level 1..3
    "combo_rotate_shadow",        # v06: rotate + shadow
    "combo_perspective_blur",     # v07: perspective + blur
    "combo_perspective_shadow",   # v08: perspective + shadow
    "combo_full_v1"               # v09: perspective -> rotate -> shadow -> blur
]


def _apply_perspective_and_rotation(clean_bgr: np.ndarray, p_strength: float, angle_deg: float, sample_rng: np.random.Generator):
    """Phối hợp nắn phối cảnh và xoay nghiêng có co lề để 4 góc không bị cắt biên."""
    h, w = clean_bgr.shape[:2]
    src_corners = np.float32([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])

    if p_strength <= 0.0 and angle_deg == 0.0:
        return clean_bgr.copy(), np.eye(3, dtype=np.float64), src_corners, np.full((h, w), 255, dtype=np.uint8)

    if p_strength > 0.0 and angle_deg == 0.0:
        cur_img, gt_persp = degrade.perspective(clean_bgr, strength=p_strength, rng=sample_rng)
        return cur_img, gt_persp["H"].astype(np.float64), gt_persp["corners"], gt_persp["page_mask"]

    if p_strength <= 0.0 and angle_deg != 0.0:
        cur_img, gt_rot = degrade.rotate(clean_bgr, angle_deg=angle_deg, border_value=(245, 245, 245), rng=sample_rng)
        return cur_img, gt_rot["H"].astype(np.float64), gt_rot["corners"], gt_rot["page_mask"]

    # Phối hợp cả perspective và rotate
    _, gt_persp = degrade.perspective(clean_bgr, strength=p_strength, rng=sample_rng)
    H_p = gt_persp["H"].astype(np.float64)

    cx, cy = (w / 2.0, h / 2.0)
    M_r = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    H_r = np.eye(3, dtype=np.float64)
    H_r[:2, :] = M_r

    H_total = np.dot(H_r, H_p)

    corners_test = cv2.perspectiveTransform(src_corners.reshape(-1, 1, 2), H_total).reshape(4, 2)
    pad = 8.0
    min_x, max_x = float(corners_test[:, 0].min()), float(corners_test[:, 0].max())
    min_y, max_y = float(corners_test[:, 1].min()), float(corners_test[:, 1].max())

    scale_x = (cx - pad) / max(cx - min_x, max_x - cx) if (min_x < pad or max_x > w - 1 - pad) else 1.0
    scale_y = (cy - pad) / max(cy - min_y, max_y - cy) if (min_y < pad or max_y > h - 1 - pad) else 1.0
    scale = min(1.0, scale_x, scale_y)
    if scale < 1.0:
        S = np.array([[scale, 0.0, cx * (1.0 - scale)], [0.0, scale, cy * (1.0 - scale)], [0.0, 0.0, 1.0]], dtype=np.float64)
        H_total = np.dot(S, H_total)

    corners = cv2.perspectiveTransform(src_corners.reshape(-1, 1, 2), H_total).reshape(4, 2)
    canvas = degrade._make_table_background(h, w, 3 if clean_bgr.ndim == 3 else 1, sample_rng)
    warped = cv2.warpPerspective(clean_bgr, H_total, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_full = np.full((h, w), 255, dtype=np.uint8)
    page_mask = cv2.warpPerspective(mask_full, H_total, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    cur_img = canvas.copy()
    cur_img[page_mask > 0] = warped[page_mask > 0]
    return cur_img, H_total, corners, page_mask


def _apply_recipe_v1(clean_bgr: np.ndarray, src_idx: int, var_idx: int, sample_rng: np.random.Generator, split: str = "train", split_src_idx: int = 0):
    """Áp dụng công thức biến đổi v1 cho ảnh nguồn.
    
    Hỗ trợ:
    - Train / Val: Phân bố 10 công thức cơ bản và nâng cao, mở rộng góc xoay ±2..±30 độ.
    - Test: Thiết kế chuẩn đáp ứng 100% Lưới D3 (15 ô, 3 mức phối cảnh x 5 góc, mỗi ô 10 ảnh từ 10 nguồn)
            và Tập D4 chỉ xoay (9 góc 0, ±2, ±5, ±10, ±15 độ, mỗi góc 10 ảnh từ 10 nguồn).
    """
    h, w = clean_bgr.shape[:2]
    src_corners = np.float32([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])

    cur_img = clean_bgr.copy()
    H_total = np.eye(3, dtype=np.float64)
    angle_deg = 0.0
    perspective_strength = 0.0
    blur_level = 0
    blur_kind = "none"
    shadow_level = 0
    shadow_type = "none"
    has_shadow = False

    applied = []
    degradation_params = {}

    if split == "test":
        # -------------------------------------------------------------
        # THIẾT KẾ RIÊNG CHO TẬP TEST ĐÁP ỨNG THÍ NGHIỆM D3 & D4 CỦA HIẾU
        # -------------------------------------------------------------
        if var_idx in (0, 1, 2, 3, 4):
            # 1. Lưới D3 (15 ô = 3 mức phối cảnh x 5 góc; mỗi ô đúng 10 ảnh từ 10 nguồn khác nhau)
            persp_levels = [0.05, 0.10, 0.20]
            d3_angles = [0, 5, 10, 20, 30]
            cell_idx = (split_src_idx * 5 + var_idx) % 15
            p_val = persp_levels[cell_idx // 5]
            a_base = d3_angles[cell_idx % 5]
            angle_val = float(a_base)

            cur_img, H_total, corners, page_mask = _apply_perspective_and_rotation(clean_bgr, p_val, angle_val, sample_rng)
            shadow_mask = np.zeros((h, w), dtype=np.uint8)
            perspective_strength = p_val
            angle_deg = angle_val
            applied.append("perspective")
            degradation_params["perspective"] = {"strength": perspective_strength}
            if angle_deg != 0.0:
                applied.append("rotate")
                degradation_params["rotate"] = {"angle_deg": angle_deg}
            recipe_name = f"d3_grid_p{int(p_val*100):02d}_a{a_base:02d}"

        elif var_idx in (5, 6, 7):
            # 2. Tập D4 chỉ xoay (9 góc: 0, ±2, ±5, ±10, ±15; mỗi góc đúng 10 ảnh từ 10 nguồn khác nhau)
            d4_angles = [0, -2, 2, -5, 5, -10, 10, -15, 15]
            ang_idx = (split_src_idx * 3 + (var_idx - 5)) % 9
            angle_val = float(d4_angles[ang_idx])
            angle_deg = angle_val
            cur_img, gt_rot = degrade.rotate(clean_bgr, angle_deg=angle_deg, border_value=(245, 245, 245), rng=sample_rng)
            H_total = gt_rot["H"].astype(np.float64)
            corners = gt_rot["corners"]
            page_mask = gt_rot["page_mask"]
            shadow_mask = np.zeros((h, w), dtype=np.uint8)
            if angle_deg != 0.0:
                applied.append("rotate")
                degradation_params["rotate"] = {"angle_deg": angle_deg}
                recipe_name = f"d4_pure_rotate_a{int(angle_deg):+03d}"
            else:
                recipe_name = "d4_pure_rotate_zero"

        elif var_idx == 8:
            # 3. Thử nghiệm độ bền mờ / bóng độc lập trong test
            corners = src_corners.copy()
            page_mask = np.full((h, w), 255, dtype=np.uint8)
            if split_src_idx % 2 == 0:
                blur_level = int([1, 2, 3, 4][split_src_idx % 4])
                blur_kind = "gaussian" if (split_src_idx % 4 < 2) else "motion"
                cur_img, gt_blur = degrade.blur(cur_img, level=blur_level, rng=sample_rng, kind=blur_kind)
                shadow_mask = np.zeros((h, w), dtype=np.uint8)
                applied.append("blur")
                degradation_params["blur"] = {
                    "level": blur_level,
                    "kind": blur_kind,
                    "ksize": gt_blur.get("ksize", 0),
                    "sigma": gt_blur.get("sigma", 0.0),
                    "radius": gt_blur.get("radius", 0.0),
                    "angle_deg": gt_blur.get("angle_deg", 0.0)
                }
                recipe_name = f"test_single_blur_{blur_kind}"
            else:
                shadow_level = int([1, 2, 3][split_src_idx % 3])
                shadow_type = "planar" if (split_src_idx % 2 == 1) else "silhouette"
                has_shadow = True
                cur_img, gt_shd = degrade.shadow(cur_img, level=shadow_level, rng=sample_rng, shadow_type=shadow_type)
                shadow_mask = gt_shd["shadow_mask"]
                applied.append("shadow")
                degradation_params["shadow"] = {
                    "level": shadow_level,
                    "shadow_type": shadow_type,
                    "darkness": gt_shd.get("darkness", 0.0)
                }
                recipe_name = f"test_single_shadow_{shadow_type}"

        else: # var_idx == 9
            # 4. Combo full v1 trên tập test
            perspective_strength = float([0.05, 0.10, 0.20][split_src_idx % 3])
            angle_base = [2, 5, 10, 15, 20][split_src_idx % 5]
            sign = 1.0 if sample_rng.random() >= 0.5 else -1.0
            angle_deg = float(angle_base * sign)

            cur_img, H_total, corners, page_mask = _apply_perspective_and_rotation(clean_bgr, perspective_strength, angle_deg, sample_rng)
            shadow_level = int([1, 2, 3][split_src_idx % 3])
            shadow_type = "planar"
            has_shadow = True
            cur_img, gt_shd = degrade.shadow(cur_img, level=shadow_level, rng=sample_rng, shadow_type=shadow_type)
            shadow_mask = gt_shd["shadow_mask"]

            blur_level = int([1, 2][split_src_idx % 2])
            blur_kind = "gaussian"
            cur_img, gt_blur = degrade.blur(cur_img, level=blur_level, rng=sample_rng, kind=blur_kind)

            applied.extend(["perspective", "rotate", "shadow", "blur"])
            degradation_params["perspective"] = {"strength": perspective_strength}
            degradation_params["rotate"] = {"angle_deg": angle_deg}
            degradation_params["shadow"] = {"level": shadow_level, "shadow_type": shadow_type, "darkness": gt_shd.get("darkness", 0.0)}
            degradation_params["blur"] = {"level": blur_level, "kind": blur_kind, "sigma": gt_blur.get("sigma", 0.0), "ksize": gt_blur.get("ksize", 1)}
            recipe_name = "combo_full_v1"

    else:
        # -------------------------------------------------------------
        # TẬP TRAIN VÀ VAL: DUY TRÌ 10 RECIPES ĐA DẠNG, BỔ SUNG GÓC ±2 VÀ ±15
        # -------------------------------------------------------------
        recipe_idx = var_idx % 10
        recipe_name = RECIPE_NAMES_V1[recipe_idx]

        if recipe_idx == 0:  # single_rotate_low_mid (bổ sung góc ±2 và ±15)
            angle_base = [2, 5, 10, 15][src_idx % 4]
            sign = 1.0 if sample_rng.random() >= 0.5 else -1.0
            angle_deg = float(angle_base * sign)
            cur_img, gt_rot = degrade.rotate(cur_img, angle_deg=angle_deg, rng=sample_rng)
            H_total = gt_rot["H"].astype(np.float64)
            corners = gt_rot["corners"]
            page_mask = gt_rot["page_mask"]
            shadow_mask = np.zeros((h, w), dtype=np.uint8)
            applied.append("rotate")
            degradation_params["rotate"] = {"angle_deg": angle_deg}

        elif recipe_idx == 1:  # single_rotate_high
            angle_base = [20, 30][src_idx % 2]
            sign = 1.0 if sample_rng.random() >= 0.5 else -1.0
            angle_deg = float(angle_base * sign)
            cur_img, gt_rot = degrade.rotate(cur_img, angle_deg=angle_deg, rng=sample_rng)
            H_total = gt_rot["H"].astype(np.float64)
            corners = gt_rot["corners"]
            page_mask = gt_rot["page_mask"]
            shadow_mask = np.zeros((h, w), dtype=np.uint8)
            applied.append("rotate")
            degradation_params["rotate"] = {"angle_deg": angle_deg}

        elif recipe_idx == 2:  # single_perspective
            perspective_strength = float([0.05, 0.10, 0.20][src_idx % 3])
            cur_img, gt_persp = degrade.perspective(cur_img, strength=perspective_strength, rng=sample_rng)
            H_total = gt_persp["H"].astype(np.float64)
            corners = gt_persp["corners"]
            page_mask = gt_persp["page_mask"]
            shadow_mask = np.zeros((h, w), dtype=np.uint8)
            applied.append("perspective")
            degradation_params["perspective"] = {"strength": perspective_strength}

        elif recipe_idx == 3:  # single_blur_gaussian
            blur_level = int([1, 2, 3, 4][src_idx % 4])
            blur_kind = "gaussian"
            cur_img, gt_blur = degrade.blur(cur_img, level=blur_level, rng=sample_rng, kind=blur_kind)
            corners = src_corners.copy()
            page_mask = np.full((h, w), 255, dtype=np.uint8)
            shadow_mask = np.zeros((h, w), dtype=np.uint8)
            applied.append("blur")
            degradation_params["blur"] = {
                "level": blur_level,
                "kind": blur_kind,
                "ksize": gt_blur.get("ksize", 0),
                "sigma": gt_blur.get("sigma", 0.0)
            }

        elif recipe_idx == 4:  # single_blur_motion
            blur_level = int([1, 2, 3, 4][(src_idx + 1) % 4])
            blur_kind = "motion"
            cur_img, gt_blur = degrade.blur(cur_img, level=blur_level, rng=sample_rng, kind=blur_kind)
            corners = src_corners.copy()
            page_mask = np.full((h, w), 255, dtype=np.uint8)
            shadow_mask = np.zeros((h, w), dtype=np.uint8)
            applied.append("blur")
            degradation_params["blur"] = {
                "level": blur_level,
                "kind": blur_kind,
                "radius": gt_blur.get("radius", 0.0),
                "angle_deg": gt_blur.get("angle_deg", 0.0),
                "ksize": gt_blur.get("ksize", 1)
            }

        elif recipe_idx == 5:  # single_shadow
            shadow_level = int([1, 2, 3][src_idx % 3])
            shadow_type = "planar" if (src_idx % 2 == 0) else "silhouette"
            has_shadow = True
            cur_img, gt_shd = degrade.shadow(cur_img, level=shadow_level, rng=sample_rng, shadow_type=shadow_type)
            corners = src_corners.copy()
            page_mask = np.full((h, w), 255, dtype=np.uint8)
            shadow_mask = gt_shd["shadow_mask"]
            applied.append("shadow")
            degradation_params["shadow"] = {
                "level": shadow_level,
                "shadow_type": shadow_type,
                "darkness": gt_shd.get("darkness", 0.0)
            }

        elif recipe_idx == 6:  # combo_rotate_shadow (bổ sung góc ±2 và ±15)
            angle_base = [2, 5, 10, 15, 20][src_idx % 5]
            sign = 1.0 if sample_rng.random() >= 0.5 else -1.0
            angle_deg = float(angle_base * sign)
            cur_img, gt_rot = degrade.rotate(cur_img, angle_deg=angle_deg, rng=sample_rng)

            shadow_level = int([1, 2, 3][(src_idx + 1) % 3])
            shadow_type = "planar"
            has_shadow = True
            cur_img, gt_shd = degrade.shadow(cur_img, level=shadow_level, rng=sample_rng, shadow_type=shadow_type)

            H_total = gt_rot["H"].astype(np.float64)
            corners = gt_rot["corners"]
            page_mask = gt_rot["page_mask"]
            shadow_mask = gt_shd["shadow_mask"]
            applied.extend(["rotate", "shadow"])
            degradation_params["rotate"] = {"angle_deg": angle_deg}
            degradation_params["shadow"] = {
                "level": shadow_level,
                "shadow_type": shadow_type,
                "darkness": gt_shd.get("darkness", 0.0)
            }

        elif recipe_idx == 7:  # combo_perspective_blur
            perspective_strength = float([0.05, 0.10, 0.20][(src_idx + 1) % 3])
            cur_img, gt_persp = degrade.perspective(cur_img, strength=perspective_strength, rng=sample_rng)

            blur_level = int([1, 2, 3][src_idx % 3])
            blur_kind = "gaussian" if (src_idx % 2 == 0) else "motion"
            cur_img, gt_blur = degrade.blur(cur_img, level=blur_level, rng=sample_rng, kind=blur_kind)

            H_total = gt_persp["H"].astype(np.float64)
            corners = gt_persp["corners"]
            page_mask = gt_persp["page_mask"]
            shadow_mask = np.zeros((h, w), dtype=np.uint8)
            applied.extend(["perspective", "blur"])
            degradation_params["perspective"] = {"strength": perspective_strength}
            degradation_params["blur"] = {
                "level": blur_level,
                "kind": blur_kind,
                "radius": gt_blur.get("radius", 0.0),
                "angle_deg": gt_blur.get("angle_deg", 0.0),
                "ksize": gt_blur.get("ksize", 1),
                "sigma": gt_blur.get("sigma", 0.0)
            }

        elif recipe_idx == 8:  # combo_perspective_shadow
            perspective_strength = float([0.05, 0.10, 0.20][(src_idx + 2) % 3])
            cur_img, gt_persp = degrade.perspective(cur_img, strength=perspective_strength, rng=sample_rng)

            shadow_level = int([1, 2, 3][(src_idx + 2) % 3])
            shadow_type = "silhouette" if (src_idx % 2 == 0) else "planar"
            has_shadow = True
            cur_img, gt_shd = degrade.shadow(cur_img, level=shadow_level, rng=sample_rng, shadow_type=shadow_type)

            H_total = gt_persp["H"].astype(np.float64)
            corners = gt_persp["corners"]
            page_mask = gt_persp["page_mask"]
            shadow_mask = gt_shd["shadow_mask"]
            applied.extend(["perspective", "shadow"])
            degradation_params["perspective"] = {"strength": perspective_strength}
            degradation_params["shadow"] = {
                "level": shadow_level,
                "shadow_type": shadow_type,
                "darkness": gt_shd.get("darkness", 0.0)
            }

        else:  # combo_full_v1 (var_09)
            perspective_strength = float([0.05, 0.10][src_idx % 2])
            angle_base = [5, 10][src_idx % 2]
            sign = 1.0 if sample_rng.random() >= 0.5 else -1.0
            angle_deg = float(angle_base * sign)

            cur_img, H_total, corners, page_mask = _apply_perspective_and_rotation(clean_bgr, perspective_strength, angle_deg, sample_rng)

            shadow_level = int([1, 2, 3][src_idx % 3])
            shadow_type = "planar"
            has_shadow = True
            cur_img, gt_shd = degrade.shadow(cur_img, level=shadow_level, rng=sample_rng, shadow_type=shadow_type)

            blur_level = int([1, 2][src_idx % 2])
            blur_kind = "gaussian"
            cur_img, gt_blur = degrade.blur(cur_img, level=blur_level, rng=sample_rng, kind=blur_kind)

            shadow_mask = gt_shd["shadow_mask"]
            applied.extend(["perspective", "rotate", "shadow", "blur"])
            degradation_params["perspective"] = {"strength": perspective_strength}
            degradation_params["rotate"] = {"angle_deg": angle_deg}
            degradation_params["shadow"] = {"level": shadow_level, "shadow_type": shadow_type, "darkness": gt_shd.get("darkness", 0.0)}
            degradation_params["blur"] = {"level": blur_level, "kind": blur_kind, "sigma": gt_blur.get("sigma", 0.0), "ksize": gt_blur.get("ksize", 1)}

    corners_list = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in corners]
    H_clean_to_deg = H_total.tolist()
    H_deg_to_clean = np.linalg.inv(H_total).tolist()

    return {
        "recipe_name": recipe_name,
        "degraded_img": cur_img,
        "H_total": H_total,
        "H_clean_to_degraded": H_clean_to_deg,
        "H_degraded_to_clean": H_deg_to_clean,
        "corners": corners_list,
        "angle_deg": angle_deg,
        "perspective_strength": perspective_strength,
        "blur_level": blur_level,
        "blur_kind": blur_kind,
        "shadow_level": shadow_level,
        "shadow_type": shadow_type,
        "has_shadow": has_shadow,
        "page_mask": page_mask,
        "shadow_mask": shadow_mask,
        "applied_degradations": applied,
        "degradation_params": degradation_params
    }


# ─────────────────────────────────────── v2 ───────────────────────────────────────
# v2 KHÔNG sinh lại v00–v09: mỗi ảnh gốc thêm 5 biến thể v10–v14, giữ nguyên toàn bộ file và manifest của v1.
RECIPE_NAMES_V2 = {
    10: "single_glare",        # chỉ lóa, mức 1..3
    11: "single_stamp",        # chỉ dấu đỏ, mức 1..3
    12: "single_pen",          # chỉ nét bút xanh / đen, mức 1..2
    13: "single_jpeg_noise",   # chỉ nén JPEG + nhiễu, mức 1..3
    14: "combo_full_v2",       # dấu + bút → phối cảnh → xoay → bóng → lóa → mờ → nén + nhiễu
}
V2_MASKS = ("glare_mask", "stamp_mask", "pen_mask")
GLARE_VARIANTS = (10, 14)   # công thức có lóa: giảm phơi sáng trước khi thêm lóa, lưu thêm clean_exposed/


def _warp_mask(mask: np.ndarray, H: np.ndarray, page_mask: np.ndarray) -> np.ndarray:
    """Đưa mask vẽ trên ảnh sạch sang hệ tọa độ ảnh hỏng (cùng H với ảnh), cắt ngoài tờ giấy."""
    h, w = mask.shape[:2]
    warped = cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped[page_mask == 0] = 0
    return warped


def _bbox_of(box_dict_or_list) -> list:
    return box_dict_or_list["bbox"] if isinstance(box_dict_or_list, dict) else box_dict_or_list


def _apply_recipe_v2(clean_bgr: np.ndarray, src_idx: int, var_idx: int, sample_rng: np.random.Generator,
                     split_src_idx: int = 0, med_region=None, exposure_gain: float = 1.0):
    """Công thức v10–v14. Mức xoay vòng theo thứ tự ảnh gốc trong split để mọi mức đều có ở train / val / test.

    Thứ tự vật lý: dấu, bút nằm trên giấy (áp lên ảnh sạch, mask đi qua cùng H với ảnh)
    → phối cảnh / xoay → bóng → lóa (đèn flash) → mờ (máy rung) → nén JPEG + nhiễu (lưu ảnh).
    exposure_gain (chỉ v10, v14): nhân ảnh sau dấu, bút và trước phối cảnh (nền bàn giữ màu), để giấy tối hơn 255
    và lóa sáng hơn mặt giấy như ảnh chụp. Không dùng sample_rng → mask, dấu, bút giống hệt khi gain = 1.
    """
    h, w = clean_bgr.shape[:2]
    k = split_src_idx
    src_corners = np.float32([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])
    zeros = np.zeros((h, w), np.uint8)
    region = med_region if med_region is not None else [0, 0, w, h]

    levels = {"perspective": 0.0, "rotate": 0.0, "blur": 0, "shadow": 0, "glare": 0, "stamp": 0, "pen": 0, "jpeg_noise": 0}
    params, applied = {}, []
    masks = {"shadow_mask": zeros, "glare_mask": zeros, "stamp_mask": zeros, "pen_mask": zeros}
    cur = clean_bgr.copy()
    H_total = np.eye(3, dtype=np.float64)
    corners = src_corners.copy()
    page_mask = np.full((h, w), 255, np.uint8)
    blur_kind, shadow_type, pen_color = "none", "none", "none"
    stamp_clean = zeros   # mask dấu trên ảnh sạch, để tính tỉ lệ chữ bị che bằng clean_binary

    def do_stamp(img, lv):
        out, gt = degrade.stamp(img, lv, sample_rng, region=region)
        levels["stamp"] = lv
        params["stamp"] = {key: gt[key] for key in ("level", "center", "radius", "opacity", "rotation_deg", "color_bgr", "text")}
        applied.append("stamp")
        return out, gt["stamp_mask"]

    def do_pen(img, lv, color):
        out, gt = degrade.pen(img, lv, sample_rng, color=color, region=region)
        levels["pen"] = lv
        params["pen"] = {key: gt[key] for key in ("level", "pen_color", "color_bgr", "n_strokes", "thickness")}
        applied.append("pen")
        return out, gt["pen_mask"]

    def do_glare(img, lv, reg, page):
        out, gt = degrade.glare(img, lv, sample_rng, region=reg)
        out = np.where(page[..., None] > 0, out, img) if img.ndim == 3 else np.where(page > 0, out, img)
        gt["glare_mask"][page == 0] = 0   # lóa là phản xạ trên tờ giấy: không tính nền bàn
        gt["glare_area_ratio"] = round(float((gt["glare_mask"] > 0).mean()), 5)
        levels["glare"] = lv
        params["glare"] = {key: gt[key] for key in ("level", "center", "sigma_xy", "angle_deg", "gain",
                                                     "glare_area_ratio", "saturated_ratio")}
        applied.append("glare")
        return out, gt["glare_mask"]

    def do_exposure(img):
        params["exposure"] = {"gain": exposure_gain}
        return degrade.exposure(img, exposure_gain) if exposure_gain != 1.0 else img

    def do_jpeg(img, lv):
        out, gt = degrade.jpeg_noise(img, lv, sample_rng)
        levels["jpeg_noise"] = lv
        params["jpeg_noise"] = gt
        applied.append("jpeg_noise")
        return out

    recipe_name = RECIPE_NAMES_V2[var_idx]
    if var_idx == 10:
        cur = do_exposure(cur)
        cur, masks["glare_mask"] = do_glare(cur, [1, 2, 3][k % 3], region, page_mask)
    elif var_idx == 11:
        cur, stamp_clean = do_stamp(cur, [1, 2, 3][k % 3])
        masks["stamp_mask"] = stamp_clean
    elif var_idx == 12:
        pen_color = ["blue", "black"][(k // 2) % 2]
        cur, masks["pen_mask"] = do_pen(cur, [1, 2][k % 2], pen_color)
    elif var_idx == 13:
        cur = do_jpeg(cur, [1, 2, 3][k % 3])
    elif var_idx == 14:
        cur, stamp_clean = do_stamp(cur, [1, 2, 3][(k + 1) % 3])
        pen_color = ["blue", "black"][((k // 2) + 1) % 2]
        cur, pen_clean = do_pen(cur, [1, 2][(k + 1) % 2], pen_color)
        cur = do_exposure(cur)

        p_strength = float([0.05, 0.10][k % 2])
        angle_deg = float([5, 10][k % 2] * (1.0 if sample_rng.random() >= 0.5 else -1.0))
        cur, H_total, corners, page_mask = _apply_perspective_and_rotation(cur, p_strength, angle_deg, sample_rng)
        masks["stamp_mask"] = _warp_mask(stamp_clean, H_total, page_mask)
        masks["pen_mask"] = _warp_mask(pen_clean, H_total, page_mask)
        levels["perspective"], levels["rotate"] = p_strength, angle_deg
        params["perspective"] = {"strength": p_strength}
        params["rotate"] = {"angle_deg": angle_deg}
        applied.extend(["perspective", "rotate"])

        shadow_type = "planar"
        cur, gt_shd = degrade.shadow(cur, level=[1, 2, 3][(k + 2) % 3], rng=sample_rng, shadow_type=shadow_type)
        masks["shadow_mask"] = gt_shd["shadow_mask"]
        levels["shadow"] = gt_shd["level"]
        params["shadow"] = {"level": gt_shd["level"], "shadow_type": shadow_type, "darkness": gt_shd["darkness"]}
        applied.append("shadow")

        reg_deg = degrade.transform_box(region, H_total, img_shape=(h, w))["bbox"]
        cur, masks["glare_mask"] = do_glare(cur, [1, 2, 3][(k + 2) % 3], reg_deg, page_mask)

        blur_kind = "gaussian"
        cur, gt_blur = degrade.blur(cur, level=[1, 2][k % 2], rng=sample_rng, kind=blur_kind)
        levels["blur"] = int(gt_blur["level"])
        params["blur"] = {"level": int(gt_blur["level"]), "kind": blur_kind, "sigma": gt_blur["sigma"], "ksize": gt_blur["ksize"]}
        applied.append("blur")

        cur = do_jpeg(cur, [1, 2, 3][(k + 1) % 3])
    else:
        raise ValueError(f"var_idx v2 phải thuộc 10..14, nhận {var_idx}")

    return {
        "recipe_name": recipe_name,
        "degraded_img": cur,
        "H_total": H_total,
        "H_clean_to_degraded": H_total.tolist(),
        "H_degraded_to_clean": np.linalg.inv(H_total).tolist(),
        "corners": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in corners],
        "levels": levels,
        "blur_kind": blur_kind,
        "shadow_type": shadow_type,
        "pen_color": pen_color,
        "page_mask": page_mask,
        "masks": masks,
        "stamp_mask_clean": stamp_clean,
        "applied_degradations": applied,
        "degradation_params": params,
    }


def _load_raw_boxes(src_info: dict) -> list:
    """Nhãn dòng gốc của ảnh nguồn (VAIPE: list {"box": [x1, y1, x2, y2], "label", "text", ...}) → bbox [x, y, w, h]."""
    raw_boxes = []
    lbl_rel = src_info.get("label_path", "")
    if not lbl_rel:
        return raw_boxes
    lbl_full = REPO_ROOT / lbl_rel if not Path(lbl_rel).is_absolute() else Path(lbl_rel)
    if not lbl_full.exists():
        return raw_boxes
    try:
        with open(lbl_full, "r", encoding="utf-8") as lf:
            v_items = json.load(lf)
        for item in v_items:
            box = item.get("box", [])
            if len(box) == 4:
                x1, y1, x2, y2 = box
                raw_boxes.append({
                    "id": item.get("id", len(raw_boxes) + 1),
                    "label": item.get("label", "other"),
                    "text": item.get("text", ""),
                    "mapping": item.get("mapping", None),
                    "bbox": [int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))]
                })
    except Exception as e:
        print(f"Lỗi đọc nhãn {lbl_full}: {e}")
    return raw_boxes


def transform_items(items_clean: list, H: np.ndarray, img_shape: tuple) -> list:
    """GT khoản thuốc trên ảnh sạch → ảnh hỏng qua H (cùng định dạng items_degraded của v1, kiểm khớp 1.000/1.000 mẫu)."""
    out = []
    for it in items_clean:
        d = {"item_id": it["item_id"], "name": it.get("name"), "fields_gt": it.get("fields_gt")}
        d.update(degrade.transform_box(it["bbox"], H, img_shape=img_shape))
        lines = []
        for lb in it.get("line_boxes", []):
            t = degrade.transform_box(lb["bbox"], H, img_shape=img_shape)
            t.pop("visible_ratio")
            lines.append({"line_id": lb["line_id"], "text": lb.get("text"), "label": lb.get("label"), **t})
        d["line_boxes"] = lines
        out.append(d)
    return out


def _generate_v2(sources: list, out_dir: Path, v1_dir: Path, seed: int, start_time: float) -> dict:
    """Sinh v10–v14 cho mọi ảnh gốc vào cùng cây thư mục với v1; ghi manifest_v2 (= hàng v1 + hàng mới)."""
    manifest_v1_csv = v1_dir / "manifest_v1.csv"
    if not manifest_v1_csv.exists():
        raise FileNotFoundError(f"v2 cần Bộ 5 v1 có sẵn (thiếu {manifest_v1_csv}). Sinh v1 trước: --version v1")

    new_rows = []
    split_counts = defaultdict(int)
    split_src_counts = defaultdict(int)
    for src_idx, src_info in enumerate(sources):   # cùng thứ tự với v1 → cùng src_idx, cùng split_src_idx
        split = src_info["split"]
        split_src_idx = split_src_counts[split]
        split_src_counts[split] += 1
        img_rel = src_info["image_path"]
        img_full = REPO_ROOT / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
        clean_bgr = cv2.imread(str(img_full)) if img_full.exists() else None
        if clean_bgr is None:
            print(f"Cảnh báo: Không đọc được ảnh nguồn: {img_full}, bỏ qua.")
            continue
        h, w = clean_bgr.shape[:2]
        clean_bin = _make_clean_binary(clean_bgr)
        raw_boxes = _load_raw_boxes(src_info)
        med_region_clean = src_info.get("medicine_region_bbox", [0, 0, w, h])
        stem = Path(img_rel).stem

        # GT khoản thuốc + cờ độ phủ lấy từ nhãn v00 của chính ảnh gốc này
        v00_label = v1_dir / split / "labels" / f"{stem}_v00.json"
        items_clean, coverage = [], {"has_page_gt": True, "has_medicine_region_gt": True,
                                     "has_line_boxes_gt": len(raw_boxes) > 0, "has_verified_item_gt": False, "has_item_gt": False}
        if v00_label.exists():
            with open(v00_label, "r", encoding="utf-8") as f:
                v00 = json.load(f)
            items_clean = v00.get("items_clean", [])
            coverage = v00.get("gt_coverage", coverage)

        split_dir = out_dir / split
        dirs = {name: split_dir / name for name in ("images", "labels", "clean", "clean_exposed", "clean_binary",
                                                    "shadow_mask", "page_mask") + V2_MASKS}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        for var_idx in RECIPE_NAMES_V2:
            sample_id = f"{stem}_v{var_idx:02d}"
            sample_rng = np.random.default_rng([seed, src_idx, var_idx])
            exposure_gain = 1.0
            if var_idx in GLARE_VARIANTS:   # RNG riêng: không làm lệch chuỗi số của sample_rng (vị trí lóa, dấu, bút giữ nguyên)
                exposure_gain = round(float(np.random.default_rng([seed, src_idx, var_idx, 1]).uniform(
                    *degrade.EXPOSURE_GAIN_RANGE)), 4)
            res = _apply_recipe_v2(clean_bgr, src_idx, var_idx, sample_rng, split_src_idx=split_src_idx,
                                   med_region=med_region_clean, exposure_gain=exposure_gain)
            H_total = res["H_total"]
            lv = res["levels"]

            cv2.imwrite(str(dirs["images"] / f"{sample_id}.png"), res["degraded_img"])
            cv2.imwrite(str(dirs["clean"] / f"{sample_id}.png"), clean_bgr)            # ảnh sạch: không dấu, không bút
            rel_clean_exposed = ""
            if var_idx in GLARE_VARIANTS:   # ảnh sạch cùng mức phơi sáng với ảnh hỏng — tham chiếu PSNR / SSIM cho D5
                cv2.imwrite(str(dirs["clean_exposed"] / f"{sample_id}.png"), degrade.exposure(clean_bgr, exposure_gain))
                rel_clean_exposed = f"{split}/clean_exposed/{sample_id}.png"
            cv2.imwrite(str(dirs["clean_binary"] / f"{sample_id}.png"), clean_bin)
            cv2.imwrite(str(dirs["page_mask"] / f"{sample_id}.png"), res["page_mask"])
            for name in ("shadow_mask",) + V2_MASKS:
                cv2.imwrite(str(dirs[name] / f"{sample_id}.png"), res["masks"][name])

            stamp_clean = res["stamp_mask_clean"]
            ink = clean_bin > 0
            if "stamp" in res["degradation_params"]:
                res["degradation_params"]["stamp"]["text_covered_ratio"] = round(
                    float((ink & (stamp_clean > 0)).sum()) / max(1, int(ink.sum())), 4)

            boxes_degraded = degrade.transform_boxes(raw_boxes, H_total, img_shape=(h, w))
            n_clipped = sum(1 for b in boxes_degraded if b.get("clipped", False))
            items_degraded = transform_items(items_clean, H_total, (h, w))
            has = {name: bool((res["masks"][name] > 0).any()) for name in ("shadow_mask",) + V2_MASKS}
            gt_dict = {
                "sample_id": sample_id,
                "source_id": src_info["source_id"],
                "split": split,
                "version": "v2",
                "variant_index": var_idx,
                "recipe_name": res["recipe_name"],
                "seed_tuple": [seed, src_idx, var_idx],
                "image_size": [int(w), int(h)],
                "applied_degradations": res["applied_degradations"],
                "degradation_levels": lv,            # mức của cả 8 loại hỏng, kể cả 0
                "H": res["H_clean_to_degraded"],
                "H_clean_to_degraded": res["H_clean_to_degraded"],
                "H_degraded_to_clean": res["H_degraded_to_clean"],
                "corners": res["corners"],
                "angle_deg": lv["rotate"],
                "exposure_gain": exposure_gain,      # 1.0 = không đổi phơi sáng (v11–v13)
                "clean_exposed": f"clean_exposed/{sample_id}.png" if rel_clean_exposed else None,
                "degradation_params": res["degradation_params"],
                "masks": {"page_mask": f"page_mask/{sample_id}.png",
                          **{name: f"{name}/{sample_id}.png" for name in ("shadow_mask",) + V2_MASKS},
                          "has_shadow": has["shadow_mask"], "has_glare": has["glare_mask"],
                          "has_stamp": has["stamp_mask"], "has_pen": has["pen_mask"]},
                "active_masks": [name for name in ("shadow_mask",) + V2_MASKS if has[name]],
                "gt_coverage": coverage,
                "medicine_region_clean": med_region_clean,
                "medicine_region_degraded": degrade.transform_box(med_region_clean, H_total, img_shape=(h, w)),
                "n_boxes_total": len(boxes_degraded),
                "n_boxes_clipped": n_clipped,
                "boxes_clean": raw_boxes,
                "boxes_degraded": boxes_degraded,
                "items_clean": items_clean,
                "items_degraded": items_degraded,
                "n_items": len(items_clean),
            }
            with open(dirs["labels"] / f"{sample_id}.json", "w", encoding="utf-8") as gf:
                json.dump(gt_dict, gf, indent=2, ensure_ascii=False)

            params = res["degradation_params"]
            new_rows.append({
                "sample_id": sample_id, "source_id": src_info["source_id"], "split": split, "variant_index": var_idx,
                "recipe_name": res["recipe_name"],
                "rel_degraded_path": f"{split}/images/{sample_id}.png", "rel_clean_path": f"{split}/clean/{sample_id}.png",
                "rel_clean_binary_path": f"{split}/clean_binary/{sample_id}.png", "rel_gt_path": f"{split}/labels/{sample_id}.json",
                "width": w, "height": h, "applied_degradations": ",".join(res["applied_degradations"]),
                "angle_deg": lv["rotate"], "perspective_strength": lv["perspective"], "blur_level": lv["blur"],
                "blur_kind": res["blur_kind"], "shadow_level": lv["shadow"], "shadow_type": res["shadow_type"],
                "has_shadow": has["shadow_mask"], "n_boxes_total": len(boxes_degraded), "n_boxes_clipped": n_clipped,
                "n_items": len(items_clean), "has_item_gt": bool(coverage.get("has_item_gt", False)),
                "version_added": "v2",
                "glare_level": lv["glare"], "stamp_level": lv["stamp"], "pen_level": lv["pen"], "pen_color": res["pen_color"],
                "jpeg_noise_level": lv["jpeg_noise"],
                "jpeg_quality": params.get("jpeg_noise", {}).get("jpeg_quality", 100),
                "noise_sigma": params.get("jpeg_noise", {}).get("noise_sigma", 0.0),
                "has_glare": has["glare_mask"], "has_stamp": has["stamp_mask"], "has_pen": has["pen_mask"],
                "glare_area_ratio": params.get("glare", {}).get("glare_area_ratio", 0.0),
                "stamp_text_covered_ratio": params.get("stamp", {}).get("text_covered_ratio", 0.0),
                **{f"rel_{name}_path": f"{split}/{name}/{sample_id}.png" for name in V2_MASKS},
                "exposure_gain": exposure_gain, "rel_clean_exposed_path": rel_clean_exposed,
            })
            split_counts[split] += 1

    # manifest_v2 = toàn bộ hàng v1 (thêm cột v2 = 0 / rỗng) + hàng mới; manifest_v1 giữ nguyên
    v1_df = pd.read_csv(manifest_v1_csv)
    if len(sources) < v1_df["source_id"].nunique():   # chạy thử với --limit: chỉ giữ hàng v1 của các nguồn đã sinh
        v1_df = v1_df[v1_df["source_id"].isin({s["source_id"] for s in sources})]
    v1_df = v1_df.assign(version_added="v1", glare_level=0, stamp_level=0, pen_level=0, pen_color="none",
                         jpeg_noise_level=0, jpeg_quality=100, noise_sigma=0.0, has_glare=False, has_stamp=False,
                         has_pen=False, glare_area_ratio=0.0, stamp_text_covered_ratio=0.0,
                         **{f"rel_{name}_path": "" for name in V2_MASKS},
                         exposure_gain=1.0, rel_clean_exposed_path="")
    manifest_df = pd.concat([v1_df, pd.DataFrame(new_rows)], ignore_index=True)
    manifest_df = manifest_df.sort_values(["split", "source_id", "variant_index"]).reset_index(drop=True)
    manifest_csv_path = out_dir / "manifest_v2.csv"
    manifest_json_path = out_dir / "manifest_v2.json"
    manifest_df.to_csv(manifest_csv_path, index=False, encoding="utf-8")
    with open(manifest_json_path, "w", encoding="utf-8") as mf:
        json.dump({"dataset_name": "Synthetic Benchmark Set 6 (V2)", "version": "v2", "seed": seed,
                   "n_sources": len(sources), "v2_variants": RECIPE_NAMES_V2,
                   "exposure_gain_range": list(degrade.EXPOSURE_GAIN_RANGE),
                   "exposure_variants": list(GLARE_VARIANTS),
                   "total_samples": len(manifest_df), "new_samples": len(new_rows),
                   "split_counts": manifest_df["split"].value_counts().to_dict(),
                   "samples": json.loads(manifest_df.to_json(orient="records", force_ascii=False))},
                  mf, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"Hoàn thành sinh {len(new_rows)} mẫu v2 mới vào {out_dir} sau {elapsed:.2f}s (manifest_v2: {len(manifest_df)} mẫu)")
    return {
        "status": "SUCCESS",
        "version": "v2",
        "total_samples": len(manifest_df),
        "new_samples": len(new_rows),
        "split_counts": dict(split_counts),
        "n_sources": len(sources),
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_json_path),
        "elapsed_seconds": round(elapsed, 2),
    }


def _make_clean_binary(clean_bgr: np.ndarray) -> np.ndarray:
    """Tạo ảnh nhị phân Otsu chuẩn trên ảnh sạch: Mực = 255, Nền = 0."""
    clean_gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY) if clean_bgr.ndim == 3 else clean_bgr
    _, clean_bin = cv2.threshold(clean_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Nếu tỷ lệ pixel 255 > 50% (ngược nền) thì đảo lại để luôn đảm bảo mực = 255, nền = 0
    if np.mean(clean_bin == 255) > 0.50:
        clean_bin = 255 - clean_bin
    return clean_bin


def _load_sources_from_manifest(manifest_path: Path, limit: int = None) -> list[dict]:
    """Nạp danh sách ảnh nguồn sạch từ file manifest JSON của T03."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sources = data.get("sources", [])
    if limit is not None and limit > 0:
        sources = sources[:limit]
    return sources


def _scan_sources_from_dir(src_dir: Path, cfg: dict, limit: int = None) -> list[dict]:
    """Quét ảnh nguồn từ một thư mục tùy ý (dành cho unit test hoặc custom dataset)."""
    valid_exts = {".png", ".jpg", ".jpeg"}
    img_files = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in valid_exts])

    n_sources = cfg.get("n_source_images", 100)
    if limit is not None and limit > 0:
        n_sources = min(n_sources, limit)
    selected_files = img_files[:n_sources]

    # Phân tập tất định bằng seed
    seed = cfg.get("seed", 42)
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(selected_files))

    split_cfg = cfg.get("split", {"train": 0.6, "val": 0.1, "test": 0.3})
    n_total = len(selected_files)
    n_train = int(round(n_total * split_cfg.get("train", 0.6)))
    n_val = int(round(n_total * split_cfg.get("val", 0.1)))

    sources = []
    for rank, orig_idx in enumerate(shuffled_indices):
        f = selected_files[orig_idx]
        if rank < n_train:
            s_split = "train"
        elif rank < n_train + n_val:
            s_split = "val"
        else:
            s_split = "test"

        # Đọc ảnh để lấy kích thước
        bgr = cv2.imread(str(f))
        h, w = (bgr.shape[:2]) if bgr is not None else (100, 100)

        # Kiểm tra nếu có file json nhãn cùng tên
        lbl_f = f.with_suffix(".json")
        if lbl_f.exists():
            try:
                lbl_rel = str(lbl_f.relative_to(REPO_ROOT)).replace("\\", "/")
            except ValueError:
                lbl_rel = str(lbl_f)
        else:
            lbl_rel = ""

        try:
            img_rel_val = str(f.relative_to(REPO_ROOT)).replace("\\", "/")
        except ValueError:
            img_rel_val = str(f)

        sources.append({
            "source_id": f"src_{f.stem}",
            "split": s_split,
            "origin_dataset": "custom",
            "image_path": img_rel_val,
            "label_path": lbl_rel,
            "width": w,
            "height": h,
            "medicine_region_bbox": [0, 0, w, h]
        })
    return sources


def generate(
    config_path: Path = None,
    src_dir: Path = None,
    out_dir: Path = None,
    version: str = "v1",
    sources_manifest: Path = None,
    limit_sources: int = None,
    v1_dir: Path = None,
) -> dict:
    """Hàm chính sinh bộ 5 benchmark tổng hợp.

    Args:
        config_path: Đường dẫn file config.yaml (mặc định lấy config.yaml nội bộ).
        src_dir: Thư mục ảnh sạch (nếu None, ưu tiên dùng sources_manifest hoặc results/t03/).
        out_dir: Thư mục đích ra (mặc định: data/5_synthetic).
        version: Phiên bản bộ sinh ("v1" hoặc "v2").
        sources_manifest: Đường dẫn file manifest JSON nguồn sạch (mặc định: results/t03/synthetic_100_sources.json).
        limit_sources: Giới hạn số lượng ảnh nguồn (tiện cho unit test hoặc pilot).
        v1_dir: (chỉ v2) thư mục Bộ 5 v1 để đọc manifest_v1 và GT khoản thuốc của v00 (mặc định = out_dir).
            v2 chỉ sinh thêm v10–v14, không ghi lại file nào của v1.

    Returns:
        dict tóm tắt kết quả sinh (tổng số mẫu, phân bố split, đường dẫn lưu).
    """
    start_time = time.time()

    # 1. Nạp config
    if config_path is None:
        config_path = Path(__file__).with_name("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config tại: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    variants_per_image = cfg.get("variants_per_image", 10)

    # 2. Xác định thư mục đầu ra
    if out_dir is None:
        out_dir = REPO_ROOT / "data" / "5_synthetic"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Nạp danh sách ảnh nguồn sạch
    if src_dir is not None and Path(src_dir).exists():
        sources = _scan_sources_from_dir(Path(src_dir), cfg, limit=limit_sources)
    else:
        if sources_manifest is None:
            sources_manifest = REPO_ROOT / "results" / "t03" / "synthetic_100_sources.json"
        sources_manifest = Path(sources_manifest)
        if sources_manifest.exists():
            sources = _load_sources_from_manifest(sources_manifest, limit=limit_sources)
        else:
            raise FileNotFoundError(
                f"Không tìm thấy manifest nguồn sạch tại: {sources_manifest}. Hãy chạy T03 trước hoặc truyền src_dir!"
            )

    if version == "v2":
        print(f"Bắt đầu sinh Bộ 5 V2: {len(sources)} ảnh nguồn x {len(RECIPE_NAMES_V2)} biến thể mới (v10–v14), giữ nguyên v1...")
        return _generate_v2(sources, out_dir, Path(v1_dir) if v1_dir is not None else out_dir, seed, start_time)

    print(f"Bắt đầu sinh Bộ 5 {version.upper()}: {len(sources)} ảnh nguồn x {variants_per_image} biến thể...")

    all_samples_meta = []
    split_counts = defaultdict(int)
    split_src_counts = defaultdict(int)

    for src_idx, src_info in enumerate(sources):
        src_id = src_info["source_id"]
        split = src_info["split"]
        split_src_idx = split_src_counts[split]
        split_src_counts[split] += 1
        img_rel = src_info["image_path"]
        img_full = REPO_ROOT / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)

        if not img_full.exists():
            print(f"Cảnh báo: Không tìm thấy ảnh nguồn: {img_full}, bỏ qua.")
            continue

        clean_bgr = cv2.imread(str(img_full))
        if clean_bgr is None:
            print(f"Cảnh báo: Lỗi đọc ảnh: {img_full}, bỏ qua.")
            continue

        h, w = clean_bgr.shape[:2]
        clean_bin = _make_clean_binary(clean_bgr)

        # Nạp các box gốc nếu có
        raw_boxes = _load_raw_boxes(src_info)

        med_region_clean = src_info.get("medicine_region_bbox", [0, 0, w, h])

        # Tạo thư mục chuẩn theo split: images/, labels/, clean/, clean_binary/, shadow_mask/, page_mask/
        split_dir = out_dir / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        clean_dir = split_dir / "clean"
        clean_bin_dir = split_dir / "clean_binary"
        shadow_dir = split_dir / "shadow_mask"
        page_dir = split_dir / "page_mask"

        for d in (img_dir, lbl_dir, clean_dir, clean_bin_dir, shadow_dir, page_dir):
            d.mkdir(parents=True, exist_ok=True)

        stem = Path(img_rel).stem
        v00_clean_path = None
        v00_clean_bin_path = None

        for var_idx in range(variants_per_image):
            sample_id = f"{stem}_v{var_idx:02d}"

            # Khởi tạo RNG độc lập tất định cho từng mẫu
            sample_rng = np.random.default_rng([seed, src_idx, var_idx])

            # Áp dụng công thức làm hỏng
            res = _apply_recipe_v1(clean_bgr, src_idx, var_idx, sample_rng, split=split, split_src_idx=split_src_idx)

            degraded_img = res["degraded_img"]
            H_total = res["H_total"]
            page_mask = res["page_mask"]
            shadow_mask = res["shadow_mask"]

            # Biến đổi bounding box
            med_region_degraded = degrade.transform_box(med_region_clean, H_total, img_shape=(h, w))
            boxes_degraded = degrade.transform_boxes(raw_boxes, H_total, img_shape=(h, w))
            n_clipped = sum(1 for b in boxes_degraded if b.get("clipped", False))

            # Ghi các file lên đĩa theo cấu trúc phẳng chuẩn
            degraded_file = img_dir / f"{sample_id}.png"
            clean_file = clean_dir / f"{sample_id}.png"
            clean_bin_file = clean_bin_dir / f"{sample_id}.png"
            shadow_mask_file = shadow_dir / f"{sample_id}.png"
            page_mask_file = page_dir / f"{sample_id}.png"
            gt_file = lbl_dir / f"{sample_id}.json"

            cv2.imwrite(str(degraded_file), degraded_img)
            cv2.imwrite(str(shadow_mask_file), shadow_mask)
            cv2.imwrite(str(page_mask_file), page_mask)

            if var_idx == 0:
                cv2.imwrite(str(clean_file), clean_bgr)
                cv2.imwrite(str(clean_bin_file), clean_bin)
                v00_clean_path = clean_file
                v00_clean_bin_path = clean_bin_file
            else:
                shutil.copyfile(v00_clean_path, clean_file)
                shutil.copyfile(v00_clean_bin_path, clean_bin_file)

            # Xây dựng gt.json đầy đủ chi tiết
            gt_dict = {
                "sample_id": sample_id,
                "source_id": src_id,
                "split": split,
                "version": version,
                "variant_index": var_idx,
                "recipe_name": res["recipe_name"],
                "seed_tuple": [seed, src_idx, var_idx],
                "image_size": [int(w), int(h)],
                "applied_degradations": res["applied_degradations"],
                "H": res["H_clean_to_degraded"],
                "H_clean_to_degraded": res["H_clean_to_degraded"],
                "H_degraded_to_clean": res["H_degraded_to_clean"],
                "corners": res["corners"],
                "angle_deg": res["angle_deg"],
                "degradation_params": res["degradation_params"],
                "masks": {
                    "shadow_mask": f"shadow_mask/{sample_id}.png",
                    "page_mask": f"page_mask/{sample_id}.png",
                    "has_shadow": res["has_shadow"]
                },
                "active_masks": ["shadow_mask"] if res["has_shadow"] else [],
                "gt_coverage": {
                    "has_page_gt": True,
                    "has_medicine_region_gt": True,
                    "has_line_boxes_gt": (len(raw_boxes) > 0),
                    "has_verified_item_gt": False
                },
                "medicine_region_clean": med_region_clean,
                "medicine_region_degraded": med_region_degraded,
                "n_boxes_total": len(boxes_degraded),
                "n_boxes_clipped": n_clipped,
                "boxes_clean": raw_boxes,
                "boxes_degraded": boxes_degraded
            }

            with open(gt_file, "w", encoding="utf-8") as gf:
                json.dump(gt_dict, gf, indent=2, ensure_ascii=False)

            # Metadata cho manifest tổng
            all_samples_meta.append({
                "sample_id": sample_id,
                "source_id": src_id,
                "split": split,
                "variant_index": var_idx,
                "recipe_name": res["recipe_name"],
                "rel_degraded_path": f"{split}/images/{sample_id}.png",
                "rel_clean_path": f"{split}/clean/{sample_id}.png",
                "rel_clean_binary_path": f"{split}/clean_binary/{sample_id}.png",
                "rel_gt_path": f"{split}/labels/{sample_id}.json",
                "width": w,
                "height": h,
                "applied_degradations": ",".join(res["applied_degradations"]),
                "angle_deg": res["angle_deg"],
                "perspective_strength": res["perspective_strength"],
                "blur_level": res["blur_level"],
                "blur_kind": res["blur_kind"],
                "shadow_level": res["shadow_level"],
                "shadow_type": res["shadow_type"],
                "has_shadow": res["has_shadow"],
                "n_boxes_total": len(boxes_degraded),
                "n_boxes_clipped": n_clipped
            })

            split_counts[split] += 1

    # Lưu manifest tổng vào thư mục out_dir
    manifest_json_path = out_dir / f"manifest_{version}.json"
    manifest_csv_path = out_dir / f"manifest_{version}.csv"

    manifest_data = {
        "dataset_name": f"Synthetic Benchmark Set 6 ({version.upper()})",
        "version": version,
        "seed": seed,
        "n_sources": len(sources),
        "variants_per_image": variants_per_image,
        "total_samples": len(all_samples_meta),
        "split_counts": dict(split_counts),
        "samples": all_samples_meta
    }

    with open(manifest_json_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2, ensure_ascii=False)

    pd.DataFrame(all_samples_meta).to_csv(manifest_csv_path, index=False, encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"Hoàn thành sinh {len(all_samples_meta)} mẫu vào {out_dir} sau {elapsed:.2f}s!")
    print(f"Phân bố split: {dict(split_counts)}")
    print(f"Manifest đã lưu: {manifest_json_path} & {manifest_csv_path}")

    return {
        "status": "SUCCESS",
        "version": version,
        "total_samples": len(all_samples_meta),
        "split_counts": dict(split_counts),
        "n_sources": len(sources),
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_json_path),
        "elapsed_seconds": round(elapsed, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Sinh bộ 5 (benchmark tổng hợp)")
    parser.add_argument("--config", type=Path, default=None, help="Đường dẫn file config.yaml")
    parser.add_argument("--src", type=Path, default=None, help="Thư mục ảnh sạch (tùy chọn)")
    parser.add_argument("--out", type=Path, default=None, help="Thư mục ra (mặc định: data/5_synthetic)")
    parser.add_argument("--version", choices=("v1", "v2"), default="v1", help="Phiên bản bộ sinh (v1 hoặc v2)")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Đường dẫn file manifest nguồn (mặc định: results/t03/synthetic_100_sources.json)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số ảnh nguồn khi chạy thử")
    parser.add_argument("--v1-dir", type=Path, default=None, help="(v2) Thư mục Bộ 5 v1 nếu khác --out, ví dụ khi chạy pilot")
    args = parser.parse_args()

    generate(
        config_path=args.config,
        src_dir=args.src,
        out_dir=args.out,
        version=args.version,
        sources_manifest=args.manifest,
        limit_sources=args.limit,
        v1_dir=args.v1_dir,
    )


if __name__ == "__main__":
    main()
