"""② Ghép R1–R9 theo hợp đồng CleaningResult (common/contracts.py)."""
import os
from pathlib import Path

import cv2
import numpy as np

from common.contracts import CleaningResult
from common.io import write_image
from common.paths import REPO_DIR
from . import background, binarize, components, ink, inpaint, postprocess, stamp

PARAMS = {
    "background_kernel": 81,      # R1 — chọn bằng T1 (PSNR trên valid bộ 5)
    "background_median": 0,
    "binarize": "sauvola",        # R7 — phương án chính đã chốt (KeHoach.md mục 6)
    "sauvola_window_factor": 2.5, # w ≈ 2–3 × chiều cao chữ
    "sauvola_k": 0.2,
    "stretch": False,             # R2b — kéo giãn tương phản (C3 · S21): thử nhưng TẮT — không cải thiện S1 (F1 valid như nhau);
                                  #       chữ phiếu XN giữ được nhờ ngưỡng diện tích R8 theo h, không cần kéo giãn
    "opening": 1,                 # R8 — tắt mặc định: Opening 2×2 xóa nét chữ mảnh 1–2 px (C7 · S14); chọn lại bằng T2
    "min_area_factor": 0.4,       # R8 — bỏ vết có diện tích < (0,4·h)², tối thiểu 3 px (tỉ lệ theo chiều cao chữ)
    "inpaint": "telea",           # R6
    "remove_blue": True,          # đơn thuốc: xóa nét bút xanh (chữ ký / viết tay)
}


def estimate_char_height(gray: np.ndarray) -> float:
    """Trung vị chiều cao thành phần liên thông trên ảnh Otsu (C5 · S84; C3 · S73) — dùng đặt cửa sổ Sauvola."""
    b = binarize.otsu(gray)
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    hs = hs[(hs >= 3) & (hs <= 0.1 * gray.shape[0])]
    return float(np.median(hs)) if len(hs) else 15.0


def min_area_for(h: float, factor: float) -> int:
    return max(3, int(round((factor * h) ** 2)))


def binarize_clean(gray_clean: np.ndarray, p: dict) -> tuple:
    """R7 + R8 trên ảnh xám đã khử bóng. Trả về (ảnh nhị phân, tham số đã dùng)."""
    p = {**PARAMS, **p}
    used = {"binarize": p["binarize"]}
    h = estimate_char_height(gray_clean)
    if p["binarize"] in ("sauvola", "niblack", "wolf"):
        window = int(max(15, round(p["sauvola_window_factor"] * h))) | 1
        kw = {"window": window}
        if p["binarize"] == "sauvola":
            kw["k"] = p["sauvola_k"]
        b = binarize.binarize(gray_clean, p["binarize"], **kw)
        used.update(char_height=round(h, 1), **{f"{p['binarize']}_{k}": v for k, v in kw.items()})
    else:
        b = binarize.binarize(gray_clean, p["binarize"])
    min_area = min_area_for(h, p["min_area_factor"])
    b = postprocess.remove_small(postprocess.opening(b, p["opening"]), min_area)
    used.update(opening=p["opening"], min_area=min_area)
    return b, used


def shadow_free(img_bgr: np.ndarray, p: dict) -> np.ndarray:
    """R1 + R2 (+ R2b): ảnh xám đã khử bóng, kéo giãn độ tương phản."""
    p = {**PARAMS, **p}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    bg = background.background_closing(gray, p["background_kernel"], p["background_median"])
    flat = background.remove_shadow(gray, bg)
    return background.stretch_contrast(flat) if p["stretch"] else flat


def _rel(path: Path) -> str:
    try:
        return Path(os.path.relpath(path, REPO_DIR)).as_posix()
    except ValueError:
        return Path(path).as_posix()


def clean(img_bgr: np.ndarray, doc_type: str, image_id: str, out_dir: Path, params: dict = None) -> CleaningResult:
    """Ảnh đã nắn → ảnh xám sạch, ảnh nhị phân, mask (stamp, handwriting, red_text), components có nhãn.

    Đơn thuốc: chạy xóa dấu / chữ ký (R6). Phiếu xét nghiệm: KHÔNG xóa, giữ chữ đỏ cảnh báo.
    Giữ nguyên kích thước và hệ tọa độ ảnh đầu vào.
    """
    p = {**PARAMS, **(params or {})}
    out_dir = Path(out_dir)
    img = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    masks_ink = ink.ink_masks(img)                                   # R3
    stamp_mask, red_text = stamp.split_stamp_vs_text(masks_ink["red"])  # R4 + R5
    hw_mask = masks_ink["blue"]

    work = img
    if doc_type == "prescription":                                   # R6 chỉ cho đơn thuốc
        marks = stamp_mask.copy()
        if p["remove_blue"]:
            marks = np.maximum(marks, hw_mask)
        marks = cv2.dilate(marks, np.ones((3, 3), np.uint8))
        work = inpaint.remove_marks(img, marks, p["inpaint"])

    gray_clean = shadow_free(work, p)                                # R1 + R2
    binary, used = binarize_clean(gray_clean, p)                     # R7 + R8
    comps = components.label_components(components.extract_components(binary, used["min_area"]),
                                        masks_ink, stamp_mask, img.shape[:2])   # R9

    paths = {
        "clean": write_image(out_dir / f"{image_id}_clean.png", gray_clean),
        "binary": write_image(out_dir / f"{image_id}_bin.png", binary),
        "stamp": write_image(out_dir / f"{image_id}_stamp.png", stamp_mask),
        "handwriting": write_image(out_dir / f"{image_id}_hw.png", hw_mask),
        "red_text": write_image(out_dir / f"{image_id}_red.png", red_text),
    }
    return CleaningResult(
        image_id=image_id, clean_gray_path=_rel(paths["clean"]), binary_path=_rel(paths["binary"]),
        masks={k: _rel(paths[k]) for k in ("stamp", "handwriting", "red_text")},
        components=comps,
        params={"background_kernel": p["background_kernel"], "doc_type": doc_type,
                "inpaint": p["inpaint"] if doc_type == "prescription" else None, **used},
    ).validate()
