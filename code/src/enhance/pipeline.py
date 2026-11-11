"""Khối A — tiền xử lý cổ điển, mỗi bước bật/tắt được (plan §5.1).

    from enhance.pipeline import enhance, PRESETS
    out, info = enhance(img, steps=PRESETS["full"])

Thứ tự cố định: rectify → deskew → shadow → denoise → clahe.
Lưu ý: rectify/deskew đổi hình học → box nhãn gốc không còn khớp; chỉ dùng khi đo độ đo đầu-cuối.
"""
from __future__ import annotations

import time

import numpy as np

from .clahe import clahe
from .denoise import denoise
from .deskew import deskew
from .rectify import rectify
from .shadow import remove_shadows

STEPS = ("rectify", "deskew", "shadow", "denoise", "clahe")

# Biến thể dùng trong E-A1 / E-X2
PRESETS = {
    "none": (),
    "geom": ("deskew",),
    "full": ("rectify", "deskew", "shadow", "denoise", "clahe"),
    "light": ("deskew", "shadow"),
}

DEFAULT_PARAMS = {
    "rectify": {"canny_low": 50, "canny_high": 150},
    "deskew": {"max_angle": 10.0, "step": 0.25},
    "shadow": {"kernel": 81},
    "denoise": {"method": "median", "ksize": 3},
    "clahe": {"clip_limit": 2.0, "tile_grid": 8},
}


def enhance(img: np.ndarray, steps=PRESETS["full"], params: dict | None = None) -> tuple:
    """Chạy các bước trong `steps` (tên thuộc STEPS hoặc tên preset). Trả về (ảnh, info)."""
    if isinstance(steps, str):
        steps = PRESETS[steps]
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"bước không hợp lệ: {unknown}; hợp lệ: {STEPS}")
    p = {k: {**v, **((params or {}).get(k, {}))} for k, v in DEFAULT_PARAMS.items()}
    info = {"steps": list(steps), "timing": {}}
    out = img
    for step in STEPS:
        if step not in steps:
            continue
        t = time.perf_counter()
        if step == "rectify":
            out, info["rectify_method"] = rectify(out, **p["rectify"])
        elif step == "deskew":
            out, info["skew_deg"] = deskew(out, **p["deskew"])
        elif step == "shadow":
            out = remove_shadows(out, **p["shadow"])
        elif step == "denoise":
            out = denoise(out, **p["denoise"])
        elif step == "clahe":
            out = clahe(out, **p["clahe"])
        info["timing"][step] = time.perf_counter() - t
    return out, info
