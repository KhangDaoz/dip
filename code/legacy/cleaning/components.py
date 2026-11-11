"""R9 thành phần liên thông + nhãn bằng luật (KeHoach.md mục 6.1). Không dùng ML."""
import cv2
import numpy as np

from common.contracts import Component


def extract_components(binary: np.ndarray, min_area: int = 10) -> list:
    """Gán nhãn liên thông (C5 · S84–85). Trả về list dict {"bbox": [x, y, w, h], "area", "mask_index"}."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a >= min_area:
            out.append({"bbox": [x, y, w, h], "area": a, "mask_index": i})
    return out


def label_components(components: list, ink: dict, stamp_mask: np.ndarray, page_shape=None) -> list:
    """Luật màu (R3) + hình dạng (R4) → list[Component], label ∈ COMPONENT_LABELS.

    stamp: nằm phần lớn trong mask dấu · handwritten: phần lớn là mực xanh · printed: mực đen hoặc chữ đỏ in ·
    other: vệt quá to / quá dẹt (đường kẻ, khung). confidence = tỉ lệ pixel của lớp thắng trong bbox.
    """
    H, W = page_shape or stamp_mask.shape[:2]
    out = []
    for c in components:
        x, y, w, h = c["bbox"]
        sl = (slice(y, y + h), slice(x, x + w))
        n = max(w * h, 1)
        frac = {"stamp": (stamp_mask[sl] > 0).sum() / n,
                "blue": (ink["blue"][sl] > 0).sum() / n,
                "red": (ink["red"][sl] > 0).sum() / n,
                "black": (ink["black"][sl] > 0).sum() / n}
        if frac["stamp"] > 0.5:
            label, conf = "stamp", frac["stamp"]
        elif w > 0.5 * W or h > 0.3 * H or max(w, h) > 20 * max(min(w, h), 1):
            label, conf = "other", 0.6
        elif frac["blue"] > max(frac["black"], frac["red"]):
            label, conf = "handwritten", frac["blue"] / max(frac["blue"] + frac["black"], 1e-6)
        else:
            label, conf = "printed", max(frac["black"], frac["red"]) / max(frac["black"] + frac["red"] + frac["blue"], 1e-6)
        out.append(Component(bbox=[int(v) for v in c["bbox"]], label=label,
                             confidence=round(float(min(max(conf, 0.0), 1.0)), 3)))
    return out
