"""④ Ghép S-B0–S-B5 theo hợp đồng PrescriptionResult (common/contracts.py)."""
import cv2
import numpy as np

from common.contracts import CleaningResult, Dose, MedicineItem, PrescriptionResult
from common.io import read_image
from common.paths import REPO_DIR
from . import fields, items, lines, region

# Giá trị chọn trên valid VAIPE (KeHoach.md mục 8.4): s, bold_ratio, opening từ S1; region_k, max_follow từ S4
PARAMS = {"region_k": 2.0, "max_follow": 1, "use_region": True, "region_pad": 10, "opening": 1,
          "smooth_window": 3, "valley_ratio": 0.05, "method": "dilation_rule", "s": 3.0, "indent_tol": 1.0,
          "bold_ratio": 1.25, "ocr": True, "ocr_lang": "vie"}


def segment(binary: np.ndarray, params: dict = None) -> dict:
    """Chỉ phần DIP: S-B0 → S-B4. Trả về {"h", "region", "lines", "items"}; tọa độ trên ảnh đầy đủ."""
    p = {**PARAMS, **(params or {})}
    b = (binary > 0).astype(np.uint8) * 255
    if p["opening"] > 1:                                                        # S-B4: Opening trước projection
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((p["opening"], p["opening"]), np.uint8))
    h = lines.char_height(b)                                                    # S-B0
    H, W = b.shape
    reg = region.find_drug_region(b, h, p["region_k"], p["bold_ratio"], p["max_follow"]) \
        if p["use_region"] else [0, 0, W, H]                                    # S-B1
    pad = p["region_pad"] if p["use_region"] else 0
    x, y = max(reg[0] - pad, 0), max(reg[1] - pad, 0)
    w, hh = min(reg[0] + reg[2] + pad, W) - x, min(reg[1] + reg[3] + pad, H) - y
    sub = b[y:y + hh, x:x + w]
    ls = lines.lines_projection(sub, h, p["smooth_window"], p["valley_ratio"])  # S-B2
    its = items.group(p["method"], sub, ls, h, **{k: v for k, v in p.items() if k != "method"})  # S-B3
    its = items.drop_fake_items(sub, its, h)                                    # S-B4
    shift = lambda bb: [bb[0] + x, bb[1] + y, bb[2], bb[3]]
    return {"h": h, "region": reg, "lines": [shift(l) for l in ls], "items": [shift(i) for i in its]}


def _ocr(gray: np.ndarray, box: list, lang: str) -> tuple:
    from ocr import engines
    x, y, w, h = box
    pad = 4
    crop = gray[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
    return engines.ocr_block(crop, lang)


def extract_prescription(cleaned: CleaningResult, params: dict = None) -> PrescriptionResult:
    """Ảnh nhị phân đơn thuốc → vùng thuốc → dòng → khoản thuốc kèm bbox (+ trường nếu OCR đọc được)."""
    p = {**PARAMS, **(params or {})}
    binary = read_image(REPO_DIR / cleaned.binary_path, gray=True)
    seg = segment(binary, p)
    gray = None
    if p["ocr"]:
        try:
            gray = read_image(REPO_DIR / cleaned.clean_gray_path, gray=True)
            from ocr import engines
            if not engines.available("tesseract"):
                gray = None
        except Exception:
            gray = None
    out = []
    for box in seg["items"]:
        name, f, conf = "", {}, 0.0
        if gray is not None:
            text, conf = _ocr(gray, box, p["ocr_lang"])
            f = fields.parse_item_fields(text)
            name = f.get("name", "")
        out.append(MedicineItem(bbox=[int(v) for v in box], name=name, strength=f.get("strength"),
                                dose=Dose(**f.get("dose", {})) if f else Dose(), days=f.get("days"),
                                note=f.get("note"), confidence=round(float(conf), 3)))
    return PrescriptionResult(image_id=cleaned.image_id, items=out).validate()
