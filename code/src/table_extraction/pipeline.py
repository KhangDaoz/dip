"""③ Ghép K1–K7 theo hợp đồng LabReportResult (common/contracts.py)."""
import cv2
import numpy as np

from common.contracts import CleaningResult, LabReportResult, LabRow
from common.io import read_image
from common.paths import REPO_DIR
from . import cells as C
from . import columns, flags, geometry, postprocess, rows

# Tham số cụm từ / dòng / cột = bộ K2 chọn giống nhau ở cả 5 fold (results/k2/folds.csv, 32 ảnh GT bộ 2).
# opening = 0: Opening 2×2 xóa mất nét chữ 1 px ở ảnh độ phân giải thấp (5-fold đều chọn không Opening).
PARAMS = {"opening": 0, "word_a": 2.0, "word_v": 0.5, "remove_lines": True, "row_se": 2.0, "smooth_window": 1,
          "valley_ratio": 0.1, "col_gap": 0.5, "col_cross": 0.6, "ocr": True, "ocr_lang": "eng", "ocr_scale": 3}


def structure(binary: np.ndarray, params: dict = None) -> dict:
    """Chỉ phần DIP: cụm từ → dòng → cột → ô. Dùng chung cho pipeline và thí nghiệm K1–K4."""
    p = {**PARAMS, **(params or {})}
    b = postprocess.opening_before_projection(binary, p["opening"])                         # K6
    h = geometry.char_height(b)                                                             # K1
    # xóa đường kẻ (Opening SE đường thẳng, C7 · S6, S28): kẻ dọc lấp thung lũng P(y), kẻ ngang nối các cột
    b = postprocess.remove_rule_lines(b) if p.get("remove_lines", True) else b              # K6
    words = geometry.words_from_binary(b, h, p["word_a"], v=p["word_v"])                    # K2
    rws = rows.rows_projection(b, words, h, p["row_se"], p["smooth_window"], p["valley_ratio"])  # K3
    rws = postprocess.drop_thin_rows(rws, words, h)                                         # K6
    cols = columns.columns_projection(b, rws, words, h, p["col_gap"], p["col_cross"])        # K4
    cols = postprocess.merge_narrow_columns(cols, h)                                        # K6
    cl = C.assign_words_to_cells(words, rws, cols)                                          # K5
    return {"h": h, "words": words, "rows": rws, "columns": cols, "cells": cl}


def _ocr_cells(gray: np.ndarray, cells: list, lang: str, scale: float = 3) -> dict:
    """OCR từng ô; vùng cắt phóng ×scale (nội suy bậc ba) vì chữ nhỏ (bộ 2 cao 4–11 px, trung vị 5 px) — như thí nghiệm K5."""
    from ocr import engines
    texts = {}
    for c in cells:
        x, y, w, h = c["bbox"]
        pad = 2
        crop = gray[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
        if crop.size and scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        texts[(c["row"], c["col"])] = engines.ocr_line(crop, lang)[0] if crop.size else ""
    return texts


def extract_table(cleaned: CleaningResult, params: dict = None) -> LabReportResult:
    """Ảnh nhị phân phiếu xét nghiệm → cụm từ → dòng → cột → ô → dòng chỉ số kèm bbox (+ OCR, cờ)."""
    p = {**PARAMS, **(params or {})}
    binary = read_image(REPO_DIR / cleaned.binary_path, gray=True)
    s = structure(binary, p)
    words, cl = s["words"], s["cells"]

    texts = {}
    if p["ocr"]:
        try:
            # phóng sao cho chữ cao ≈ 20 px (Tesseract kém khi chữ nhỏ), tối đa ×ocr_scale; ảnh chụp lớn không phóng
            scale = float(min(p["ocr_scale"], max(1.0, 20.0 / max(s["h"], 1.0))))
            texts = _ocr_cells(read_image(REPO_DIR / cleaned.clean_gray_path, gray=True), cl, p["ocr_lang"], scale)
        except Exception:   # chưa cài OCR: vẫn trả cấu trúc bảng
            texts = {}
    roles = flags.column_roles(cl, texts) if texts else {}

    out_rows = []
    for r, row in enumerate(s["rows"]):
        bbox = rows.row_bbox(words, row)
        by_role = {roles.get(c["col"]): texts.get((r, c["col"]), "") for c in cl if c["row"] == r}
        result = flags.parse_number(by_role.get("result", ""))
        rng = flags.parse_range(by_role.get("range", ""))
        if rng is not None and not np.isfinite(rng[1]):
            rng = [rng[0], 1e12]
        out_rows.append(LabRow(bbox=[int(v) for v in bbox], parameter=by_role.get("parameter", "") or "",
                               result=result, unit=by_role.get("unit") or None, range=rng,
                               flag=flags.flag(result, rng), confidence=0.0))
    return LabReportResult(image_id=cleaned.image_id, report_type=None, rows=out_rows).validate()
