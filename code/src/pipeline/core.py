"""Ghép pipeline A → B → C → D (plan §4). Dùng chung cho CLI, demo và các thí nghiệm E-D2, E-X1, E-A1.

    from pipeline import Pipeline
    pipe = Pipeline.preset("dl_c3")                  # YOLO + VietOCR fine-tune + khối D (chính)
    result = pipe(img_bgr, "VAIPE_P_TRAIN_1")        # PrescriptionResult (items, diagnose, date, boxes, timing)

Preset (các dòng của bảng chính E-D2):
    classical  B cổ điển (Sauvola + giãn) + C0 Tesseract + D phân lớp bằng chữ
    hybrid     YOLO + C0 Tesseract
    dl_c2      YOLO + C2 VietOCR pretrained
    dl_c3      YOLO + C3 VietOCR fine-tune      ← pipeline chính
`enhance` (khối A): "none" (mặc định) | "geom" | "light" | "full" — xem enhance.pipeline.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from common.contracts import BOX_LABELS, PrescriptionResult

PRESETS = {
    "classical": {"detector": "classical", "ocr": "C0"},
    "hybrid": {"detector": "yolo", "ocr": "C0"},
    "dl_c2": {"detector": "yolo", "ocr": "C2"},
    "dl_c3": {"detector": "yolo", "ocr": "C3"},
}
_ENGINES: dict = {}


def get_engine(name: str):
    """C0/C1 Tesseract, C2/C3 VietOCR — nạp một lần rồi dùng lại."""
    if name not in _ENGINES:
        if name in ("C0", "C1"):
            from recognize.tesseract_engine import TesseractEngine
            _ENGINES[name] = TesseractEngine("gray" if name == "C0" else "binary")
        elif name in ("C2", "C3"):
            from recognize.vietocr_engine import VietOCREngine
            _ENGINES[name] = VietOCREngine("pretrained" if name == "C2" else "finetuned")
        else:
            raise ValueError(f"ocr phải thuộc C0/C1/C2/C3, nhận {name}")
    return _ENGINES[name]


class Pipeline:
    def __init__(self, detector: str = "yolo", ocr: str = "C3", enhance="none", lexicon="train"):
        """detector: yolo | classical · ocr: C0 | C1 | C2 | C3 · lexicon: "train" (từ điển train) | None | Lexicon."""
        if detector not in ("yolo", "classical"):
            raise ValueError("detector phải là 'yolo' hoặc 'classical'")
        self.detector, self.ocr, self.enhance = detector, ocr, enhance
        if lexicon == "train":
            from assemble.lexicon import Lexicon
            lexicon = Lexicon.from_split("train")
        self.lexicon = lexicon

    @classmethod
    def preset(cls, name: str, **kw) -> "Pipeline":
        if name not in PRESETS:
            raise ValueError(f"preset phải thuộc {list(PRESETS)}")
        return cls(**{**PRESETS[name], **kw})

    def config(self) -> dict:
        return {"detector": self.detector, "ocr": self.ocr, "enhance": self.enhance,
                "lexicon": len(self.lexicon) if self.lexicon is not None else None}

    def read(self, img: np.ndarray) -> tuple:
        """A → B → C: trả về (ảnh sau khối A, list[TextBox] đã có chữ, timing)."""
        timing = {}
        t = time.perf_counter()
        if self.enhance and self.enhance != "none":
            from enhance.pipeline import enhance
            img, _ = enhance(img, self.enhance)
        timing["A"] = time.perf_counter() - t

        t = time.perf_counter()
        if self.detector == "yolo":
            from detect.yolo import detect_yolo
            boxes = detect_yolo(img)
        else:
            from detect.classical import detect_classical
            boxes = detect_classical(img)
        timing["B"] = time.perf_counter() - t

        t = time.perf_counter()
        get_engine(self.ocr).recognize(img, boxes)
        timing["C"] = time.perf_counter() - t
        return img, boxes, timing

    def __call__(self, img: np.ndarray, image_id: str = "") -> PrescriptionResult:
        img, boxes, timing = self.read(img)
        t = time.perf_counter()
        from assemble.group import assemble_result
        res = assemble_result(image_id, boxes, lexicon=self.lexicon)
        timing["D"] = time.perf_counter() - t
        res.timing = timing
        return res


def handoff_record(image_id: str, boxes: list, gt: list | None = None, iou_thr: float = 0.5) -> dict:
    """Bản ghi bàn giao B+C cho khối D (định dạng assemble/load.py đọc được).

    gt (list VAIPEAnnotation, tùy chọn): box dự đoán ghép tham lam với box đáp án (IoU ≥ iou_thr) nhận thêm trường
    "gt" = chữ chuẩn — chỉ dùng cho dòng kiểm "text chuẩn" của E-D2, KHÔNG dùng khi dự đoán.
    """
    gt_text = {}
    if gt:
        from common.boxes import greedy_match
        for gi, pi, _ in greedy_match([a.box for a in gt], [b.box for b in boxes], iou_thr):
            gt_text[pi] = gt[gi].text
    recs = [{"idx": i, "class": b.label, "conf": round(b.score, 4), "xyxy": list(b.box), "ocr": b.text,
             "ocr_conf": round(b.text_conf, 4), **({"gt": gt_text.get(i, "")} if gt else {})}
            for i, b in enumerate(boxes)]
    return {"image_id": image_id, "boxes": recs}


COLORS = {"drugname": (0, 0, 220), "quantity": (0, 150, 0), "usage": (200, 80, 0), "diagnose": (160, 0, 160),
          "date": (0, 140, 200), "other": (150, 150, 150), None: (0, 190, 190)}


def draw_overlay(img: np.ndarray, result: PrescriptionResult) -> np.ndarray:
    """Box vùng chữ tô theo lớp + khung đỏ đánh số cho từng khoản thuốc."""
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = max(1, img.shape[1] // 700)
    for b in result.boxes:
        cv2.rectangle(vis, tuple(b.box[:2]), tuple(b.box[2:]), COLORS.get(b.label, COLORS[None]), t)
    for i, it in enumerate(result.items, 1):
        if it.box:
            x0, y0, x1, y1 = it.box
            cv2.rectangle(vis, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 255), t + 1)
            cv2.putText(vis, str(i), (max(0, x0 - 26), y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), t + 1)
    return vis


LEGEND = {k: v for k, v in COLORS.items() if k in BOX_LABELS}
