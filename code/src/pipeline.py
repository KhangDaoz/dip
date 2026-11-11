"""Ghép A → B → C → D (plan §4). Chọn biến thể bằng tham số — dùng chung cho CLI, demo, mọi thí nghiệm.

    from pipeline import Pipeline
    pipe = Pipeline(enhance="none", detector="yolo", ocr="C3")
    result = pipe(img, image_id)

Biến thể E-D2:   (i) classical+C0+rules · (ii) yolo+C0 · (iii) yolo+C3 (chính) · (iv) yolo+C2 · (v) VLM (assemble.vlm)
Ablation E-X1:   detector="gt" (B chuẩn) và/hoặc ocr="gt" (C chuẩn) — cần truyền `gt` = list VAIPEAnnotation.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from common.boxes import greedy_match
from common.contracts import BOX_LABELS, PrescriptionResult, TextBox

PRESET_PIPELINES = {
    "classical": {"detector": "classical", "ocr": "C0", "classify": "rules"},
    "hybrid": {"detector": "yolo", "ocr": "C0", "classify": "model"},
    "dl": {"detector": "yolo", "ocr": "C3", "classify": "model"},
    "dl_c2": {"detector": "yolo", "ocr": "C2", "classify": "model"},
}


class Pipeline:
    def __init__(self, enhance="none", detector: str = "yolo", ocr: str = "C3", classify: str | None = None,
                 lexicon="default", detector_kw: dict | None = None, ocr_kw: dict | None = None,
                 enhance_params: dict | None = None):
        """
        enhance:  tên preset (none/geom/light/full) hoặc tuple bước — xem enhance.pipeline
        detector: yolo | classical | gt
        ocr:      C0 | C1 | C2 | C3 | gt
        classify: model (dùng nhãn khối B) | rules (D1 trên chữ). Mặc định: rules nếu detector=classical.
        lexicon:  "default" (từ điển train) | None (tắt D3) | đối tượng Lexicon
        """
        self.enhance_steps = enhance
        self.enhance_params = enhance_params
        self.detector = detector
        self.detector_kw = detector_kw or {}
        self.ocr_name = ocr
        self.classify = classify or ("rules" if detector == "classical" else "model")
        self._engine = None
        self._ocr_kw = ocr_kw or {}
        if lexicon == "default":
            from assemble.lexicon import default_lexicon
            lexicon = default_lexicon()
        self.lexicon = lexicon

    @classmethod
    def preset(cls, name: str, **kw) -> "Pipeline":
        return cls(**{**PRESET_PIPELINES[name], **kw})

    @property
    def engine(self):
        if self._engine is None and self.ocr_name != "gt":
            from recognize import get_engine
            self._engine = get_engine(self.ocr_name, **self._ocr_kw)
        return self._engine

    def config(self) -> dict:
        return {"enhance": self.enhance_steps, "detector": self.detector, "detector_kw": self.detector_kw,
                "ocr": self.ocr_name, "classify": self.classify, "lexicon": bool(self.lexicon)}

    def __call__(self, img: np.ndarray, image_id: str = "", gt: list | None = None) -> PrescriptionResult:
        timing = {}
        # A — tiền xử lý
        t = time.perf_counter()
        if self.enhance_steps and self.enhance_steps != "none":
            if gt is not None and (self.detector == "gt" or self.ocr_name == "gt"):
                raise ValueError("enhance đổi hình học → không dùng chung với B/C chuẩn")
            from enhance.pipeline import enhance
            img, _ = enhance(img, self.enhance_steps, self.enhance_params)
        timing["A"] = time.perf_counter() - t

        # B — phát hiện
        t = time.perf_counter()
        if self.detector == "gt":
            boxes = gt_boxes(gt)
        else:
            from detect import detect
            boxes = detect(img, self.detector, **self.detector_kw)
        timing["B"] = time.perf_counter() - t

        # C — nhận dạng
        t = time.perf_counter()
        if self.ocr_name == "gt":
            if self.detector == "gt":   # B chuẩn + C chuẩn: chép chữ theo đúng thứ tự
                for b, a in zip(boxes, gt):
                    b.text, b.text_conf = a.text, 1.0
            else:
                transfer_gt_text(boxes, gt)
        else:
            from recognize import recognize
            recognize(img, boxes, self.engine)
        timing["C"] = time.perf_counter() - t

        # D — phân lớp (nếu cần) + ghép khoản + từ điển
        t = time.perf_counter()
        from assemble.classify_rules import classify_boxes
        from assemble.group import assemble
        if self.classify == "rules":
            classify_boxes(boxes, overwrite=True)
        out = assemble(boxes, self.lexicon)
        timing["D"] = time.perf_counter() - t

        return PrescriptionResult(image_id=image_id, items=out["items"], diagnose=out["diagnose"], date=out["date"],
                                  boxes=boxes, timing=timing)


def gt_boxes(gt: list, with_text: bool = False) -> list:
    """VAIPEAnnotation → TextBox (B chuẩn). with_text=True điền luôn chữ chuẩn (B + C chuẩn)."""
    if gt is None:
        raise ValueError("detector='gt' cần truyền gt (list VAIPEAnnotation)")
    return [TextBox(box=list(a.box), label=a.label, score=1.0, text=a.text if with_text else "",
                    text_conf=1.0 if with_text else 0.0) for a in gt]


def transfer_gt_text(boxes: list, gt: list, iou_threshold: float = 0.5) -> list:
    """C chuẩn trên box dự đoán: box ghép được với GT (IoU ≥ ngưỡng) nhận chữ GT, còn lại rỗng."""
    if gt is None:
        raise ValueError("ocr='gt' cần truyền gt")
    for b in boxes:
        b.text, b.text_conf = "", 0.0
    for gi, pi, _ in greedy_match([a.box for a in gt], [b.box for b in boxes], iou_threshold):
        boxes[pi].text, boxes[pi].text_conf = gt[gi].text, 1.0
    return boxes


# ---------- overlay cho demo ----------
COLORS = {"drugname": (0, 0, 220), "quantity": (0, 150, 0), "usage": (200, 80, 0),
          "diagnose": (160, 0, 160), "date": (0, 140, 200), "other": (128, 128, 128), None: (0, 200, 200)}


def draw_overlay(img: np.ndarray, result: PrescriptionResult, show_text: bool = False) -> np.ndarray:
    """Vẽ box theo màu lớp + khung từng khoản (đánh số)."""
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = max(1, img.shape[1] // 600)
    for b in result.boxes:
        lab = b.label if hasattr(b, "label") else b["label"]
        box = b.box if hasattr(b, "box") else b["box"]
        cv2.rectangle(vis, tuple(box[:2]), tuple(box[2:]), COLORS.get(lab, COLORS[None]), t)
    for i, it in enumerate(result.items, 1):
        box = it.box if hasattr(it, "box") else it["box"]
        if box:
            x0, y0, x1, y1 = box
            cv2.rectangle(vis, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 255), t + 1)
            cv2.putText(vis, str(i), (max(0, x0 - 24), y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), t + 1)
    return vis


LEGEND = {k: v for k, v in COLORS.items() if k in BOX_LABELS}
