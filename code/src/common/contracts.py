"""Hợp đồng dữ liệu giữa các mắt xích của pipeline.

    ① Hiếu  (preprocessing + classification) ── PreprocessResult ──► ② Trường (cleaning)
    ② Trường (cleaning) ── CleaningResult ──► ③ Khang (table_extraction) / ④ Thắng (prescription_segmentation)
    ③ Khang ── LabReportResult ──► app
    ④ Thắng ── PrescriptionResult ──► app

Quy ước chung:
* bbox = [x, y, w, h] (pixel, số nguyên), gốc tọa độ ở góc trên-trái.
* Tọa độ của ②, ③, ④ tính trên **ảnh đã nắn** (rectified) của ①. Muốn chiếu về ảnh gốc
  thì dùng `PreprocessResult.to_source(points)` (nhân với nghịch đảo của H).
* corners_src theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái, trên **ảnh gốc**.
* homography H là **toàn bộ** biến đổi ảnh gốc → ảnh rectified_path, gồm cả nắn phối cảnh
  và xoay chỉnh nghiêng (H = R · H_persp). skew_deg chỉ để báo cáo, không cần áp lại.
* Ảnh rectified_path: màu BGR, đã nắn + chỉnh nghiêng, **chưa CLAHE** (CLAHE chỉ chạy trên bản
  sao dùng cho phân loại). ② giữ nguyên kích thước và hệ tọa độ của ảnh này.
  (Chốt Hiếu ↔ Trường ngày 31/10/2026.)
* Đường dẫn ảnh là đường dẫn tương đối tính từ thư mục gốc repo (ví dụ "out/rx_000123_rect.png").

Muốn đổi hợp đồng: báo nhóm, sửa file này + tests/test_contracts.py trong cùng một commit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

DOC_TYPES = ("prescription", "lab_report", "other")
CORNER_METHODS = ("contour", "hough", "minrect", "full_image")  # full_image: không tìm được góc, dùng cả ảnh
COMPONENT_LABELS = ("printed", "handwritten", "stamp", "other")
LAB_FLAGS = ("LOW", "NORMAL", "HIGH", "UNKNOWN")


def _check_bbox(bbox, name="bbox"):
    if len(bbox) != 4 or bbox[2] < 0 or bbox[3] < 0:
        raise ValueError(f"{name} phải là [x, y, w, h] với w, h >= 0, nhận được {bbox}")


def _check_conf(value, name):
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} phải trong [0, 1], nhận được {value}")


class _Contract:
    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self):
        return self


# ─────────────────────────── ① Hiếu → ② Trường ───────────────────────────

@dataclass
class Quality(_Contract):
    blur_var: float            # phương sai Laplacian (càng lớn càng nét)
    glare_ratio: float         # tỉ lệ diện tích lóa, 0..1
    is_blurry: bool
    has_glare: bool
    glare_removed: bool = False


@dataclass
class PreprocessResult(_Contract):
    image_id: str
    rectified_path: str        # BGR, đã nắn + chỉnh nghiêng, chưa CLAHE
    homography: list           # 3×3, ảnh gốc → rectified_path (gồm cả xoay)
    corners_src: list          # 4×2, TL, TR, BR, BL trên ảnh gốc
    corner_method: str
    skew_deg: float            # chỉ để báo cáo, đã nằm trong homography
    doc_type: str
    doc_confidence: float
    quality: Quality

    def validate(self):
        if np.asarray(self.homography).shape != (3, 3):
            raise ValueError("homography phải là ma trận 3×3")
        if abs(np.linalg.det(np.asarray(self.homography, dtype=float))) < 1e-12:
            raise ValueError("homography không khả nghịch")
        if np.asarray(self.corners_src).shape != (4, 2):
            raise ValueError("corners_src phải có 4 điểm [x, y]")
        if self.corner_method not in CORNER_METHODS:
            raise ValueError(f"corner_method phải thuộc {CORNER_METHODS}")
        if self.doc_type not in DOC_TYPES:
            raise ValueError(f"doc_type phải thuộc {DOC_TYPES}")
        _check_conf(self.doc_confidence, "doc_confidence")
        return self

    def to_source(self, points) -> np.ndarray:
        """Chiếu điểm (N×2) từ ảnh đã nắn về ảnh gốc — dùng cho màn hình xác nhận."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        h_inv = np.linalg.inv(np.asarray(self.homography, dtype=float))
        homog = np.hstack([pts, np.ones((len(pts), 1))]) @ h_inv.T
        return homog[:, :2] / homog[:, 2:3]

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessResult":
        return cls(**{**d, "quality": Quality(**d["quality"])}).validate()


# ─────────────────────── ② Trường → ③ Khang / ④ Thắng ───────────────────────

@dataclass
class Component(_Contract):
    bbox: list
    label: str
    confidence: float

    def validate(self):
        _check_bbox(self.bbox)
        if self.label not in COMPONENT_LABELS:
            raise ValueError(f"label phải thuộc {COMPONENT_LABELS}")
        _check_conf(self.confidence, "confidence")
        return self


@dataclass
class CleaningResult(_Contract):
    image_id: str
    clean_gray_path: str
    binary_path: str                                   # mực = 255, nền = 0
    masks: dict = field(default_factory=dict)          # {"stamp", "handwriting", "red_text"} -> path
    components: list = field(default_factory=list)     # list[Component]
    params: dict = field(default_factory=dict)         # tham số đã dùng

    def validate(self):
        for c in self.components:
            c.validate()
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "CleaningResult":
        comps = [Component(**c) for c in d.get("components", [])]
        return cls(**{**d, "components": comps}).validate()


# ───────────────────────────── ③ Khang → app ─────────────────────────────

@dataclass
class LabRow(_Contract):
    bbox: list
    parameter: str
    result: Optional[float]
    unit: Optional[str]
    range: Optional[list]          # [thấp, cao]; None nếu không đọc được
    flag: str
    loinc: Optional[str] = None
    confidence: float = 0.0

    def validate(self):
        _check_bbox(self.bbox)
        if self.flag not in LAB_FLAGS:
            raise ValueError(f"flag phải thuộc {LAB_FLAGS}")
        if self.range is not None and len(self.range) != 2:
            raise ValueError("range phải là [thấp, cao]")
        _check_conf(self.confidence, "confidence")
        return self


@dataclass
class LabReportResult(_Contract):
    image_id: str
    report_type: Optional[str]
    rows: list = field(default_factory=list)   # list[LabRow]

    def validate(self):
        for r in self.rows:
            r.validate()
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "LabReportResult":
        return cls(**{**d, "rows": [LabRow(**r) for r in d.get("rows", [])]}).validate()


# ───────────────────────────── ④ Thắng → app ─────────────────────────────

@dataclass
class TextField(_Contract):
    bbox: list
    text: str


@dataclass
class Dose(_Contract):
    morning: float = 0
    noon: float = 0
    evening: float = 0


@dataclass
class MedicineItem(_Contract):
    bbox: list
    name: str
    strength: Optional[str] = None
    dose: Dose = field(default_factory=Dose)
    days: Optional[int] = None
    note: Optional[str] = None
    confidence: float = 0.0

    def validate(self):
        _check_bbox(self.bbox)
        _check_conf(self.confidence, "confidence")
        return self


@dataclass
class PrescriptionResult(_Contract):
    image_id: str
    patient: Optional[TextField] = None
    diagnosis: Optional[TextField] = None
    items: list = field(default_factory=list)   # list[MedicineItem]

    def validate(self):
        for it in self.items:
            it.validate()
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "PrescriptionResult":
        def text(x):
            return TextField(**x) if x else None
        items = [MedicineItem(**{**it, "dose": Dose(**it.get("dose", {}))}) for it in d.get("items", [])]
        return cls(image_id=d["image_id"], patient=text(d.get("patient")),
                   diagnosis=text(d.get("diagnosis")), items=items).validate()
