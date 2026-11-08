"""Nghiệm thu bộ bàn giao đầu ra output_handoff_v1 (③ LabReportResult, ④ PrescriptionResult).

Kiểm tra:
1. JSON parse + validate qua hợp đồng trong common/contracts.py, round-trip to_dict ↔ from_dict không đổi.
2. bbox khớp nội dung ảnh: mỗi bbox nằm trong ảnh, chứa mực trên binary.png của cleaning_handoff_v1,
   và ôm khít vùng mực bên trong (IoU giữa bbox và khung mực ≥ 0.5) — bắt lỗi box trỏ vào vùng trống / lệch dòng.
3. Xuất overlay để người duyệt so chữ trên ảnh với nội dung JSON: output_handoff_v1/preview_<loại>.png

Chạy (từ thư mục code/): python examples/output_handoff_v1/verify_output_handoff.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR / "src"))

from common.contracts import LabReportResult, PrescriptionResult  # noqa: E402
from common.io import load_json, read_image, write_image  # noqa: E402

HANDOFF_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = CODE_DIR / "examples" / "cleaning_handoff_v1"
MIN_IOU = 0.5


def ink_fit(binary: np.ndarray, bbox: list) -> float:
    """IoU giữa bbox và khung bao mực nằm trong bbox; 0 nếu bbox không chứa mực."""
    x, y, w, h = bbox
    ys, xs = np.nonzero(binary[y:y + h, x:x + w])
    if len(xs) == 0:
        return 0.0
    ink_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    return ink_area / (w * h)   # khung mực nằm trong bbox nên IoU = diện tích khung mực / diện tích bbox


def check(kind: str, contract, json_name: str, entries) -> None:
    raw = load_json(HANDOFF_DIR / json_name)
    result = contract.from_dict(raw)
    assert contract.from_dict(json.loads(json.dumps(result.to_dict()))).to_dict() == raw, "round-trip không khớp"
    print(f"[{kind}] {result.image_id}: parse + validate + round-trip OK")

    sample_dir = FIXTURE_DIR / result.image_id
    rectified = read_image(sample_dir / "rectified.png")
    binary = read_image(sample_dir / "binary.png", gray=True) > 0
    ih, iw = binary.shape
    for label, bbox in entries(result):
        x, y, w, h = bbox
        assert 0 <= x and 0 <= y and x + w <= iw and y + h <= ih, f"{label}: bbox {bbox} ra ngoài ảnh {iw}x{ih}"
        fit = ink_fit(binary, bbox)
        assert fit >= MIN_IOU, f"{label}: bbox {bbox} không khớp chữ trên ảnh (IoU khung mực = {fit:.2f})"
        print(f"  {label:<40} bbox={bbox}  IoU khung mực={fit:.2f}")
        cv2.rectangle(rectified, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(rectified, label, (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    print(f"  overlay: {write_image(HANDOFF_DIR / f'preview_{kind}.png', rectified)}")


def lab_entries(r: LabReportResult):
    return [(f"{row.parameter}={row.result} {row.unit} {row.flag}", row.bbox) for row in r.rows]


def prescription_entries(r: PrescriptionResult):
    out = [(f"{k}: {f.text}", f.bbox) for k, f in (("patient", r.patient), ("diagnosis", r.diagnosis)) if f]
    for it in r.items:
        d = it.dose
        out.append((f"{it.name} {it.strength or ''} lieu {d.morning:g}-{d.noon:g}-{d.evening:g}", it.bbox))
    return out


def main():
    check("lab_report", LabReportResult, "lab_report_sample.json", lab_entries)
    check("prescription", PrescriptionResult, "prescription_sample.json", prescription_entries)
    print("Đạt các kiểm tra tự động. Vẫn cần người xem preview_*.png đối chiếu chữ trên ảnh với nội dung JSON.")


if __name__ == "__main__":
    main()
