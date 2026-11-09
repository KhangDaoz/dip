"""④ Ghép S-B0–S-B5 theo hợp đồng PrescriptionResult (common/contracts.py)."""
from common.contracts import CleaningResult, PrescriptionResult


def extract_prescription(cleaned: CleaningResult) -> PrescriptionResult:
    """Ảnh nhị phân đơn thuốc → vùng thuốc → dòng → khoản thuốc kèm bbox (+ trường nếu OCR đọc được)."""
    raise NotImplementedError("extract_prescription")
