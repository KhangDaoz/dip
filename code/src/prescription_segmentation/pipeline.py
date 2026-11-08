"""Ghép các bước của ④, trả về PrescriptionResult (hợp đồng ④ → app)."""
from common.contracts import CleaningResult, PrescriptionResult


def extract_prescription(cleaned: CleaningResult, ocr_engine: str = "tesseract") -> PrescriptionResult:
    """Đọc ảnh nhị phân của ② → vùng thuốc → dòng → khoản → OCR → trường."""
    raise NotImplementedError("extract_prescription")
