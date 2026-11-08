"""Ghép các bước của ③, trả về LabReportResult (hợp đồng ③ → app)."""
from common.contracts import CleaningResult, LabReportResult


def extract_table(cleaned: CleaningResult, ocr_engine: str = "tesseract") -> LabReportResult:
    """Đọc ảnh nhị phân của ② → cụm từ → dòng → cột → ô → OCR → vai trò → cờ."""
    raise NotImplementedError("extract_table")
