"""③ Ghép K1–K7 theo hợp đồng LabReportResult (common/contracts.py)."""
from common.contracts import CleaningResult, LabReportResult


def extract_table(cleaned: CleaningResult) -> LabReportResult:
    """Ảnh nhị phân phiếu xét nghiệm → cụm từ → dòng → cột → ô → dòng chỉ số kèm bbox (+ OCR, cờ)."""
    raise NotImplementedError("extract_table")
