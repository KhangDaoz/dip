"""Đường dẫn chuẩn của repo. Mọi script lấy đường dẫn từ đây, không viết cứng."""
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]   # <repo>/code
REPO_DIR = CODE_DIR.parent

DATA_DIR = REPO_DIR / "data"          # dữ liệu gốc bộ 1–5 (không commit)
RESULTS_DIR = REPO_DIR / "results"    # CSV + hình của thí nghiệm (không commit)
OUT_DIR = REPO_DIR / "out"            # ảnh trung gian của pipeline (không commit)
SAMPLES_DIR = CODE_DIR / "samples"    # 5–10 ảnh mẫu, CÓ commit, dùng cho demo + mốc tích hợp

# Tên thư mục con của từng bộ dữ liệu trong data/
DATASETS = {
    1: DATA_DIR / "1_vaipe",
    2: DATA_DIR / "2_roboflow_report_keywords",   # 57 ảnh gốc; bản tăng cường ở 2_roboflow_report_keywords_augmented/
    3: DATA_DIR / "3_other_rvl_cdip",
    4: DATA_DIR / "4_self_captured",
    5: DATA_DIR / "5_synthetic",
}
