"""Đường dẫn chuẩn của repo v2 (plan.md §4.1)."""
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]   # <repo>/code
REPO_DIR = CODE_DIR.parent

DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = REPO_DIR / "results"
OUT_DIR = REPO_DIR / "out"
SAMPLES_DIR = CODE_DIR / "samples"
WEIGHTS_DIR = CODE_DIR / "weights"

# VAIPE — bộ dữ liệu chính của v2
VAIPE_DIR = DATA_DIR / "1_vaipe"
VAIPE_SPLITS = {
    "train": VAIPE_DIR / "train",
    "valid": VAIPE_DIR / "valid",
    "test": VAIPE_DIR / "test",
    "test_unlabeled": VAIPE_DIR / "test_unlabeled",
}
VAIPE_DEGRADED = VAIPE_DIR / "test_degraded"   # L0/L2/L4 — do make_degraded.py sinh
VAIPE_CROPS = VAIPE_DIR / "crops"             # cho VietOCR
VAIPE_YOLO = VAIPE_DIR / "yolo"               # nhãn .txt + data.yaml

# Tên file trọng số
YOLO_WEIGHTS = WEIGHTS_DIR / "yolo_vaipe.pt"
VIETOCR_PRETRAINED = WEIGHTS_DIR / "vietocr_pretrained.pth"
VIETOCR_FINETUNED = WEIGHTS_DIR / "vietocr_finetuned.pth"
VIETOCR_CONFIG = CODE_DIR / "configs" / "vietocr_vgg_transformer.yml"   # config lúc fine-tune (max width 1024)


def vaipe_split(split: str) -> Path:
    """Trả về đường dẫn thư mục split ('train'/'valid'/'test')."""
    if split not in VAIPE_SPLITS:
        raise ValueError(f"split phải thuộc {set(VAIPE_SPLITS)}, nhận {split}")
    return VAIPE_SPLITS[split]


def vaipe_images(split: str) -> Path:
    return vaipe_split(split) / "images"


def vaipe_labels(split: str) -> Path:
    return vaipe_split(split) / "labels"
