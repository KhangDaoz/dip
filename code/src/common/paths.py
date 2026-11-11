from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = CODE_DIR.parent

DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = REPO_DIR / "results"
OUT_DIR = REPO_DIR / "out"
SAMPLES_DIR = CODE_DIR / "samples"
WEIGHTS_DIR = CODE_DIR / "weights"

VAIPE_DIR = DATA_DIR / "1_vaipe"
VAIPE_SPLITS = {
    "train": VAIPE_DIR / "train",
    "valid": VAIPE_DIR / "valid",
    "test": VAIPE_DIR / "test",
    "test_unlabeled": VAIPE_DIR / "test_unlabeled",
}
VAIPE_DEGRADED = VAIPE_DIR / "test_degraded"
VAIPE_CROPS = VAIPE_DIR / "crops"
VAIPE_YOLO = VAIPE_DIR / "yolo"

YOLO_WEIGHTS = WEIGHTS_DIR / "yolo_vaipe.pt"
VIETOCR_PRETRAINED = WEIGHTS_DIR / "vietocr_pretrained.pth"
VIETOCR_FINETUNED = WEIGHTS_DIR / "vietocr_finetuned.pth"
VIETOCR_CONFIG = CODE_DIR / "configs" / "vietocr_vgg_transformer.yml"


def vaipe_split(split: str) -> Path:
    if split not in VAIPE_SPLITS:
        raise ValueError(f"split phải thuộc {set(VAIPE_SPLITS)}, nhận {split}")
    return VAIPE_SPLITS[split]


def vaipe_images(split: str) -> Path:
    return vaipe_split(split) / "images"


def vaipe_labels(split: str) -> Path:
    return vaipe_split(split) / "labels"
