"""Receiver smoke test; no OCR or downstream extraction is claimed."""
from pathlib import Path
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
import numpy as np

from common.contracts import CleaningResult, PreprocessResult
from common.io import load_json
from common.paths import REPO_DIR


def asset(value):
    path = (REPO_DIR / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(REPO_DIR.resolve()):
        raise ValueError("Asset is outside the repository")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def image(value):
    # Decode unchanged so unexpected color masks are rejected rather than hidden
    # by an automatic grayscale conversion during this validation.
    arr = cv2.imdecode(np.fromfile(asset(value), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Could not decode {value}")
    return arr


def check(condition, message):
    if not condition:
        raise ValueError(message)


def asset_sha256(path):
    data = path.read_bytes()
    if path.suffix.lower() in {".md", ".py", ".json"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def main():
    folder = Path(__file__).resolve().parent
    manifest = load_json(folder / "manifest.json")
    for name, digest in manifest["sha256"].items():
        check(asset_sha256(asset(name)) == digest, f"Hash mismatch: {name}")
    for sample in manifest["samples"]:
        pre = PreprocessResult.from_dict(load_json(asset(sample["preprocess_json"])))
        result = CleaningResult.from_dict(load_json(asset(sample["cleaning_json"])))
        check(result.image_id == pre.image_id, "Image IDs differ")
        rect, gray, binary = image(pre.rectified_path), image(result.clean_gray_path), image(result.binary_path)
        check(rect.ndim == 3 and rect.shape[2] == 3 and rect.dtype == np.uint8, "Expected BGR input")
        for a in [gray, binary]:
            check(a.ndim == 2 and a.dtype == np.uint8 and a.shape == rect.shape[:2], "Image frame/type differs")
        check(set(np.unique(binary)) == {0, 255}, "Expected binary ink=255/background=0")
        check(result.masks == {} and result.components == [], "Unexpected v1 optional outputs")
        x,y,w,h = sample["reader_crop_bboxes"]["example_row"]
        check(0 <= x < x+w <= gray.shape[1] and 0 <= y < y+h <= gray.shape[0], "Crop out of bounds")
        crop = gray[y:y+h, x:x+w]
        expected = image(sample["reader_crop_files"]["example_row"])
        check(np.array_equal(crop, expected), "Example OCR crop differs")
        # Coordinates in this fixture map exactly to the prepared input frame.
        check(np.allclose(pre.to_source([[x,y], [x+w,y+h]]), [[x,y], [x+w,y+h]]), "Identity mapping differs")
        print(f"PASS {result.image_id}: gray/binary {gray.shape}, OCR crop {crop.shape}, masks/components not computed")
    print("PASS: both interface samples readable. OCR and downstream extractors NOT run.")


if __name__ == "__main__":
    main()
