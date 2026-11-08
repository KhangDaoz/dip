"""Build or verify a portable, synthetic CleaningResult integration sample.

This is an interface fixture, not an implementation of cleaning.pipeline.clean.
Run from any directory; JSON paths are always relative to the repository root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from cleaning.background import estimate_background, remove_shadow
from cleaning.binarize import sauvola
from common.contracts import CleaningResult, PreprocessResult
from common.io import load_json, read_image, save_json, write_image
from common.paths import REPO_DIR
from preprocessing.quality import assess_quality

PACKAGE = Path("code/examples/cleaning_handoff_v1")

# This fixture has one fixed recipe. These are not restrictions on the shared
# CleaningResult contract or on future cleaning pipeline configurations.
V1_PARAMS = {
    "stage": "minimal_integration_sample_v1", "background_method": "median",
    "background_kernel": 41, "binarization": "sauvola", "sauvola_window": 31,
    "sauvola_k": .2, "sauvola_r": 128, "clahe_applied": False,
    "mark_removal_applied": False, "masks_computed": False,
    "components_computed": False, "input_origin": "synthetic_already_flat_fixture",
}


def asset_sha256(path: Path) -> str:
    """Git may convert text newlines on checkout; PNGs stay byte-exact."""
    data = path.read_bytes()
    if path.suffix.lower() in {".md", ".py", ".json"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def validate_v1_params(params):
    """Reject missing or misleading declarations before recomputing outputs."""
    if not isinstance(params, dict):
        raise ValueError("v1 params must be an object")
    for name, expected in V1_PARAMS.items():
        if name not in params:
            raise ValueError(f"Missing v1 parameter: {name}")
        value = params[name]
        # In particular, 0/1 must not masquerade as JSON false/true.
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Unsupported v1 parameter {name}: expected {expected!r}, got {value!r}")


def resolve_asset(repo: Path, value: str) -> Path:
    """Do not resolve a repo-relative asset against the JSON file's folder."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Expected a portable repo-relative path: {value!r}")
    p = Path(value)
    resolved = (repo / p).resolve()
    if p.is_absolute() or ".." in p.parts or not resolved.is_relative_to(repo.resolve()):
        raise ValueError(f"Path outside repo: {value!r}")
    if not resolved.is_file():
        raise ValueError(f"Missing asset: {value}")
    return resolved


def synthetic_page(doc_type: str):
    """Already flat documents, with no names or real clinical observations."""
    img = np.full((1100, 800, 3), 240, np.uint8)

    def line(text, x, y, color=(25, 25, 25), scale=.7):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    line("SYNTHETIC INTEGRATION SAMPLE", 55, 65, scale=.85)
    line("NOT A REAL MEDICAL DOCUMENT", 55, 105)
    line("ID: " + doc_type.upper() + "_DEMO", 55, 155)
    if doc_type == "lab_report":
        line("LAB REPORT - MOCK VALUES ONLY", 55, 220)
        line("Parameter", 55, 285)
        line("Value", 335, 285)
        line("Unit", 455, 285)
        line("Reference", 565, 285)
        rows = [("DEMO_A", "12.1", "unit_A", "10.0 - 15.0"),
                ("DEMO_B", "4.2", "unit_B", "3.0 - 5.0"),
                ("DEMO_C", "7.6", "unit_C", "6.0 - 9.0")]
        for i, row in enumerate(rows):
            for x, text in zip([55, 335, 455, 565], row):
                line(text, x, 345 + i * 80)
        line("KEEP THIS RED TEXT", 55, 650, color=(35, 35, 185))
        crop = [45, 315, 710, 45]
    else:
        line("PRESCRIPTION - MOCK ITEMS ONLY", 55, 220)
        line("Item", 55, 285)
        line("Instruction", 345, 285)
        line("DEMO_ITEM_A 500", 55, 350)
        line("MOCK MORNING 1", 345, 350)
        line("DEMO_ITEM_B 4.2", 55, 435)
        line("MOCK EVENING 2", 345, 435)
        line("NO REAL DRUG OR DOSING ADVICE", 55, 650, scale=.65)
        crop = [45, 320, 710, 45]
    # Controlled mild illumination gradient; document geometry remains identity.
    gradient = np.linspace(.68, 1.0, img.shape[1], dtype=np.float32)[None, :, None]
    img = np.rint(img * gradient).astype(np.uint8)
    return img, crop


def verify_package(repo: Path = REPO_DIR) -> dict:
    """Validate actual assets as well as dataclass round trips.

    The shared validate() does not check files, dtype, shape or binary values.
    Crop bboxes below are reader examples, not component or table predictions.
    """
    package = repo / PACKAGE
    manifest = load_json(package / "manifest.json")
    checked = []
    for sample in manifest["samples"]:
        pre = PreprocessResult.from_dict(load_json(resolve_asset(repo, sample["preprocess_json"])))
        data = load_json(resolve_asset(repo, sample["cleaning_json"]))
        if set(data) != {"image_id", "clean_gray_path", "binary_path", "masks", "components", "params"}:
            raise ValueError("CleaningResult fields do not match the current contract")
        result = CleaningResult.from_dict(data)
        validate_v1_params(result.params)
        if CleaningResult.from_dict(json.loads(json.dumps(result.to_dict()))).to_dict() != data:
            raise ValueError("CleaningResult round-trip changed data")
        if result.image_id != pre.image_id:
            raise ValueError("Preprocess and cleaning image IDs differ")
        rect = read_image(resolve_asset(repo, pre.rectified_path))
        gray = read_image(resolve_asset(repo, result.clean_gray_path), gray=True)
        binary = read_image(resolve_asset(repo, result.binary_path), gray=True)
        for value in [result.clean_gray_path, result.binary_path]:
            raw = cv2.imdecode(np.fromfile(resolve_asset(repo, value), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.ndim != 2 or raw.dtype != np.uint8:
                raise ValueError("Expected an actual single-channel uint8 PNG")
        for array in [gray, binary]:
            if array.dtype != np.uint8 or array.ndim != 2 or array.shape != rect.shape[:2]:
                raise ValueError("Image dtype, dimensions or coordinate frame differ")
        if not set(np.unique(binary)) <= {0, 255}:
            raise ValueError("Binary image has values outside {0,255}")
        if not np.any(binary == 255) or not np.any(binary == 0):
            raise ValueError("Sample must contain both ink and paper")
        # Recompute to prove the artifact contains the real baseline algorithm.
        params = result.params
        expected_gray = remove_shadow(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY),
            estimate_background(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY),
                                method=params["background_method"], kernel=params["background_kernel"]))
        if not np.array_equal(gray, expected_gray):
            raise ValueError("clean_gray does not match declared processing")
        expected_binary = sauvola(gray, params["sauvola_window"], params["sauvola_k"])
        if not np.array_equal(binary, expected_binary):
            raise ValueError("binary does not match declared processing/polarity")
        if result.masks != {} or result.components != []:
            raise ValueError("v1 must not claim uncomputed masks or component labels")
        if not np.allclose(pre.homography, np.eye(3)) or pre.skew_deg != 0:
            raise ValueError("Synthetic flat fixture must use identity geometry")
        for name, bounds in sample["reader_crop_bboxes"].items():
            x, y, w, h = bounds
            if any(type(v) is not int for v in bounds) or min(x, y) < 0 or min(w, h) <= 0:
                raise ValueError("Invalid [x,y,w,h] example bbox")
            if x + w > gray.shape[1] or y + h > gray.shape[0]:
                raise ValueError("Example bbox outside image")
            crop = gray[y:y+h, x:x+w]
            if crop.shape != (h, w):
                raise ValueError("Crop dimensions differ")
            saved_crop = read_image(resolve_asset(repo, sample["reader_crop_files"][name]), gray=True)
            if not np.array_equal(crop, saved_crop):
                raise ValueError("OCR crop does not match the same-coordinate grayscale image")
        checked.append({"image_id": result.image_id, "shape_hw": list(gray.shape),
                        "json_roundtrip": "pass", "images_and_polarity": "pass",
                        "declared_params": "pass", "recomputed_outputs": "pass", "reader_crop": "pass"})
    for rel, expected in manifest["sha256"].items():
        path = resolve_asset(repo, rel)
        if asset_sha256(path) != expected:
            raise ValueError(f"Hash mismatch: {rel}")
    return {"status": "passed", "sample_count": len(checked), "samples": checked,
            "scope": "Contract/assets/crop smoke test only; downstream extractors and OCR not executed"}


def build_package(repo: Path = REPO_DIR):
    target = repo / PACKAGE
    if (target / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {target}; use --verify-only to check it")
    target.mkdir(parents=True, exist_ok=True)
    samples = []
    for doc_type in ["lab_report", "prescription"]:
        image_id = f"handoff_{doc_type}_001"
        sub = PACKAGE / image_id
        img, bbox = synthetic_page(doc_type)
        input_path = sub / "rectified.png"
        write_image(repo / input_path, img)
        h, w = img.shape[:2]
        pre = PreprocessResult(image_id=image_id, rectified_path=input_path.as_posix(),
            homography=np.eye(3).tolist(), corners_src=[[0,0], [w-1,0], [w-1,h-1], [0,h-1]],
            corner_method="full_image", skew_deg=0.0, doc_type=doc_type,
            doc_confidence=1.0, quality=assess_quality(img)).validate()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat = remove_shadow(gray, estimate_background(gray, kernel=41))
        binary = sauvola(flat, window=31, k=.2)
        clean_path, binary_path = sub / "clean_gray.png", sub / "binary.png"
        write_image(repo / clean_path, flat)
        write_image(repo / binary_path, binary)
        cleaned = CleaningResult(image_id=image_id, clean_gray_path=clean_path.as_posix(),
            binary_path=binary_path.as_posix(), masks={}, components=[], params=V1_PARAMS.copy()).validate()
        pre_path, clean_json = sub / "preprocess.json", sub / "cleaning_result.json"
        save_json(repo / pre_path, pre.to_dict())
        save_json(repo / clean_json, cleaned.to_dict())
        x,y,cw,ch = bbox
        crop_path = sub / "ocr_crop_example.png"
        write_image(repo / crop_path, flat[y:y+ch, x:x+cw])
        # Viewer-only inversion: never use this panel as binary_path.
        views = [img, cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR), cv2.cvtColor(255-binary, cv2.COLOR_GRAY2BGR)]
        preview = np.hstack([cv2.resize(v, (400,550)) for v in views])
        write_image(repo / sub / "preview.png", preview)
        samples.append({"doc_type": doc_type, "preprocess_json": pre_path.as_posix(),
            "cleaning_json": clean_json.as_posix(), "reader_crop_bboxes": {"example_row": bbox},
            "reader_crop_files": {"example_row": crop_path.as_posix()},
            "provenance": "Generated flat fixture, not Hiếu's preprocessing/classification output; identity H is deliberate. doc_type and confidence are assigned for this fixture, not model predictions."})
    hashes = {p.relative_to(repo).as_posix(): asset_sha256(p)
              for p in sorted(target.rglob("*")) if p.is_file()}
    save_json(target / "manifest.json", {"version": 1, "purpose": "Minimal integration fixture, not clinical/benchmark data",
                                          "sha256_text_newlines": "lf",
                                          "samples": samples, "sha256": hashes})
    report = verify_package(repo)
    save_json(target / "validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    report = verify_package() if args.verify_only else build_package()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
