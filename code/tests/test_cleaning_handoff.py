"""Integration fixture must actually be consumable and portable."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "handoff_builder", Path(__file__).resolve().parents[1] / "scripts/build_cleaning_handoff.py")
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def test_build_read_crop_and_detect_tampering(tmp_path):
    assert handoff.build_package(tmp_path)["sample_count"] == 2
    assert handoff.verify_package(tmp_path)["status"] == "passed"
    manifest = json.loads((tmp_path / handoff.PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    json_file = tmp_path / manifest["samples"][0]["cleaning_json"]
    data = json.loads(json_file.read_text(encoding="utf-8"))
    data["binary_path"] = "out/missing.png"
    json_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing asset"):
        handoff.verify_package(tmp_path)


@pytest.mark.parametrize("path", ["../outside.png", "/outside.png", "C:/outside.png", "out\\image.png"])
def test_reject_nonportable_asset_paths(tmp_path, path):
    with pytest.raises(ValueError):
        handoff.resolve_asset(tmp_path, path)


@pytest.mark.parametrize("key,value", [
    ("background_method", "closing"),
    ("binarization", "otsu"),
    ("sauvola_r", 64),
    ("background_kernel", 43),
    ("sauvola_window", 33),
    ("sauvola_k", .3),
    ("clahe_applied", True),
    ("mark_removal_applied", True),
    ("masks_computed", True),
    ("components_computed", True),
    ("input_origin", "real_preprocessing_output"),
    ("stage", "complete_pipeline"),
    ("clahe_applied", 0),
    ("sauvola_r", "128"),
    ("params", None),
])
def test_reject_incorrect_declared_params_even_with_valid_hash(tmp_path, key, value):
    handoff.build_package(tmp_path)
    manifest_path = tmp_path / handoff.PACKAGE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = manifest["samples"][0]["cleaning_json"]
    json_file = tmp_path / relative
    data = json.loads(json_file.read_text(encoding="utf-8"))
    if key == "params":
        data["params"] = value
    else:
        data["params"][key] = value
    json_file.write_text(json.dumps(data), encoding="utf-8")
    # Keep the checksum valid: failure must come from semantic validation.
    manifest["sha256"][relative] = hashlib.sha256(json_file.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    match = "v1 params must be an object" if key == "params" else f"Unsupported v1 parameter {key}:"
    with pytest.raises(ValueError, match=match):
        handoff.verify_package(tmp_path)


@pytest.mark.parametrize("key", ["background_method", "binarization", "sauvola_r"])
def test_reject_missing_declared_parameter(tmp_path, key):
    handoff.build_package(tmp_path)
    manifest = json.loads((tmp_path / handoff.PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    json_file = tmp_path / manifest["samples"][0]["cleaning_json"]
    data = json.loads(json_file.read_text(encoding="utf-8"))
    del data["params"][key]
    json_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=f"Missing v1 parameter: {key}"):
        handoff.verify_package(tmp_path)


def test_git_text_newline_conversion_keeps_package_readable(tmp_path):
    handoff.build_package(tmp_path)
    manifest = json.loads((tmp_path / handoff.PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    path = tmp_path / manifest["samples"][0]["cleaning_json"]
    original = path.read_bytes().replace(b"\r\n", b"\n")
    path.write_bytes(original.replace(b"\n", b"\r\n"))
    assert handoff.verify_package(tmp_path)["status"] == "passed"
    # Newline normalization must not hide an actual content change.
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="Hash mismatch"):
        handoff.verify_package(tmp_path)
