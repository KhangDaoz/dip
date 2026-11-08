"""Build a new, checksum-verified T1/T2 v2.1 bundle; never overwrite history."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "results/t1_t2_repro_audit"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=AUDIT / "t1_t2_v2_1_handoff.zip")
    args = parser.parse_args()
    target = args.output.resolve()
    manifest_path = target.with_name("v2_1_handoff_manifest.json")
    report_path = target.with_name("v2_1_handoff_verification.json")
    if any(p.exists() for p in (target, manifest_path, report_path)):
        raise FileExistsError("Choose a new empty destination; refusing to replace a bundle/report")
    verification = json.loads((AUDIT / "v2_verification_current.json").read_text(encoding="utf-8"))
    if (not verification["scores_and_pixel_hashes_match"]
            or verification["replayed_t1_test_rows"] != 120
            or verification["replayed_t2_test_rows"] != 300):
        raise ValueError("Current frozen-result verification is missing or incomplete")
    if sha(ROOT / "data/6_synthetic/manifest_v2.json") != verification["compatibility"]["current_manifest_sha256"]["manifest_v2.json"]:
        raise ValueError("Dataset changed after verification")
    if sha(ROOT / "code/src/synthetic/degrade.py") != verification["compatibility"]["current_generator_sha256"]:
        raise ValueError("Generator changed after verification")
    evidence = AUDIT / "t1_t2_v2_handoff.zip"
    if sha(evidence) != verification["compatibility"]["evidence_archive_sha256"]:
        raise ValueError("Historical evidence ZIP changed")
    frozen_count = 0
    with zipfile.ZipFile(evidence) as old:
        manifest = json.loads(old.read("results/t1_t2_repro_audit/v2_handoff_manifest.json"))["files"]
        for rel, expected in manifest.items():
            if rel.startswith(("results/t1_v2/", "results/t2_v2/")) and not rel.endswith("/README.md"):
                if sha(ROOT / rel) != expected:
                    raise ValueError("Historical result changed: " + rel)
                frozen_count += 1
    files = []
    for directory in ("results/t1_v2", "results/t2_v2"):
        files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    extras = [
        "code/scripts/build_t1_t2_handoff.py",
        "code/experiments/t1_shadow.py", "code/experiments/t2_binarization.py",
        "code/experiments/t1_t2_compat.py", "code/experiments/t1_shadow.md", "code/experiments/t2_binarization.md",
        "code/src/common/__init__.py", "code/src/common/io.py", "code/src/common/paths.py", "code/src/common/seed.py",
        "code/src/cleaning/__init__.py", "code/src/cleaning/background.py", "code/src/cleaning/binarize.py",
        "code/src/cleaning/metrics.py", "code/src/synthetic/__init__.py", "code/src/synthetic/degrade.py",
        "code/tests/test_t1_shadow.py", "code/tests/test_t2_binarization.py", "code/tests/test_t1_t2_compat.py",
        "code/tests/test_cleaning_basic.py", "code/tests/test_cleaning_methods.py", "code/tests/test_cleaning_metrics.py",
        "results/t1_t2_repro_audit/verify_v2.py", "results/t1_t2_repro_audit/v2_verification_current.json",
        "results/t1_t2_repro_audit/v2_verification_current.log", "results/t1_t2_repro_audit/compat_step12_report.json",
        "results/t1_t2_repro_audit/compat_step12_tests.xml", "results/t1_t2_repro_audit/compat_step12_tests.log",
        "results/t1_t2_repro_audit/v2_verification.json", "results/t1_t2_repro_audit/v2_requirements.txt",
        "results/t1_t2_repro_audit/HANDOFF_V2_1.md", "results/t1_t2_repro_audit/t1_t2_v2_handoff.zip",
        "results/primary_dataset_audit/bo6_exposure_update_check.json",
    ]
    files.extend(ROOT / rel for rel in extras)
    files = sorted(set(files))
    for p in files:
        if not p.is_file() or not p.resolve().is_relative_to(ROOT):
            raise ValueError("Missing or out-of-repo package member: " + str(p))
    data = {"version": "2.1", "dataset_included": False, "frozen_result_files_unchanged": frozen_count,
            "files": {p.relative_to(ROOT).as_posix(): sha(p) for p in files}}
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for p in files:
            archive.write(p, p.relative_to(ROOT).as_posix())
        archive.write(manifest_path, "results/t1_t2_repro_audit/v2_1_handoff_manifest.json")
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        for rel, expected in data["files"].items():
            if hashlib.sha256(archive.read(rel)).hexdigest() != expected:
                raise ValueError("ZIP member checksum mismatch: " + rel)
    report = {"version": "2.1", "archive": str(target), "sha256": sha(target),
              "bytes": target.stat().st_size, "members": len(files) + 1,
              "crc_and_member_sha256_match": True, "frozen_result_files_unchanged": frozen_count,
              "historical_evidence_zip_unchanged": True, "dataset_included": False}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
