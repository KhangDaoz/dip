"""Compatibility needs positive evidence, not an ignore-checksum switch."""
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import t1_shadow as t1
import t2_binarization as t2
import t1_t2_compat as compat


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def history(tmp_path):
    root, run = tmp_path / "data", tmp_path / "run"
    root.mkdir(); run.mkdir()
    rows = []
    for split in ("train", "val", "test"):
        row = dict(source_id=split, sample_id=split + "_v00", split=split,
                   variant_index=0, width=50, height=40)
        for key in compat.ASSETS:
            path = split + "_" + key + ".dat"
            row[key] = path
            # Metadata/hash preflight must not try to decode these bytes.
            (root / path).write_bytes(path.encode())
        rows.append(row)
    manifest = dict(version="v2", n_sources=3, split_counts=dict(train=1, val=1, test=1), samples=rows)
    dump(root / "manifest_v2.json", manifest)
    protocol_path = Path(t1.__file__)
    generator_path = compat.REPO_DIR / "code/src/synthetic/degrade.py"
    original_generator = generator_path.read_bytes() + b"\n# unrelated old file metadata\n"
    versions = dict(python=t1.platform.python_version(), opencv=t1.cv2.__version__,
                    numpy=t1.np.__version__, skimage=t1.skimage.__version__)
    config = dict(dataset_version="v2", seed=42, data_dir=str(root), versions=versions,
                  generator_sha256=compat.sha(original_generator),
                  script_sha256=compat.file_sha(protocol_path),
                  implementation_sha256=compat.runtime_fingerprints()["implementations"],
                  input_manifest_sha256={"manifest_v2.json": compat.file_sha(root / "manifest_v2.json")})
    samples = t1.load_data(config)
    selection = dict(selection_split="val", winner=dict(method="closing", kernel=81))
    dump(run / "config.json", config); dump(run / "selection.json", selection)
    dump(run / "samples.json", samples)
    fields = ["sample_id", "source_id", "split", *[k + "_sha256" for k in compat.ASSETS]]
    table = io.StringIO(); writer = csv.DictWriter(table, fieldnames=fields); writer.writeheader()
    for s in samples:
        writer.writerow({**{k: s[k] for k in fields[:3]},
                         **{k + "_sha256": compat.file_sha(root / s[k]) for k in compat.ASSETS}})
    members = {"results/t1_v2/" + n: (run / n).read_bytes()
               for n in ("config.json", "selection.json", "samples.json")}
    members["results/t1_v2/table.csv"] = table.getvalue().encode()
    members["code/experiments/t1_shadow.py"] = protocol_path.read_bytes()
    members["code/src/synthetic/degrade.py"] = original_generator
    for name in config["implementation_sha256"]:
        members["code/src/cleaning/" + name] = (compat.REPO_DIR / "code/src/cleaning" / name).read_bytes()
    members["code/src/common/io.py"] = (compat.REPO_DIR / "code/src/common/io.py").read_bytes()
    archive = tmp_path / "history.zip"

    def write_archive():
        checksums = {k: compat.sha(v) for k, v in members.items()}
        with zipfile.ZipFile(archive, "w") as z:
            for key, value in members.items(): z.writestr(key, value)
            z.writestr("results/t1_t2_repro_audit/v2_handoff_manifest.json", json.dumps({"files": checksums}))

    write_archive()
    return dict(root=root, run=run, config=config, samples=samples, manifest=manifest,
                archive=archive, members=members, write_archive=write_archive, versions=versions)


def validate(h, archive=True):
    samples = t1.load_data({**h["config"], "data_dir": str(h["root"])})
    return compat.validate_t1_inputs(h["run"] / "selection.json", h["root"], samples,
                                     h["versions"], h["archive"] if archive else None)


def bind_contract(h):
    c = h["config"]
    dump(h["run"] / "input_contract.json", compat.build_contract(c, h["samples"]))
    c["input_contract_sha256"] = compat.file_sha(h["run"] / "input_contract.json")
    c["selection_sha256"] = compat.file_sha(h["run"] / "selection.json")
    dump(h["run"] / "config.json", c)


def test_cli_defaults_are_v2_and_legacy_is_explicit():
    assert t1.parse_args([]).dataset_version == "v2"
    assert t1.parse_args([]).output_dir.name == "t1_v2"
    legacy = t1.parse_args(["--dataset-version", "legacy"])
    assert legacy.output_dir.name == "t1"
    assert t2.parse_args([]).t1_selection.parent.name == "t1_v2"
    assert t2.parse_args([]).output_dir.name == "t2_v2"


def test_historical_compatibility_allows_only_proven_unrelated_manifest_change(history, monkeypatch):
    h = history
    h["manifest"]["exposure_variants"] = [10, 14]
    h["manifest"]["samples"][0]["exposure_gain"] = 1.0
    dump(h["root"] / "manifest_v2.json", h["manifest"])
    monkeypatch.setattr(t1, "read_image", lambda *a, **k: pytest.fail("Pixel decode in preflight"))
    before = (h["run"] / "config.json").read_bytes()
    report = validate(h)
    assert report["mode"] == "historical_evidence_archive" and not report["manifest_matches"]
    assert report["assets_verified"] == 9
    assert (h["run"] / "config.json").read_bytes() == before


def test_historical_run_without_evidence_is_rejected_even_when_manifest_matches(history):
    with pytest.raises(ValueError, match="compatibility-evidence"):
        validate(history, archive=False)


@pytest.mark.parametrize("key", compat.ASSETS)
def test_changed_reference_file_is_rejected_without_trusting_manifest(history, key):
    (history["root"] / history["samples"][0][key]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Reference asset checksum changed"):
        validate(history)


@pytest.mark.parametrize("field,value", [("seed", 43)])
def test_seed_change_rejected(history, field, value):
    history["config"][field] = value
    with pytest.raises(ValueError, match="protocol changed"):
        validate(history)


def test_split_change_rejected(history):
    h = history
    h["manifest"]["samples"][0]["split"] = "test"
    h["manifest"]["split_counts"] = dict(val=1, test=2)
    dump(h["root"] / "manifest_v2.json", h["manifest"])
    with pytest.raises(ValueError, match="protocol changed"):
        validate(h)


def test_changed_selection_or_wrong_archive_cannot_authorize_compatibility(history):
    h = history
    dump(h["run"] / "selection.json", dict(selection_split="val", winner=dict(method="median", kernel=15)))
    with pytest.raises(ValueError, match="selection differs"):
        validate(h)


def test_incomplete_evidence_missing_required_source_is_rejected(history):
    h = history
    key = "results/t1_v2/table.csv"
    lines = h["members"][key].decode().splitlines()
    h["members"][key] = ("\n".join(lines[:-1]) + "\n").encode()
    h["write_archive"]()
    with pytest.raises(ValueError, match="Missing historical hashes"):
        validate(h)


def test_tampered_zip_member_rejected(history):
    h = history
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(h["archive"], "a") as z:
            z.writestr("results/t1_v2/table.csv", b"tampered")
    with pytest.raises(ValueError, match="Evidence archive checksum"):
        validate(h)


def test_runtime_version_change_rejected(history):
    history["versions"] = {**history["versions"], "numpy": "different"}
    with pytest.raises(ValueError, match="library versions"):
        validate(history)


def test_relevant_code_change_rejected(history, monkeypatch):
    real = compat.runtime_fingerprints
    monkeypatch.setattr(compat, "runtime_fingerprints", lambda: {**real(), "shadow_dependencies": "different"})
    with pytest.raises(ValueError, match="Relevant generator"):
        validate(history)


def test_dependency_closure_ignores_unrelated_addition_but_checks_helpers_constants_imports():
    source = "import numpy as np\nK=2\ndef helper(x): return x*K\ndef shadow(x): return helper(x)\n"
    fingerprint = compat.dependency_fingerprint(source, ["shadow"])
    assert compat.dependency_fingerprint(source + "def exposure(x): return x*0.8\n", ["shadow"]) == fingerprint
    for changed in (source.replace("K=2", "K=3"), source.replace("x*K", "x+K"), source.replace("numpy", "math")):
        assert compat.dependency_fingerprint(changed, ["shadow"]) != fingerprint


def test_bound_contract_supports_positive_evidence_without_zip(history):
    h = history; bind_contract(h)
    h["manifest"]["unrelated_glare_update"] = True
    dump(h["root"] / "manifest_v2.json", h["manifest"])
    report = validate(h, archive=False)
    assert report["mode"] == "bound_input_contract" and not report["manifest_matches"]


@pytest.mark.parametrize("target", ["input_contract.json", "selection.json"])
def test_bound_contract_and_selection_tampering_rejected(history, target):
    bind_contract(history)
    (history["run"] / target).write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch|not selected on val"):
        validate(history, archive=False)


def test_reference_path_escape_rejected(history):
    h = history; h["manifest"]["samples"][0]["rel_clean_path"] = "../outside"
    dump(h["root"] / "manifest_v2.json", h["manifest"])
    with pytest.raises(ValueError, match="out-of-root"):
        validate(h)


def test_t2_preflight_does_not_measure_or_create_output(history, monkeypatch, capsys):
    h = history; output = h["run"] / "no_output"
    monkeypatch.setattr(sys, "argv", ["t2", "--data-dir", str(h["root"]),
                        "--t1-selection", str(h["run"] / "selection.json"),
                        "--compatibility-evidence", str(h["archive"]),
                        "--output-dir", str(output), "--check-inputs-only"])
    monkeypatch.setattr(t2, "run", lambda *a, **k: pytest.fail("Measurement in preflight"))
    t2.main()
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["output_written"] is False


def test_t1_preflight_does_not_measure_or_create_output(history, monkeypatch, capsys):
    h = history; output = h["run"] / "no_output"
    monkeypatch.setattr(sys, "argv", ["t1", "--data-dir", str(h["root"]),
                        "--output-dir", str(output), "--check-inputs-only"])
    monkeypatch.setattr(t1, "run", lambda *a, **k: pytest.fail("Measurement in preflight"))
    t1.main()
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["reference_assets"] == 9


def test_t2_rejects_existing_output_after_successful_preflight(history, monkeypatch):
    h = history
    sentinel = h["run"] / "untouched.txt"
    sentinel.write_bytes(b"historical data")
    monkeypatch.setattr(sys, "argv", ["t2", "--data-dir", str(h["root"]),
                        "--t1-selection", str(h["run"] / "selection.json"),
                        "--compatibility-evidence", str(h["archive"]), "--output-dir", str(h["run"])])
    monkeypatch.setattr(t2, "run", lambda *a, **k: pytest.fail("Overwriting existing output"))
    with pytest.raises(FileExistsError, match="overwrite T2"):
        t2.main()
    assert sentinel.read_bytes() == b"historical data"


def test_t1_rejects_existing_output_without_overwriting(history, monkeypatch):
    h = history
    before = (h["run"] / "config.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["t1", "--data-dir", str(h["root"]), "--output-dir", str(h["run"])])
    with pytest.raises(FileExistsError, match="overwrite a previous T1"):
        t1.main()
    assert (h["run"] / "config.json").read_bytes() == before


def test_bound_contract_cannot_omit_required_reference_hash(history):
    h = history; bind_contract(h)
    path = h["run"] / "input_contract.json"
    contract = json.loads(path.read_text())
    del contract["assets"][h["samples"][0]["rel_clean_path"]]
    dump(path, contract)
    h["config"]["input_contract_sha256"] = compat.file_sha(path)
    dump(h["run"] / "config.json", h["config"])
    with pytest.raises(ValueError, match="omits required assets"):
        validate(h, archive=False)


def test_t2_inherits_predecessor_seed_instead_of_its_default(history, monkeypatch, capsys):
    h = history; h["config"]["seed"] = 7
    h["samples"] = t1.load_data(h["config"])
    bind_contract(h)
    monkeypatch.setattr(sys, "argv", ["t2", "--data-dir", str(h["root"]),
                        "--t1-selection", str(h["run"] / "selection.json"), "--check-inputs-only"])
    t2.main()
    assert json.loads(capsys.readouterr().out)["relevant_code_and_versions_match"]
