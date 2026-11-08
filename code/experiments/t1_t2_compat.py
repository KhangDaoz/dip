"""Fail-closed input contracts for derived-shadow T1/T2; never rewrite history.

Manifest changes alone are not proof of compatibility. Historical runs need an
explicit evidence ZIP containing their original config, samples, table and code.
New T1 runs publish a config-bound input contract for all reference assets.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from common.io import load_json
from common.paths import REPO_DIR

ASSETS = ("rel_clean_path", "rel_clean_binary_path", "rel_gt_path")
SAMPLE_FIELDS = ("sample_id", "source_id", "split", "recipe_name", "derived_shadow",
                 "generation_seed", "shadow_level", "shadow_type", "width", "height",
                 "provenance_sample_id", "H_clean_to_derived", *ASSETS)
PROTOCOL_FUNCTIONS = ("asset", "load_v2", "read_pair", "input_hashes", "correct", "measure")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def checked_asset(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Invalid reference path")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-root reference: {relative}")
    return path


def dependency_fingerprint(source, entrypoints):
    """AST closure includes referenced module functions, globals and imports.

    Formatting/comments do not count; unrelated added functions do not count.
    Referenced constants, helper bodies, defaults and imports DO count.
    """
    tree = ast.parse(source)
    definitions, imports = {}, []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.dump(node, include_attributes=False))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        definitions[name.id] = node
    pending, selected = list(entrypoints), {}
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        if name not in definitions:
            raise ValueError(f"Missing protocol definition: {name}")
        node = definitions[name]
        selected[name] = ast.dump(node, include_attributes=False)
        pending.extend(n.id for n in ast.walk(node)
                       if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                       and n.id in definitions and n.id not in selected)
    return sha(json.dumps([imports, sorted(selected.items())], sort_keys=True).encode())


def descriptors(samples):
    result = {}
    for sample in samples:
        sid = sample["source_id"]
        if sid in result:
            raise ValueError("Duplicate protocol source")
        result[sid] = {key: sample.get(key) for key in SAMPLE_FIELDS}
    return result


def required_sources(samples):
    """All val/test + the exact first train source per level/type sanity cell."""
    selected, seen = set(), set()
    for s in samples:
        if s["split"] in ("val", "test"):
            selected.add(s["source_id"])
        elif s["split"] == "train":
            cell = (s["shadow_level"], s["shadow_type"])
            if cell not in seen:
                selected.add(s["source_id"])
                seen.add(cell)
    return selected


def runtime_fingerprints():
    generator = REPO_DIR / "code/src/synthetic/degrade.py"
    protocol = Path(__file__).with_name("t1_shadow.py")
    return {
        "shadow_dependencies": dependency_fingerprint(generator.read_text(encoding="utf-8"), ["shadow"]),
        "protocol_dependencies": dependency_fingerprint(protocol.read_text(encoding="utf-8"), PROTOCOL_FUNCTIONS),
        "implementations": {name: file_sha(REPO_DIR / "code/src/cleaning" / name)
                            for name in ("background.py", "binarize.py")},
        "common_io": file_sha(REPO_DIR / "code/src/common/io.py"),
    }


def build_contract(config, samples):
    assets = {s[key]: file_sha(checked_asset(config["data_dir"], s[key]))
              for s in samples for key in ASSETS}
    return {"schema": 1, "dataset_version": "v2", "seed": config["seed"],
            "samples": descriptors(samples), "assets": assets,
            "versions": config["versions"], "fingerprints": runtime_fingerprints()}


def _archive_contract(archive_path, config_path, config, selection_path, samples):
    """Recover evidence without manufacturing missing historical input hashes."""
    with zipfile.ZipFile(archive_path) as archive:
        manifest_name = "results/t1_t2_repro_audit/v2_handoff_manifest.json"
        manifest = json.loads(archive.read(manifest_name))["files"]

        def read(name):
            data = archive.read(name)
            if manifest.get(name) != sha(data):
                raise ValueError(f"Evidence archive checksum mismatch: {name}")
            return data

        matches = [name for name in manifest if name.endswith("/config.json")
                   and manifest[name] == file_sha(config_path)]
        if len(matches) != 1:
            raise ValueError("Evidence archive does not bind this historical T1 config")
        prefix = matches[0].rsplit("/", 1)[0]
        read(matches[0])
        if sha(read(prefix + "/selection.json")) != file_sha(selection_path):
            raise ValueError("Historical selection differs from evidence archive")
        baseline_samples = json.loads(read(prefix + "/samples.json"))
        if descriptors(baseline_samples) != descriptors(samples):
            raise ValueError("Source/split/seed/geometry/path protocol changed")
        old_generator = read("code/src/synthetic/degrade.py")
        old_protocol = read("code/experiments/t1_shadow.py")
        if sha(old_generator) != config["generator_sha256"] or sha(old_protocol) != config["script_sha256"]:
            raise ValueError("Historical code checksum disagrees with config")
        implementations = {}
        for name, expected in config["implementation_sha256"].items():
            if sha(read("code/src/cleaning/" + name)) != expected:
                raise ValueError("Historical implementation checksum disagrees")
            implementations[name] = expected
        fingerprints = {
            "shadow_dependencies": dependency_fingerprint(old_generator.decode("utf-8"), ["shadow"]),
            "protocol_dependencies": dependency_fingerprint(old_protocol.decode("utf-8"), PROTOCOL_FUNCTIONS),
            "implementations": implementations,
            "common_io": sha(read("code/src/common/io.py")),
        }
        table = list(csv.DictReader(io.StringIO(read(prefix + "/table.csv").decode("utf-8-sig"))))
        baseline = descriptors(baseline_samples)
        covered, assets = set(), {}
        for row in table:
            sid = row["source_id"]
            if sid not in baseline or row["sample_id"] != baseline[sid]["sample_id"] or row["split"] != baseline[sid]["split"]:
                raise ValueError("Evidence CSV source metadata disagrees")
            covered.add(sid)
            for key in ASSETS:
                rel, expected = baseline[sid][key], row[key + "_sha256"]
                if len(expected) != 64 or (rel in assets and assets[rel] != expected):
                    raise ValueError("Missing or conflicting historical asset checksum")
                assets[rel] = expected
        if not required_sources(samples) <= covered:
            raise ValueError("Missing historical hashes for required T2 sources")
        return {"schema": 1, "dataset_version": "v2", "seed": config["seed"],
                "samples": baseline, "assets": assets, "versions": config["versions"],
                "fingerprints": fingerprints}


def validate_t1_inputs(selection_path, root, samples, versions, evidence_archive=None):
    """No writes or pixel decoding. Returns a report or raises ValueError.

    An explicit historical archive is mandatory for old v2 configs without a
    bound input contract. Current checksums are never substituted for old ones.
    """
    selection_path, root = Path(selection_path), Path(root)
    config_path = selection_path.parent / "config.json"
    config = load_json(config_path)
    selection = load_json(selection_path)
    if selection.get("selection_split") != "val":
        raise ValueError("T1 recipe was not selected on val")
    if config.get("selection_sha256") and file_sha(selection_path) != config["selection_sha256"]:
        raise ValueError("T1 selection checksum mismatch")
    manifest_matches = all(file_sha(checked_asset(root, name)) == expected
                           for name, expected in config["input_manifest_sha256"].items())
    if config.get("dataset_version", "legacy") != "v2":
        if not manifest_matches:
            raise ValueError("Legacy dataset manifests changed since T1")
        return {"mode": "legacy_exact_manifest", "manifest_matches": True}
    if config.get("input_contract_sha256"):
        path = selection_path.parent / "input_contract.json"
        if file_sha(path) != config["input_contract_sha256"]:
            raise ValueError("Input contract checksum mismatch")
        contract = load_json(path)
        mode = "bound_input_contract"
    else:
        if evidence_archive is None:
            raise ValueError("Historical T1 requires --compatibility-evidence; no input contract is bound")
        try:
            contract = _archive_contract(Path(evidence_archive), config_path, config, selection_path, samples)
        except (KeyError, zipfile.BadZipFile) as exc:
            raise ValueError("Incomplete or damaged compatibility evidence") from exc
        mode = "historical_evidence_archive"
    if contract.get("schema") != 1 or contract.get("dataset_version") != "v2":
        raise ValueError("Unsupported input contract")
    if contract["seed"] != config["seed"] or contract["samples"] != descriptors(samples):
        raise ValueError("Source/split/seed/geometry/path protocol changed")
    if contract["versions"] != versions:
        raise ValueError("Runtime library versions changed; independent verification required")
    if contract["fingerprints"] != runtime_fingerprints():
        raise ValueError("Relevant generator/protocol/implementation code changed")
    assets = contract["assets"]
    needed = {s[key] for s in samples if s["source_id"] in required_sources(samples) for key in ASSETS}
    if not needed <= set(assets):
        raise ValueError("Input contract omits required assets")
    for relative, expected in assets.items():
        if file_sha(checked_asset(root, relative)) != expected:
            raise ValueError(f"Reference asset checksum changed: {relative}")
    return {"mode": mode, "manifest_matches": manifest_matches,
            "current_manifest_sha256": {n: file_sha(checked_asset(root, n))
                                        for n in config["input_manifest_sha256"]},
            "historical_manifest_sha256": config["input_manifest_sha256"],
            "current_generator_sha256": file_sha(REPO_DIR / "code/src/synthetic/degrade.py"),
            "historical_generator_sha256": config["generator_sha256"],
            "assets_verified": len(assets), "sources_in_protocol": len(samples),
            "reference_scope": "historically_scored_sources" if mode == "historical_evidence_archive" else "bound_contract_assets",
            "required_scored_sources": len(required_sources(samples)),
            "evidence_archive_sha256": file_sha(evidence_archive) if mode == "historical_evidence_archive" else None,
            "historical_config_sha256": file_sha(config_path),
            "t1_selection_sha256": file_sha(selection_path),
            "relevant_code_and_versions_match": True}
