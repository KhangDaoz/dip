"""T2: five binarizers x no correction / frozen T1 shadow correction.

Tune on val single_shadow samples; freeze all ten recipes before reading test
pixels. Reference is synthetic Otsu(clean), NOT a manual transcription mask.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import skimage

from cleaning.binarize import METHODS, otsu
from cleaning.metrics import BinaryReference
from common.io import load_json, read_image, save_json
from common.paths import DATASETS, REPO_DIR, RESULTS_DIR
from common.seed import set_seed
import t1_shadow as t1

CONFIG = {
    "seed": 42, "recipe": "single_shadow", "windows": [15, 31, 61],
    "niblack_k": [-.1, -.2, -.3], "sauvola_k": [.1, .2, .3, .5],
    "wolf_k": [.2, .5, .8], "gaussian_c": [5, 10, 15],
    "branches": ["none", "t1"],
    "selection": "mean per-image F1 desc, pFM-skeleton desc, DRD asc, method/branch/parameter ID asc",
    "reference": "Otsu(clean_binary.png), auto-generated, not manual; favors Otsu-like decisions",
    "scope": "single_shadow only, same original resolution and identity geometry as T1; not full v1/T3",
    "pfm": "H-DIBCO 2010: ordinary precision and recall on Zhang skeleton of GT; not weighted pFM 2013+",
    "drd": "5x5 reciprocal-distance weights, valid neighbors only, nonuniform 8x8 blocks incl partial blocks",
    "aggregation": "macro image means; summary PSNR=-10log10(mean per-image binary error fraction); also store mean per-image PSNR and exact-match count",
    "test_policy": "one val-selected recipe per method/branch; frozen before any test pixels are read",
}


def candidate(method, branch, params):
    encoded = json.dumps(params, sort_keys=True, separators=(",", ":"))
    label = "_".join(f"{k}{v}" for k, v in sorted(params.items())) or "base"
    return {"method": method, "branch": branch, "params": params,
            "params_json": encoded, "candidate": f"{branch}__{method}__{label}"}


def grid(config):
    recipes = []
    for branch in config["branches"]:
        recipes.append(candidate("otsu", branch, {}))
        for method in ("niblack", "sauvola", "wolf"):
            for window, k in itertools.product(config["windows"], config[method + "_k"]):
                recipes.append(candidate(method, branch, {"window": window, "k": k}))
        for block, c in itertools.product(config["windows"], config["gaussian_c"]):
            recipes.append(candidate("gaussian", branch, {"block": block, "c": c}))
    return recipes


def defaults():
    params = {"otsu": {}, "niblack": {"window": 31, "k": -.2},
              "sauvola": {"window": 31, "k": .2}, "wolf": {"window": 31, "k": .5},
              "gaussian": {"block": 31, "c": 10}}
    return [candidate(m, b, p) for b in CONFIG["branches"] for m, p in params.items()]


def summary(frame):
    def from_mse(values):
        mean = float(values.mean())
        return float("inf") if mean == 0 else -10*np.log10(mean)
    return (frame.groupby(["method", "branch", "params_json", "candidate"], as_index=False)
            .agg(n_images=("source_id", "nunique"), f1=("f1", "mean"),
                 pfm_skeleton=("pfm_skeleton", "mean"), drd=("drd", "mean"),
                 precision=("precision", "mean"), recall=("recall", "mean"),
                 pseudo_recall=("pseudo_recall", "mean"),
                 psnr_db=("error_fraction", from_mse),
                 mean_image_psnr_db=("psnr_db", lambda v: float(np.mean(v))),
                 exact_matches=("error_pixels", lambda v: int((v == 0).sum()))))


def select_parameters(validation):
    if validation.empty or set(validation.split) != {"val"} or set(validation.phase) != {"validation"}:
        raise ValueError("Selection requires validation rows only")
    sources = set(validation.source_id)
    for _, rows in validation.groupby("candidate"):
        if len(rows) != len(sources) or set(rows.source_id) != sources:
            raise ValueError("All candidates require the same val sources exactly once")
    if validation[["f1", "pfm_skeleton", "drd"]].isna().any().any():
        raise ValueError("Undefined selection metrics")
    ranking = summary(validation).sort_values(
        ["f1", "pfm_skeleton", "drd", "method", "branch", "candidate"],
        ascending=[False, False, True, True, True, True]).reset_index(drop=True)
    selected = ranking.drop_duplicates(["method", "branch"])
    if set(zip(selected.method, selected.branch)) != set(itertools.product(METHODS, CONFIG["branches"])):
        raise ValueError("Expected all five methods in both branches")
    recipes = [candidate(r.method, r.branch, json.loads(r.params_json)) for r in selected.itertuples()]
    return ranking, recipes


def read_sample(sample, root):
    degraded, clean, _ = t1.read_pair(sample, root)
    binary = read_image(t1.asset(root, sample["rel_clean_binary_path"]), gray=True)
    if not np.array_equal(otsu(clean), binary):
        raise ValueError("Reference does not match declared Otsu(clean): " + sample["sample_id"])
    return degraded, clean, binary


def branch_inputs(image, shadow_recipe):
    return {"none": image, "t1": t1.correct(image, shadow_recipe["method"], shadow_recipe["kernel"])}


def run(data, config, recipes, output):
    records = []
    root = Path(config["data_dir"])
    for sample in data:
        degraded, clean, binary = read_sample(sample, root)
        reference = BinaryReference(binary)
        if reference.nubn == 0 or reference.n_skeleton == 0:
            raise ValueError("Document reference lacks nonuniform blocks or text skeleton")
        image = clean if config.get("clean_control", False) else degraded
        inputs = branch_inputs(image, config["shadow_recipe"])
        hashes = t1.input_hashes(sample, root, degraded)
        for recipe in recipes:
            start = time.perf_counter()
            predicted = METHODS[recipe["method"]](inputs[recipe["branch"]], **recipe["params"])
            elapsed = (time.perf_counter()-start)*1000
            scores = reference.score(predicted)
            records.append({"sample_id": sample["sample_id"], "source_id": sample["source_id"],
                            "split": sample["split"], "phase": config["phase"],
                            "shadow_level": sample["shadow_level"], "shadow_type": sample["shadow_type"],
                            **{k: recipe[k] for k in ("method", "branch", "params_json", "candidate")},
                            **scores, "error_fraction": scores["error_pixels"] / predicted.size,
                            "binarizer_ms": elapsed, "reference_matches_clean_otsu": True, **hashes})
        print(f'{config["phase"]}: {sample["sample_id"]} ({len(recipes)} candidates)', flush=True)
        # A checkpoint after each source; no need to rerun test to recover a CSV.
        save_table(pd.DataFrame(records), output, config["phase"])
    return pd.DataFrame(records)


def save_table(frame, output, name):
    frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")


def panel(images, labels, height=650):
    tiles = []
    for array, label in zip(images, labels):
        im = Image.fromarray(array).convert("RGB")
        im = im.resize((round(im.width*height/im.height), height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (im.width, height+32), "white")
        tile.paste(im, (0, 32)); ImageDraw.Draw(tile).text((6, 8), label, fill="black")
        tiles.append(tile)
    canvas = Image.new("RGB", (sum(t.width for t in tiles), height+32), "white")
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0)); x += tile.width
    return canvas


def save_val_examples(data, root, recipes, shadow_recipe, output):
    output.mkdir()
    for level in (1, 2, 3):
        sample = next(s for s in data if s["shadow_level"] == level)
        degraded, _, binary = read_sample(sample, root)
        inputs = branch_inputs(degraded, shadow_recipe)
        for branch in CONFIG["branches"]:
            current = [r for r in recipes if r["branch"] == branch]
            images, labels = [degraded, 255-binary], ["Degraded gray", "Otsu clean reference"]
            for r in current:
                images.append(255-METHODS[r["method"]](inputs[branch], **r["params"]))
                labels.append(r["method"]+" "+r["params_json"])
            panel(images, labels).save(output / f"val_level{level}_{branch}.png")
            h, w = degraded.shape
            crops = [im[int(.15*h):int(.6*h), int(.04*w):int(.96*w)] for im in images]
            panel(crops, labels, 500).save(output / f"val_level{level}_{branch}_detail.png")


def save_development_preview(selection, shadow_recipe, output):
    """Unscored local train/development images; no synthetic masks for color cases."""
    manifest_path = REPO_DIR / "img_test/manifest.json"
    if not manifest_path.is_file():
        return
    output.mkdir()
    ranking = pd.DataFrame(selection["ranking"])
    winners = ranking.drop_duplicates("branch")
    recipes = [candidate(r.method, r.branch, json.loads(r.params_json)) for r in winners.itertuples()]
    for entry in load_json(manifest_path):
        source = manifest_path.parent / entry["file"]
        if t1.digest(source) != entry["sha256"]:
            raise ValueError("Development image changed")
        bgr = read_image(source)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        inputs = branch_inputs(gray, shadow_recipe)
        images, labels = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)], ["Original color - unscored"]
        for r in recipes:
            images.append(255-METHODS[r["method"]](inputs[r["branch"]], **r["params"]))
            labels.append(r["branch"]+" / "+r["method"]+" "+r["params_json"])
        panel(images, labels, 1000).save(output / (source.stem+".png"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATASETS[6])
    parser.add_argument("--t1-selection", type=Path, default=RESULTS_DIR / "t1_v2/selection.json")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "t2_v2")
    parser.add_argument("--compatibility-evidence", type=Path,
                        help="Explicit original handoff ZIP for historical T1 without an input contract")
    parser.add_argument("--check-inputs-only", action="store_true",
                        help="Verify T1 compatibility without measuring or writing results")
    return parser.parse_args(argv)


def main():
    from t1_t2_compat import validate_t1_inputs

    args = parse_args()
    output, root = args.output_dir.resolve(), args.data_dir.resolve()
    prior = load_json(args.t1_selection)
    if prior["selection_split"] != "val":
        raise ValueError("T1 recipe was not selected on val")
    prior_config = load_json(args.t1_selection.parent / "config.json")
    shadow_recipe = {k: prior["winner"][k] for k in ("method", "kernel")}
    config = {**CONFIG, "seed": prior_config["seed"],
              "dataset_version": prior_config.get("dataset_version", "legacy"),
              "recipe": prior_config.get("recipe", "single_shadow"),
              "scope": prior_config.get("generation", CONFIG["scope"]),
              "generation": prior_config.get("generation"),
              "generator_sha256": prior_config.get("generator_sha256"), "data_dir": str(root), "shadow_recipe": shadow_recipe,
              "t1_selection_sha256": t1.digest(args.t1_selection),
              "input_manifest_sha256": prior_config["input_manifest_sha256"],
              "versions": {"python": platform.python_version(), "opencv": cv2.__version__,
                           "skimage": skimage.__version__, "numpy": np.__version__},
              "code_sha256": {str(p.relative_to(REPO_DIR)): t1.digest(p) for p in
                  [Path(__file__).resolve(), Path(t1.__file__).resolve(),
                   Path(__file__).with_name("t1_t2_compat.py"),
                   REPO_DIR/"code/src/cleaning/binarize.py", REPO_DIR/"code/src/cleaning/background.py",
                   REPO_DIR/"code/src/cleaning/metrics.py"]}}
    data = t1.load_data(config)
    report = validate_t1_inputs(args.t1_selection, root, data, config["versions"], args.compatibility_evidence)
    if args.check_inputs_only:
        print(json.dumps({**report, "output_written": False}, ensure_ascii=True), flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite T2 results; select a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "compatibility.json", report)
    config["compatibility_sha256"] = t1.digest(output / "compatibility.json")
    config["t1_input_manifest_sha256"] = prior_config["input_manifest_sha256"]
    config["input_manifest_sha256"] = {name: t1.digest(root / name)
                                       for name in prior_config["input_manifest_sha256"]}
    if config["dataset_version"] == "v2":
        config["t1_generator_sha256"] = prior_config["generator_sha256"]
        config["generator_sha256"] = t1.digest(REPO_DIR / "code/src/synthetic/degrade.py")
    save_json(output / "config.json", config)
    if config["dataset_version"] == "v2":
        save_json(output / "samples.json", data)
    set_seed(config["seed"])
    seen, sanity = set(), []
    for s in (s for s in data if s["split"] == "train"):
        key = (s["shadow_level"], s["shadow_type"])
        if key not in seen:
            seen.add(key); sanity.append(s)
    train = run(sanity, {**config, "phase": "train_sanity"}, defaults(), output)
    val = [s for s in data if s["split"] == "val"]
    validation = run(val, {**config, "phase": "validation"}, grid(config), output)
    ranking, recipes = select_parameters(validation)
    save_table(ranking, output, "validation_ranking")
    # Exclude infinite-PSNR diagnostics from JSON; all actual selection metrics
    # are finite F1/pFM/DRD. No arbitrary PSNR cap is introduced.
    columns = ["method", "branch", "params_json", "candidate", "f1", "pfm_skeleton", "drd"]
    ranking_json = ranking[columns].to_dict(orient="records")
    selection = {"selection_split": "val", "rule": config["selection"],
                 "frozen_before_test_pixels": True, "reference_is_auto_otsu": True,
                 "winner": ranking_json[0], "frozen_recipes": recipes, "ranking": ranking_json}
    save_json(output / "selection.json", selection)
    print("FROZEN WINNER: " + json.dumps(selection["winner"]), flush=True)
    test = [s for s in data if s["split"] == "test"]
    test_scores = run(test, {**config, "phase": "heldout_test"}, recipes, output)
    save_table(summary(test_scores), output, "test_summary")
    controls = run(val, {**config, "phase": "val_clean_control", "clean_control": True}, recipes, output)
    save_table(summary(controls), output, "clean_control_summary")
    save_table(pd.concat([train, validation, test_scores, controls], ignore_index=True), output, "table")
    scores = summary(test_scores)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for branch in config["branches"]:
        group = scores[scores.branch == branch].set_index("method").reindex(list(METHODS))
        axes[0].plot(list(METHODS), group.f1*100, marker="o", label=branch)
        axes[1].plot(list(METHODS), group.drd, marker="o", label=branch)
    for ax, label in zip(axes, ["F1 (%) - test", "DRD - test (lower is better)"]):
        ax.set_title(label); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"figure.png", dpi=180); fig.savefig(output/"figure.pdf"); plt.close(fig)
    save_val_examples(val, root, recipes, shadow_recipe, output/"examples")
    if config["dataset_version"] != "v2":
        save_development_preview(selection, shadow_recipe, output/"development_unscored")
    print("COMPLETE: " + str(output), flush=True)


if __name__ == "__main__":
    main()
