"""T1: select grayscale shadow removal on val, evaluate frozen recipes on test.

From repo root: python code/experiments/t1_shadow.py --data-dir <dataset6-root>
Only single_shadow samples are compared, without geometry or blur confounds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import skimage
from skimage.metrics import structural_similarity

from cleaning.background import estimate_background, remove_shadow
from common.io import load_json, read_image, save_json
from common.paths import DATASETS, REPO_DIR, RESULTS_DIR
from common.seed import set_seed

EXP_ID = "t1"
CONFIG = {
    "seed": 42, "methods": ["none", "median", "closing_median"],
    "kernels": [15, 21, 31, 41, 61, 81],
    "recipe": "single_shadow", "selection_split": "val",
    "selection_rule": "mean per-image PSNR desc, SSIM desc, smaller kernel, method name",
    "metrics": {"data_range": 255, "ssim_win_size": 7, "ssim_gaussian_weights": False,
                "ssim_use_sample_covariance": True},
    "training_sanity": "first source per (shadow_level, shadow_type), kernel=41",
    "test_policy": "none plus one val-selected kernel per method, no test tuning",
    "preprocessing": "BGR to grayscale, original resolution, no intensity fitting or alignment",
    "background_border": "replicate", "closing_se": "square",
    "control": "val clean images processed with frozen recipes; diagnostic only",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-root dataset asset: {relative}")
    return path


def candidate_id(method, kernel):
    return "none" if method == "none" else f"{method}_k{kernel}"


def candidates(config):
    return [("none", 0)] + [(m, k) for m in config["methods"] if m != "none" for k in config["kernels"]]


def input_hashes(sample, root, degraded):
    keys = ("rel_clean_path", "rel_clean_binary_path", "rel_gt_path")
    if not sample.get("derived_shadow"):
        keys += ("rel_degraded_path",)
    hashes = {k + "_sha256": digest(asset(root, sample[k])) for k in keys}
    if sample.get("derived_shadow"):
        hashes["derived_gray_pixels_sha256"] = hashlib.sha256(degraded.tobytes()).hexdigest()
        hashes["generation_seed"] = sample["generation_seed"]
    return hashes


def load_v2(config):
    root = Path(config["data_dir"]).resolve()
    manifest = load_json(root / "manifest_v2.json")
    if manifest["version"] != "v2":
        raise ValueError("Expected v2 manifest")
    groups = {}
    for row in manifest["samples"]:
        groups.setdefault(row["source_id"], []).append(row)
    if len(groups) != manifest["n_sources"]:
        raise ValueError("Incorrect v2 source count")
    selected = []
    for sid, rows in sorted(groups.items()):
        if len({r["split"] for r in rows}) != 1:
            raise ValueError("Source crosses splits: " + sid)
        bases = [r for r in rows if r["variant_index"] == 0]
        if len(bases) != 1 or bases[0]["split"] not in ("train", "val", "test"):
            raise ValueError("Expected one v00 provenance row per source")
        row = dict(bases[0])
        for key in ("rel_clean_path", "rel_clean_binary_path", "rel_gt_path"):
            asset(root, row[key])
        selected.append(row)
    if Counter(r["split"] for r in manifest["samples"]) != Counter(manifest["split_counts"]):
        raise ValueError("Manifest split counts disagree")
    for split in ("train", "val", "test"):
        for rank, row in enumerate(r for r in selected if r["split"] == split):
            row["provenance"] = dict(row)
            row["provenance_sample_id"] = row["sample_id"]
            row.pop("rel_degraded_path", None)
            row.update(angle_deg=0, perspective_strength=0, blur_level=0, blur_kind="none",
                       glare_level=0, stamp_level=0, pen_level=0, jpeg_noise_level=0,
                       has_glare=False, has_stamp=False, has_pen=False,
                       applied_degradations="shadow", has_shadow=True,
                       H_clean_to_derived=[[1,0,0],[0,1,0],[0,0,1]])
            row["shadow_level"] = (rank // 2) % 3 + 1
            row["shadow_type"] = ("planar", "silhouette")[rank % 2]
            token = f'{config["seed"]}:{row["source_id"]}:derived_shadow_v2'
            row["generation_seed"] = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
            row["sample_id"] += "_derived_shadow_v2"
            row["derived_shadow"] = True
            row["recipe_name"] = "derived_shadow_v2"
    return selected


def load_data(config: dict):
    """Audit source membership and return paths without reading test images."""
    if config.get("dataset_version") == "v2":
        return load_v2(config)
    root = Path(config["data_dir"]).resolve()
    manifest = load_json(root / "manifest_v1.json")
    source_manifest = load_json(root / "synthetic_100_sources.json")
    sources = source_manifest["sources"]
    by_id = {s["source_id"]: s for s in sources}
    if len(by_id) != len(sources) or len(sources) != source_manifest["n_sources"]:
        raise ValueError("Duplicate source IDs or incorrect source count")
    if dict(Counter(s["split"] for s in sources)) != source_manifest["split_counts"]:
        raise ValueError("Source split counts disagree")
    samples = manifest["samples"]
    if {s["source_id"] for s in samples} != set(by_id):
        raise ValueError("Source manifest and dataset manifest contain different sources")
    for sample in samples:
        if sample["split"] != by_id[sample["source_id"]]["split"]:
            raise ValueError("Source crosses splits: " + sample["source_id"])
    selected = sorted((s for s in samples if s["recipe_name"] == config["recipe"]), key=lambda s: s["source_id"])
    if Counter(s["source_id"] for s in selected) != Counter({sid: 1 for sid in by_id}):
        raise ValueError("Expected exactly one single-shadow sample per source")
    for sample in selected:
        if (sample["angle_deg"] != 0 or sample["perspective_strength"] != 0 or sample["blur_level"] != 0
                or sample["shadow_level"] not in (1, 2, 3)):
            raise ValueError("Single-shadow sample has geometry/blur confound")
        for key in ("rel_degraded_path", "rel_clean_path", "rel_clean_binary_path", "rel_gt_path"):
            asset(root, sample[key])
    return selected


def read_pair(sample, root):
    if sample.get("derived_shadow"):
        from synthetic.degrade import shadow
        gt = load_json(asset(root, sample["rel_gt_path"]))
        if gt["source_id"] != sample["source_id"] or gt["split"] != sample["split"]:
            raise ValueError("Source provenance disagrees")
        color = read_image(asset(root, sample["rel_clean_path"]))
        damaged, _ = shadow(color, level=sample["shadow_level"],
                            shadow_type=sample["shadow_type"],
                            rng=np.random.default_rng(sample["generation_seed"]))
        clean = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        degraded = cv2.cvtColor(damaged, cv2.COLOR_BGR2GRAY)
        binary = read_image(asset(root, sample["rel_clean_binary_path"]), gray=True)
        if binary.shape != clean.shape or not set(np.unique(binary)) <= {0, 255}:
            raise ValueError("Invalid clean-frame binary reference")
        return degraded, clean, binary > 0
    gt = load_json(asset(root, sample["rel_gt_path"]))
    if (gt["source_id"] != sample["source_id"] or gt["split"] != sample["split"]
            or gt["recipe_name"] != "single_shadow"
            or not np.allclose(np.asarray(gt["H"], dtype=float), np.eye(3), rtol=0, atol=1e-12)
            or set(gt["degradation_params"]) != {"shadow"}):
        raise ValueError("Ground-truth metadata/coordinate frame disagrees with single-shadow sample")
    degraded = cv2.cvtColor(read_image(asset(root, sample["rel_degraded_path"])), cv2.COLOR_BGR2GRAY)
    clean = cv2.cvtColor(read_image(asset(root, sample["rel_clean_path"])), cv2.COLOR_BGR2GRAY)
    binary = read_image(asset(root, sample["rel_clean_binary_path"]), gray=True)
    if (degraded.shape != clean.shape or binary.shape != clean.shape
            or list(clean.shape[::-1]) != [sample["width"], sample["height"]]
            or not set(np.unique(binary)) <= {0, 255}):
        raise ValueError("Pair dimensions or reference mask values disagree")
    return degraded, clean, binary > 0


def correct(gray, method, kernel):
    return gray.copy() if method == "none" else remove_shadow(gray, estimate_background(gray, method, kernel))


def measure(clean, predicted, ink):
    """Full-resolution per-image scores; no fitting of intensities to the GT."""
    difference = clean.astype(np.float64) - predicted.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    return {
        "psnr_db": float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse)),
        "ssim": float(structural_similarity(clean, predicted, data_range=255, win_size=7,
                                           gaussian_weights=False, use_sample_covariance=True)),
        "ink_mae": float(np.abs(difference[ink]).mean()) if ink.any() else float("nan"),
        "paper_mae": float(np.abs(difference[~ink]).mean()) if (~ink).any() else float("nan"),
    }


def run(data, config: dict) -> pd.DataFrame:
    root = Path(config["data_dir"])
    records = []
    for sample in data:
        degraded, clean, ink = read_pair(sample, root)
        image = clean if config.get("clean_control", False) else degraded
        hashes = input_hashes(sample, root, degraded)
        for method, kernel in config["candidates"]:
            start = time.perf_counter()
            out = correct(image, method, kernel)
            elapsed = (time.perf_counter() - start) * 1000
            scores = measure(clean, out, ink)
            records.append({"sample_id": sample["sample_id"], "source_id": sample["source_id"],
                            "split": sample["split"], "phase": config["phase"],
                            "shadow_level": sample["shadow_level"], "shadow_type": sample["shadow_type"],
                            "method": method, "kernel": kernel, "candidate": candidate_id(method, kernel),
                            "width": clean.shape[1], "height": clean.shape[0],
                            **scores, "processing_ms": elapsed,
                            **hashes})
        print(f'{config["phase"]}: {sample["sample_id"]} ({len(config["candidates"])} recipes)', flush=True)
    return pd.DataFrame(records)


def summary(df):
    return (df.groupby(["method", "kernel", "candidate"], as_index=False)
            .agg(n_images=("source_id", "nunique"), psnr_db=("psnr_db", "mean"),
                 ssim=("ssim", "mean"), ink_mae=("ink_mae", "mean"), paper_mae=("paper_mae", "mean")))


def select_parameters(validation):
    """Rank only comparable val rows; never use test/train scores for selection."""
    if validation.empty or set(validation["split"]) != {"val"} or set(validation["phase"]) != {"validation"}:
        raise ValueError("Parameter selection requires validation rows only")
    ids = set(validation.source_id)
    for _, group in validation.groupby("candidate"):
        if len(group) != len(ids) or set(group.source_id) != ids:
            raise ValueError("Candidates must have one score for each of the same val sources")
    if validation[["psnr_db", "ssim"]].isna().any().any():
        raise ValueError("Cannot select from missing metrics")
    ranking = summary(validation).sort_values(["psnr_db", "ssim", "kernel", "method"],
                                             ascending=[False, False, True, True]).reset_index(drop=True)
    best = ranking.drop_duplicates("method")
    return ranking, best, ranking.iloc[0].to_dict()


def save_table(frame, output, name):
    frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")


def plot(df: pd.DataFrame):
    scores = summary(df).sort_values("method")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ["psnr_db", "ssim"], ["PSNR (dB)", "SSIM"]):
        ax.bar(scores.candidate, scores[metric], color=["#3978a8", "#5b9d70", "#b19a61"][:len(scores)])
        ax.set_title(title + " - held-out test")
        ax.tick_params(axis="x", labelrotation=18)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    return fig


def save_examples(samples, root, recipes, output):
    output.mkdir()
    # One predetermined val example per shadow level, not best-looking ones.
    for level in (1, 2, 3):
        sample = next(s for s in samples if s["shadow_level"] == level)
        degraded, clean, _ = read_pair(sample, root)
        views = [("Clean reference", clean), ("Shadow input", degraded)]
        views += [(candidate_id(m, k), correct(degraded, m, k)) for m, k in recipes if m != "none"]
        x1, y1, x2, y2 = int(clean.shape[1]*.04), int(clean.shape[0]*.05), int(clean.shape[1]*.96), int(clean.shape[0]*.55)
        for detail in (False, True):
            tiles = []
            for label, array in views:
                array = array[y1:y2, x1:x2] if detail else array
                im = Image.fromarray(array).convert("RGB")
                height = 650 if detail else 850
                im = im.resize((round(im.width * height / im.height), height), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (im.width, im.height+30), "white")
                tile.paste(im, (0, 30))
                ImageDraw.Draw(tile).text((8, 8), label, fill="black")
                tiles.append(tile)
            panel = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height), "white")
            x = 0
            for tile in tiles:
                panel.paste(tile, (x, 0)); x += tile.width
            panel.save(output / f'level_{level}_{"detail" if detail else "comparison"}.png')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", choices=["legacy", "v2"], default="v2")
    parser.add_argument("--data-dir", type=Path, default=DATASETS[5])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-inputs-only", action="store_true",
                        help="Validate metadata/reference files without running or writing results")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = RESULTS_DIR / ("t1_v2" if args.dataset_version == "v2" else EXP_ID)
    return args


def main():
    from t1_t2_compat import build_contract

    args = parse_args()
    output = args.output_dir.resolve()
    config = {**CONFIG, "dataset_version": args.dataset_version, "data_dir": str(args.data_dir.resolve()),
              "compatibility_code_sha256": digest(Path(__file__).with_name("t1_t2_compat.py")),
              "versions": {"python": platform.python_version(), "opencv": cv2.__version__,
                           "numpy": np.__version__, "skimage": skimage.__version__},
              "script_sha256": digest(Path(__file__)),
              "implementation_sha256": {name: digest(REPO_DIR / "code/src/cleaning" / name)
                                          for name in ("background.py", "binarize.py")}}
    if args.dataset_version == "v2":
        config["methods"] = ["none", "median", "closing", "closing_median"]
        config["recipe"] = "derived_shadow_v2"
    data = load_data(config)
    config["input_manifest_sha256"] = {name: digest(args.data_dir / name)
        for name in (("manifest_v2.json",) if args.dataset_version == "v2" else ("manifest_v1.json", "synthetic_100_sources.json"))}
    if args.dataset_version == "v2":
        config["generation"] = "derived_shadow_v2: clean BGR, identity H, levels .2/.45/.7; sorted source rank cycles level/type within split; SHA256 seed"
        config["generator_sha256"] = digest(REPO_DIR / "code/src/synthetic/degrade.py")
    contract = build_contract(config, data) if args.dataset_version == "v2" else None
    if args.check_inputs_only:
        print(json.dumps({"dataset_version": args.dataset_version, "sources": len(data),
                          "reference_assets": len(contract["assets"]) if contract else None,
                          "output_written": False}), flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a previous T1 run; choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    if contract is not None:
        save_json(output / "input_contract.json", contract)
        config["input_contract_sha256"] = digest(output / "input_contract.json")
        save_json(output / "samples.json", data)
    save_json(output / "config.json", config)
    set_seed(config["seed"])
    train = [s for s in data if s["split"] == "train"]
    sanity, seen = [], set()
    for sample in train:
        key = (sample["shadow_level"], sample["shadow_type"])
        if key not in seen:
            sanity.append(sample); seen.add(key)
    train_scores = run(sanity, {**config, "phase": "train_sanity",
                               "candidates": [(m, 0 if m == "none" else 41) for m in config["methods"]]})
    save_table(train_scores, output, "train_sanity")
    val = [s for s in data if s["split"] == "val"]
    validation = run(val, {**config, "phase": "validation", "candidates": candidates(config)})
    save_table(validation, output, "validation")
    ranking, best, winner = select_parameters(validation)
    save_table(ranking, output, "validation_ranking")
    recipes = [(r.method, int(r.kernel)) for r in best.itertuples()]
    selection = {"selection_split": "val", "rule": config["selection_rule"],
                 "winner": winner, "best_per_method": best.to_dict(orient="records"),
                 "frozen_test_candidates": recipes, "frozen_before_test_images_read": True}
    # Persist frozen choices before reading ANY test pixels.
    save_json(output / "selection.json", selection)
    config["selection_sha256"] = digest(output / "selection.json")
    save_json(output / "config.json", config)
    print("FROZEN: " + json.dumps(selection, ensure_ascii=True), flush=True)
    test = [s for s in data if s["split"] == "test"]
    test_scores = run(test, {**config, "phase": "heldout_test", "candidates": recipes})
    save_table(test_scores, output, "test")
    save_table(summary(test_scores), output, "test_summary")
    grouped = test_scores.groupby(["method", "kernel", "shadow_level", "shadow_type"], as_index=False).agg(
        n_images=("source_id", "nunique"), psnr_db=("psnr_db", "mean"), ssim=("ssim", "mean"))
    save_table(grouped, output, "test_by_shadow")
    controls = run(val, {**config, "phase": "val_clean_control", "clean_control": True, "candidates": recipes})
    save_table(controls, output, "clean_control")
    save_table(summary(controls), output, "clean_control_summary")
    save_table(pd.concat([train_scores, validation, test_scores, controls], ignore_index=True), output, "table")
    fig = plot(test_scores)
    fig.savefig(output / "figure.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "figure.pdf", bbox_inches="tight")
    plt.close(fig)
    save_examples(val, args.data_dir.resolve(), recipes, output / "examples")
    print("COMPLETE: " + str(output), flush=True)


if __name__ == "__main__":
    main()
