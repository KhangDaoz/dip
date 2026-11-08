"""T1 scoring and selection must not silently leak test data or mispair images."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "t1_shadow", Path(__file__).resolve().parents[1] / "experiments/t1_shadow.py")
t1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t1)


def val_scores():
    rows = []
    for source in ("a", "b"):
        for method, kernel, psnr, ssim in [("none", 0, 20, .7), ("median", 15, 30, .9),
                                           ("median", 21, 31, .8), ("closing_median", 15, 29, .95)]:
            rows.append(dict(source_id=source, split="val", phase="validation", method=method,
                             kernel=kernel, candidate=t1.candidate_id(method, kernel),
                             psnr_db=psnr, ssim=ssim, ink_mae=2, paper_mae=1))
    return pd.DataFrame(rows)


def test_metrics_have_fixed_range_and_no_intensity_fitting():
    clean = np.full((11, 13), 100, np.uint8)
    ink = np.zeros_like(clean, bool)
    ink[3:6, 4:7] = True
    same = t1.measure(clean, clean, ink)
    assert same["psnr_db"] == float("inf") and same["ssim"] == pytest.approx(1)
    shifted = t1.measure(clean, clean + 10, ink)
    assert shifted["psnr_db"] == pytest.approx(28.1308036087)
    assert shifted["ink_mae"] == shifted["paper_mae"] == 10
    assert 0 < shifted["ssim"] < 1


def test_selection_follows_psnr_then_ssim_and_retains_none_baseline():
    _, best, winner = t1.select_parameters(val_scores())
    assert (winner["method"], winner["kernel"]) == ("median", 21)
    assert set(best.method) == {"none", "median", "closing_median"}
    tied = val_scores()
    tied.loc[tied.candidate == "median_k21", "psnr_db"] = 30
    assert t1.select_parameters(tied)[2]["kernel"] == 15


@pytest.mark.parametrize("split,phase", [("train", "validation"), ("test", "validation"),
                                         ("val", "val_clean_control")])
def test_selection_rejects_train_test_and_control_rows(split, phase):
    rows = val_scores()
    rows.loc[0, ["split", "phase"]] = [split, phase]
    with pytest.raises(ValueError, match="validation rows only"):
        t1.select_parameters(rows)


@pytest.mark.parametrize("bad", ["missing", "duplicate"])
def test_selection_requires_paired_source_sets(bad):
    rows = val_scores()
    rows = rows.iloc[1:] if bad == "missing" else pd.concat([rows, rows.iloc[:1]])
    with pytest.raises(ValueError, match="same val sources"):
        t1.select_parameters(rows)


def test_source_manifest_prevents_cross_split_sources_and_path_escape(tmp_path):
    sources, samples = [], []
    for split in ("train", "val", "test"):
        folder = tmp_path / split
        folder.mkdir()
        sample = dict(source_id=split, sample_id=split, split=split, recipe_name="single_shadow",
                      angle_deg=0, perspective_strength=0, blur_level=0, shadow_level=1)
        sources.append(dict(source_id=split, split=split))
        for key, name in [("rel_degraded_path", "degraded.png"), ("rel_clean_path", "clean.png"),
                          ("rel_clean_binary_path", "clean_binary.png"), ("rel_gt_path", "gt.json")]:
            # Deliberately not images: load_data must not read test pixels.
            (folder / name).write_bytes(b"metadata preflight only")
            sample[key] = f"{split}/{name}"
        samples.append(sample)
    source_manifest = dict(n_sources=3, sources=sources, split_counts=dict(train=1, val=1, test=1))
    (tmp_path / "synthetic_100_sources.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    path = tmp_path / "manifest_v1.json"
    path.write_text(json.dumps(dict(samples=samples)), encoding="utf-8")
    config = {**t1.CONFIG, "data_dir": str(tmp_path)}
    assert len(t1.load_data(config)) == 3
    samples[-1]["split"] = "val"
    path.write_text(json.dumps(dict(samples=samples)), encoding="utf-8")
    with pytest.raises(ValueError, match="crosses splits"):
        t1.load_data(config)
    with pytest.raises(ValueError, match="out-of-root"):
        t1.asset(tmp_path, "../outside.png")


def test_v2_derived_protocol_is_deterministic_and_rejects_cross_split(tmp_path):
    import json
    from common.io import write_image
    rows = []
    for split in ("train", "val", "test"):
        for i in range(6):
            sid = f"{split}_{i}"
            image = np.full((40, 50, 3), 240, np.uint8)
            image[10:30, 20:25] = 20
            paths = {"rel_clean_path": sid+".png", "rel_clean_binary_path": sid+"_binary.png",
                     "rel_gt_path": sid+".json"}
            write_image(tmp_path / paths["rel_clean_path"], image)
            write_image(tmp_path / paths["rel_clean_binary_path"], (image[:,:,0] < 100).astype(np.uint8)*255)
            (tmp_path / paths["rel_gt_path"]).write_text(json.dumps({"source_id": sid, "split": split}))
            rows.append({"source_id": sid, "sample_id": sid+"_v00", "split": split,
                         "variant_index": 0, **paths})
    manifest = {"version": "v2", "n_sources": 18, "split_counts": {s:6 for s in ("train","val","test")}, "samples": rows}
    path = tmp_path / "manifest_v2.json"
    path.write_text(json.dumps(manifest))
    config = {**t1.CONFIG, "dataset_version": "v2", "data_dir": str(tmp_path)}
    data = t1.load_data(config)
    assert data == t1.load_data(config)
    assert len({(s["shadow_level"], s["shadow_type"]) for s in data if s["split"]=="test"}) == 6
    before = (tmp_path / data[0]["rel_clean_path"]).read_bytes()
    a, clean, _ = t1.read_pair(data[0], tmp_path)
    b, _, _ = t1.read_pair(data[0], tmp_path)
    np.testing.assert_array_equal(a, b)
    assert np.any(a != clean)
    assert before == (tmp_path / data[0]["rel_clean_path"]).read_bytes()
    assert "rel_degraded_path_sha256" not in t1.input_hashes(data[0], tmp_path, a)
    manifest["samples"].append({**rows[0], "split": "test", "variant_index": 1})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="crosses splits"):
        t1.load_data(config)
