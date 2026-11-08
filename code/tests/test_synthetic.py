"""Bộ sinh bộ 5 (Thắng)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from common.paths import DATASETS
from common.testing import todo
from synthetic import degrade

CONFIG = Path(__file__).resolve().parents[1] / "src" / "synthetic" / "config.yaml"


def test_config_is_consistent():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for version, names in cfg["versions"].items():
        assert set(names) <= set(cfg["degradations"]), version
    assert set(cfg["degradations"]) <= set(degrade.DEGRADATIONS)
    assert sum(cfg["split"].values()) == pytest.approx(1.0)
    assert tuple(cfg["exposure_before_glare"]["gain_range"]) == degrade.EXPOSURE_GAIN_RANGE


@todo
def test_rotate_records_angle(page):
    out, gt = degrade.rotate(page, 10, np.random.default_rng(0))
    assert gt["angle_deg"] == 10 and out.ndim == 3


@todo
def test_perspective_gt_consistent(page):
    out, gt = degrade.perspective(page, 0.10, np.random.default_rng(0))
    H, corners = np.asarray(gt["H"], float), np.asarray(gt["corners"], float)
    assert H.shape == (3, 3) and corners.shape == (4, 2)
    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    assert np.allclose(cv2.perspectiveTransform(src, H).reshape(4, 2), corners, atol=1.0)


@todo
def test_blur_levels_monotonic(page):
    rng = np.random.default_rng(0)
    scores = [cv2.Laplacian(cv2.cvtColor(degrade.blur(page, lv, rng)[0], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
              for lv in (1, 2, 3, 4)]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("name, mask_key", [("glare", "glare_mask"), ("stamp", "stamp_mask"),
                                            ("shadow", "shadow_mask"), ("pen", "pen_mask")])
@todo
def test_degradation_saves_mask(page, name, mask_key):
    out, gt = degrade.DEGRADATIONS[name](page, 2, np.random.default_rng(0))
    assert out.shape == page.shape
    assert gt[mask_key].shape == page.shape[:2] and (gt[mask_key] > 0).any()


def test_blur_motion_and_reproducibility(page):
    # Monotonic decrease in Laplacian variance for kind="motion"
    scores = []
    for r in (0.0, 1.0, 2.5, 4.5):
        rng = np.random.default_rng(42)
        out, gt = degrade.blur(page, radius=r, rng=rng, kind="motion")
        assert out.shape == page.shape
        assert out.dtype == np.uint8
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)

    # Reproducibility with same seed
    rng1 = np.random.default_rng(123)
    out1, gt1 = degrade.blur(page, radius=2.5, rng=rng1, kind="motion")
    rng2 = np.random.default_rng(123)
    out2, gt2 = degrade.blur(page, radius=2.5, rng=rng2, kind="motion")
    assert np.array_equal(out1, out2)
    assert gt1["angle_deg"] == gt2["angle_deg"]
    assert gt1["ksize"] == gt2["ksize"]

    # Difference with gaussian blur
    out_gauss, _ = degrade.blur(page, radius=2.5, rng=np.random.default_rng(123), kind="gaussian")
    assert not np.array_equal(out1, out_gauss)

    # Invalid kind raises ValueError
    with pytest.raises(ValueError, match="kind không hợp lệ"):
        degrade.blur(page, radius=1.0, rng=np.random.default_rng(0), kind="invalid_kind")

    # radius <= 0 returns copy without blur
    out_zero, gt_zero = degrade.blur(page, radius=0.0, rng=np.random.default_rng(0))
    assert np.array_equal(out_zero, page)
    assert gt_zero["radius"] == 0.0


def test_shadow_levels_and_reproducibility(page):
    # Monotonic decrease in mean intensity as darkness increases with fixed seed
    means = []
    for d in (0.2, 0.45, 0.7):
        rng = np.random.default_rng(42)
        out, gt = degrade.shadow(page, darkness=d, rng=rng)
        assert out.shape == page.shape
        assert out.dtype == np.uint8
        assert gt["mask"].shape == page.shape[:2]
        assert gt["mask"].dtype == np.uint8
        assert np.array_equal(gt["mask"], gt["shadow_mask"])
        means.append(out.mean())
    assert means[0] > means[1] > means[2]
    assert means[-1] > 50.0

    # Reproducibility with same seed
    rng1 = np.random.default_rng(999)
    out1, gt1 = degrade.shadow(page, darkness=0.45, rng=rng1)
    rng2 = np.random.default_rng(999)
    out2, gt2 = degrade.shadow(page, darkness=0.45, rng=rng2)
    assert np.array_equal(out1, out2)
    assert np.array_equal(gt1["mask"], gt2["mask"])
    assert gt1["shadow_type"] == gt2["shadow_type"]

    # Coverage ratio in realistic range [0.20, 0.75]
    assert 0.20 <= gt1["coverage_ratio"] <= 0.75

    # Both shadow types supported
    rng = np.random.default_rng(10)
    _, gt_planar = degrade.shadow(page, darkness=0.45, rng=rng, shadow_type="planar")
    assert gt_planar["shadow_type"] == "planar"
    _, gt_silh = degrade.shadow(page, darkness=0.45, rng=rng, shadow_type="silhouette")
    assert gt_silh["shadow_type"] == "silhouette"


def test_blur_and_shadow_grayscale(page_gray):
    rng = np.random.default_rng(7)
    out_b, gt_b = degrade.blur(page_gray, radius=2.0, rng=rng, kind="gaussian")
    assert out_b.shape == page_gray.shape
    assert out_b.dtype == np.uint8

    out_bm, gt_bm = degrade.blur(page_gray, radius=2.0, rng=rng, kind="motion")
    assert out_bm.shape == page_gray.shape
    assert out_bm.dtype == np.uint8

    out_s, gt_s = degrade.shadow(page_gray, darkness=0.45, rng=rng)
    assert out_s.shape == page_gray.shape
    assert out_s.dtype == np.uint8
    assert gt_s["mask"].shape == page_gray.shape


def test_perspective_levels_and_invertibility(page):
    h, w = page.shape[:2]
    src_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    disps = []
    for s in (0.05, 0.10, 0.20):
        rng = np.random.default_rng(42)
        out, gt = degrade.perspective(page, strength=s, rng=rng)
        assert out.shape == page.shape
        assert out.dtype == np.uint8
        assert gt["page_mask"].shape == (h, w)
        assert (gt["page_mask"] > 0).any()

        H = gt["H"]
        assert abs(np.linalg.det(H)) > 1e-4

        H_inv = np.array(gt["H_degraded_to_clean"])
        assert np.allclose(H @ H_inv, np.eye(3), atol=1e-5)

        corners = gt["corners"]
        assert corners.shape == (4, 2)
        assert (corners[:, 0] >= 0.0).all() and (corners[:, 0] <= w - 1.0).all()
        assert (corners[:, 1] >= 0.0).all() and (corners[:, 1] <= h - 1.0).all()

        disp = float(np.linalg.norm(corners - src_corners, axis=1).mean())
        disps.append(disp)

    # Average corner displacement strictly increases with strength
    assert disps[0] < disps[1] < disps[2]

    # Reproducibility with same seed
    out1, gt1 = degrade.perspective(page, 0.10, rng=np.random.default_rng(101))
    out2, gt2 = degrade.perspective(page, 0.10, rng=np.random.default_rng(101))
    assert np.array_equal(out1, out2)
    assert np.allclose(gt1["H"], gt2["H"])

    # strength <= 0 returns identity and unwarped copy
    out0, gt0 = degrade.perspective(page, 0.0)
    assert np.array_equal(out0, page)
    assert np.allclose(gt0["H"], np.eye(3))


def test_rotate_angles_and_matrix_consistency(page, page_gray):
    h, w = page.shape[:2]
    for a in (0, 5, 10, 20, 30, -15):
        out, gt = degrade.rotate(page, angle_deg=a)
        assert gt["angle_deg"] == a
        assert out.shape == page.shape
        assert out.dtype == np.uint8
        assert gt["page_mask"].shape == (h, w)

        H_rot = gt["H"]
        H_inv = np.array(gt["H_degraded_to_clean"])
        assert np.allclose(H_rot @ H_inv, np.eye(3), atol=1e-5)

    # Angle 0 returns identical image and identity matrix
    out0, gt0 = degrade.rotate(page, 0)
    assert np.array_equal(out0, page)
    assert np.allclose(gt0["H"], np.eye(3))

    # Works on 2D grayscale
    out_g, gt_g = degrade.rotate(page_gray, 15)
    assert out_g.shape == page_gray.shape
    assert out_g.ndim == 2
    assert out_g.dtype == np.uint8


def test_transform_boxes_accuracy(page):
    h, w = page.shape[:2]
    boxes = [
        [100, 150, 200, 50],
        [50, 400, 300, 60],
        [60, 800, 250, 45],
    ]

    out, gt = degrade.perspective(page, strength=0.10, rng=np.random.default_rng(0))
    t_boxes = degrade.transform_boxes(boxes, gt["H"], img_shape=(h, w))
    assert len(t_boxes) == len(boxes)

    for i, tb in enumerate(t_boxes):
        assert "polygon" in tb and len(tb["polygon"]) == 4
        assert "bbox" in tb and len(tb["bbox"]) == 4
        assert "visible_ratio" in tb and 0.0 <= tb["visible_ratio"] <= 1.0
        assert tb["valid"] is True

        bx, by, bw, bh = tb["bbox"]
        assert 0 <= bx <= w and 0 <= by <= h
        assert bw > 0 and bh > 0

    # Dict annotation input preserved
    dict_boxes = [{"bbox": [100, 150, 200, 50], "label": "drugname"}]
    t_dict = degrade.transform_boxes(dict_boxes, gt["H"], img_shape=(h, w))
    assert t_dict[0]["label"] == "drugname"
    assert "polygon" in t_dict[0]

    # Clipping detection when box rotated outside canvas
    _, gt_rot = degrade.rotate(page, 30)
    t_edge = degrade.transform_box([w - 60, 20, 50, 50], gt_rot["H"], img_shape=(h, w))
    assert t_edge["clipped"] is True
    assert t_edge["visible_ratio"] < 0.95


def test_generate_v1_end_to_end(tmp_path, page):
    from synthetic import generate as synth_generate

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    out_dir_1 = tmp_path / "out1"
    out_dir_2 = tmp_path / "out2"

    # Create 3 small mock source images
    for i in range(3):
        img_mock = page.copy()
        cv2.putText(img_mock, f"DOC_{i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(str(src_dir / f"doc_{i:02d}.png"), img_mock)
        if i == 0:
            mock_label = [
                {"id": 1, "label": "drugname", "text": "Paracetamol", "box": [50, 80, 250, 110]}
            ]
            (src_dir / f"doc_{i:02d}.json").write_text(json.dumps(mock_label), encoding="utf-8")

    cfg_path = tmp_path / "config.yaml"
    cfg_content = """seed: 42
n_source_images: 3
variants_per_image: 2
split: {train: 0.6, val: 0.1, test: 0.3}
degradations:
  perspective: {levels: [0.05, 0.10, 0.20]}
  rotate: {levels: [0, 5, 10, 20, 30]}
  blur: {levels: [1, 2, 3, 4], kinds: [gaussian, motion]}
  shadow: {levels: [1, 2, 3]}
versions:
  v1: [perspective, rotate, blur, shadow]
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")

    # Run generator run 1
    res1 = synth_generate.generate(config_path=cfg_path, src_dir=src_dir, out_dir=out_dir_1, version="v1")
    assert res1["status"] == "SUCCESS"
    assert res1["total_samples"] == 3 * 2  # 6 samples

    # Check manifest files exist
    assert (out_dir_1 / "manifest_v1.json").exists()
    assert (out_dir_1 / "manifest_v1.csv").exists()

    # Check each sample in new structured layout (images/, labels/, clean/, clean_binary/, shadow_mask/, page_mask/)
    img_files = sorted(out_dir_1.glob("*/images/*.png"))
    assert len(img_files) == 6

    for img_p in img_files:
        sample_id = img_p.stem
        sp_dir = img_p.parent.parent

        clean_p = sp_dir / "clean" / f"{sample_id}.png"
        clean_bin_p = sp_dir / "clean_binary" / f"{sample_id}.png"
        shadow_p = sp_dir / "shadow_mask" / f"{sample_id}.png"
        page_p = sp_dir / "page_mask" / f"{sample_id}.png"
        gt_p = sp_dir / "labels" / f"{sample_id}.json"

        assert clean_p.exists()
        assert clean_bin_p.exists()
        assert shadow_p.exists()
        assert page_p.exists()
        assert gt_p.exists()

        # Check clean binary format (ink = 255, background = 0)
        c_bin = cv2.imread(str(clean_bin_p), cv2.IMREAD_GRAYSCALE)
        assert c_bin is not None and c_bin.ndim == 2
        assert set(np.unique(c_bin)).issubset({0, 255})
        assert np.mean(c_bin == 255) < 0.50

        # Check gt.json structure
        with open(gt_p, "r", encoding="utf-8") as gf:
            gt_data = json.load(gf)
        assert "sample_id" in gt_data
        assert "source_id" in gt_data
        assert "split" in gt_data
        assert "version" in gt_data and gt_data["version"] == "v1"
        assert "recipe_name" in gt_data
        assert "H" in gt_data and len(gt_data["H"]) == 3
        assert "corners" in gt_data and len(gt_data["corners"]) == 4
        assert "masks" in gt_data and "shadow_mask" in gt_data["masks"]
        assert "gt_coverage" in gt_data
        assert "medicine_region_clean" in gt_data
        assert "medicine_region_degraded" in gt_data

    # Check reproducibility (run 2 gives byte-identical degraded images and same labels)
    res2 = synth_generate.generate(config_path=cfg_path, src_dir=src_dir, out_dir=out_dir_2, version="v1")
    assert res2["total_samples"] == 6

    for img_p1 in img_files:
        sample_id = img_p1.stem
        rel_img = img_p1.relative_to(out_dir_1)
        img_p2 = out_dir_2 / rel_img
        assert img_p2.exists()

        deg1 = cv2.imread(str(img_p1))
        deg2 = cv2.imread(str(img_p2))
        assert np.array_equal(deg1, deg2), f"Degraded image not identical for {rel_img}"

        sp_dir1 = img_p1.parent.parent
        sp_dir2 = img_p2.parent.parent
        gt_p1 = sp_dir1 / "labels" / f"{sample_id}.json"
        gt_p2 = sp_dir2 / "labels" / f"{sample_id}.json"

        with open(gt_p1, "r", encoding="utf-8") as f1, open(gt_p2, "r", encoding="utf-8") as f2:
            gt1, gt2 = json.load(f1), json.load(f2)
            assert gt1["sample_id"] == gt2["sample_id"]
            assert gt1["recipe_name"] == gt2["recipe_name"]
            assert np.allclose(gt1["H"], gt2["H"])
            assert np.allclose(gt1["corners"], gt2["corners"])


def test_recipe_v09_dark_table_and_corners_in_frame(page):
    from synthetic.generate import _apply_recipe_v1
    h, w = page.shape[:2]
    src_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)

    for src_idx in (0, 1, 2, 3):
        rng = np.random.default_rng(src_idx + 100)
        res = _apply_recipe_v1(page, src_idx=src_idx, var_idx=9, sample_rng=rng)
        assert res["recipe_name"] == "combo_full_v1"

        corners = np.array(res["corners"])
        assert len(corners) == 4
        assert (corners[:, 0] >= 0.0).all() and (corners[:, 0] <= w - 1.0).all()
        assert (corners[:, 1] >= 0.0).all() and (corners[:, 1] <= h - 1.0).all()

        # Check 4 outer corners are dark table background (brightness < 120, not white triangles >= 200)
        deg = res["degraded_img"]
        corner_pixels = [deg[0, 0], deg[0, w - 1], deg[h - 1, 0], deg[h - 1, w - 1]]
        for px in corner_pixels:
            assert np.mean(px) < 120.0, f"Corner pixel is unexpectedly bright: {px}"

        # Check Homography consistency
        H = np.array(res["H_clean_to_degraded"])
        H_inv = np.array(res["H_degraded_to_clean"])
        assert np.allclose(np.dot(H, H_inv), np.eye(3), atol=1e-5)

        mapped_corners = cv2.perspectiveTransform(src_corners, H).reshape(4, 2)
        assert np.allclose(mapped_corners, corners, atol=1.0)


def test_v1_motion_blur_metadata_has_angle_deg(page):
    from synthetic.generate import _apply_recipe_v1
    rng = np.random.default_rng(42)
    res = _apply_recipe_v1(page, src_idx=0, var_idx=4, sample_rng=rng)
    assert res["recipe_name"] == "single_blur_motion"
    blur_params = res["degradation_params"]["blur"]
    assert "angle_deg" in blur_params
    assert 0.0 <= blur_params["angle_deg"] <= 180.0
    assert blur_params["radius"] > 0
    assert blur_params["ksize"] >= 3


def test_v1_items_gt_mapping():
    """Kiểm tra ánh xạ items_clean -> items_degraded qua Homography H trong Bộ 5 v1."""
    samples = sorted(DATASETS[5].glob("*/labels/*.json"))
    if not samples:
        pytest.skip("Chưa có dữ liệu Bộ 5 v1 để kiểm tra.")

    sample_path = samples[0]
    with open(sample_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    # 1. Kiểm tra cờ gt_coverage
    assert "gt_coverage" in gt
    assert gt["gt_coverage"].get("has_verified_item_gt") is True
    assert gt["gt_coverage"].get("has_item_gt") is True

    # 2. Kiểm tra danh sách items
    assert "items_clean" in gt
    assert "items_degraded" in gt
    items_deg = gt["items_degraded"]
    items_clean = gt["items_clean"]

    assert len(items_clean) >= 2, f"Số khoản thuốc sạch phải >= 2, nhận {len(items_clean)}"
    assert len(items_deg) >= 2, f"Số khoản thuốc suy thoái phải >= 2, nhận {len(items_deg)}"
    assert len(items_deg) == len(items_clean)

    # 3. Kiểm tra tính khớp nối ma trận Homography H
    H = np.array(gt["H"], dtype=np.float64)
    img_size = gt["image_size"]  # [w, h]
    img_shape = (img_size[1], img_size[0])

    for it_c, it_d in zip(items_clean, items_deg):
        assert it_c["item_id"] == it_d["item_id"]
        assert it_c["name"] == it_d["name"]
        assert "fields_gt" in it_d
        assert "line_boxes" in it_d

        # Kiểm tra transform_box dự tính khớp với item_bbox trong degraded
        expected_box = degrade.transform_box(it_c["bbox"], H, img_shape=img_shape)
        assert it_d["bbox"] == expected_box["bbox"]
        assert np.allclose(it_d["polygon"], expected_box["polygon"], atol=1e-2)


def test_v1_test_split_d3_d4_coverage():
    """Kiểm tra độ phủ 100% của Lưới D3 (15 ô) và Tập D4 (9 góc xoay) trong tập test Bộ 5 v1."""
    manifest_csv = DATASETS[5] / "manifest_v1.csv"
    if not manifest_csv.exists():
        pytest.skip("Chưa có manifest_v1.csv để kiểm tra.")

    import pandas as pd
    df = pd.read_csv(manifest_csv)
    test_df = df[df["split"] == "test"]
    assert len(test_df) == 300, f"Tập test phải có đúng 300 mẫu, nhận {len(test_df)}"

    # 1. Kiểm tra Lưới D3 (15 ô = 3 mức phối cảnh x 5 góc)
    d3_df = test_df[test_df["recipe_name"].str.startswith("d3_grid")]
    assert len(d3_df) == 150, f"Lưới D3 phải có đúng 150 mẫu, nhận {len(d3_df)}"
    d3_group = d3_df.groupby(["perspective_strength", "angle_deg"]).agg(
        count=("source_id", "count"),
        unique_sources=("source_id", "nunique")
    )
    assert len(d3_group) == 15, f"Lưới D3 phải đủ 15 ô, nhận {len(d3_group)}: {d3_group.index.tolist()}"
    assert (d3_group["unique_sources"] >= 10).all(), "Mỗi ô trong 15 ô Lưới D3 phải có >= 10 nguồn ảnh khác nhau"

    # 2. Kiểm tra Tập D4 chỉ xoay (9 góc: 0, ±2, ±5, ±10, ±15)
    d4_df = test_df[test_df["recipe_name"].str.startswith("d4_pure_rotate")]
    assert len(d4_df) == 90, f"Tập D4 chỉ xoay phải có 90 mẫu, nhận {len(d4_df)}"
    d4_group = d4_df.groupby("angle_deg").agg(
        count=("source_id", "count"),
        unique_sources=("source_id", "nunique")
    )
    assert len(d4_group) == 9, f"Tập D4 phải đủ 9 góc xoay, nhận {len(d4_group)}: {d4_group.index.tolist()}"
    expected_angles = {-15.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 15.0}
    assert set(d4_group.index.tolist()) == expected_angles, f"Góc xoay D4 không khớp: {d4_group.index.tolist()}"
    assert (d4_group["unique_sources"] >= 10).all(), "Mỗi góc xoay trong 9 góc D4 phải có >= 10 nguồn ảnh khác nhau"






# ─────────────────────────────── v2: lóa, dấu, bút, nén ───────────────────────────────

def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


@pytest.mark.parametrize("name", ["glare", "stamp", "pen", "jpeg_noise"])
def test_v2_degradations_reproducible_and_level_zero(page, name):
    fn = degrade.DEGRADATIONS[name]
    out1, _ = fn(page, 2, np.random.default_rng(7))
    out2, _ = fn(page, 2, np.random.default_rng(7))
    assert out1.shape == page.shape and out1.dtype == np.uint8
    assert np.array_equal(out1, out2), "cùng seed phải cho cùng ảnh"
    out0, gt0 = fn(page, 0, np.random.default_rng(7))
    assert np.array_equal(out0, page) and gt0["level"] == 0


def test_glare_washes_text_inside_mask_and_grows_with_level(page):
    areas = []
    for lv in (1, 2, 3):
        out, gt = degrade.glare(page, lv, np.random.default_rng(3), center=(300, 400))
        m = gt["glare_mask"] > 0
        assert m.any()
        g_in, g_out = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY), cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        assert g_out[m].mean() > g_in[m].mean() + 5, "vùng lóa phải sáng hơn ảnh gốc"
        far = np.zeros_like(m)
        far[:40, :40] = True
        assert np.array_equal(out[far], page[far]), "xa tâm lóa không được đổi"
        areas.append(m.sum())
    assert areas[0] < areas[1] < areas[2]


def test_stamp_is_red_inside_mask_and_unchanged_outside(page):
    out, gt = degrade.stamp(page, 2, np.random.default_rng(1))
    m = gt["stamp_mask"] > 0
    paper = m & (cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) > 150)   # giấy dưới dấu (không phải chữ đen)
    b, g, r = [out[..., i][paper].astype(float).mean() for i in range(3)]
    assert r > g + 20 and r > b + 20, "giấy dưới dấu phải ngả đỏ"
    changed = np.any(out != page, axis=2)
    near = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    assert not (changed & ~near).any(), "dấu không được làm đổi pixel xa mask"
    assert 0.0 <= gt["text_covered_ratio"] <= 1.0


@pytest.mark.parametrize("color, channel", [("blue", 0), ("black", None)])
def test_pen_color_and_unchanged_outside(page, color, channel):
    out, gt = degrade.pen(page, 2, np.random.default_rng(2), color=color)
    m = gt["pen_mask"] > 0
    assert gt["pen_color"] == color and m.any()
    core = cv2.erode(m.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    px = out[core].astype(float)
    if channel == 0:
        assert px[:, 0].mean() > px[:, 2].mean() + 30, "mực xanh: kênh B > R"
    else:
        assert px.mean() < 120, "mực đen phải tối"
    changed = np.any(out != page, axis=2)
    near = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert not (changed & ~near).any()


def test_pen_level1_avoids_text(page):
    out, gt = degrade.pen(page, 1, np.random.default_rng(4))
    ink = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) < 128
    m = gt["pen_mask"] > 0
    assert (ink & m).sum() / m.sum() < 0.05, "chữ ký mức 1 phải nằm ở vùng trống"


def test_jpeg_noise_levels_monotonic(page):
    psnrs = [_psnr(page, degrade.jpeg_noise(page, lv, np.random.default_rng(0))[0]) for lv in (1, 2, 3)]
    assert psnrs[0] > psnrs[1] > psnrs[2]
    _, gt = degrade.jpeg_noise(page, 3, np.random.default_rng(0))
    assert gt == {"level": 3, "jpeg_quality": 30, "noise_sigma": 10.0}


def test_v2_works_on_grayscale(page_gray):
    for name in ("glare", "stamp", "pen", "jpeg_noise"):
        out, _ = degrade.DEGRADATIONS[name](page_gray, 2, np.random.default_rng(0))
        assert out.shape == page_gray.shape and out.dtype == np.uint8


@pytest.mark.parametrize("var_idx", [10, 14])
def test_exposure_keeps_masks_and_makes_glare_visible_on_scan(page, var_idx):
    """Giấy scan 255: lóa chỉ sáng hơn mặt giấy sau khi giảm phơi sáng; mask, dấu, bút giữ nguyên (góp ý Hiếu 03/11)."""
    from synthetic.generate import _apply_recipe_v2
    scan = np.where(page == 225, 255, page).astype(np.uint8)   # giấy trắng tuyệt đối như ảnh VAIPE
    run = lambda gain: _apply_recipe_v2(scan, 0, var_idx, np.random.default_rng([42, 0, var_idx]),
                                        split_src_idx=2, exposure_gain=gain)   # k = 2 → lóa mức 3 (v10)
    base, exp = run(1.0), run(0.85)
    for name in ("shadow_mask", "glare_mask", "stamp_mask", "pen_mask"):
        assert np.array_equal(base["masks"][name], exp["masks"][name]), f"{name} phải giữ nguyên"
    assert np.array_equal(base["H_total"], exp["H_total"])
    for key in ("glare", "stamp", "pen"):
        if key in base["degradation_params"]:
            assert base["degradation_params"][key].get("center") == exp["degradation_params"][key].get("center")
    assert exp["degradation_params"]["exposure"] == {"gain": 0.85}
    assert not np.array_equal(base["degraded_img"], exp["degraded_img"])

    if var_idx == 10:   # chỉ lóa: so mặt giấy trong / ngoài mask
        m = exp["masks"]["glare_mask"] > 0
        paper = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY) == 255
        v = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
        assert np.median(v(base["degraded_img"])[m & paper]) == np.median(v(base["degraded_img"])[~m & paper]) == 255
        v_in, v_out = np.median(v(exp["degraded_img"])[m & paper]), np.median(v(exp["degraded_img"])[~m & paper])
        assert v_out == round(255 * 0.85) and v_in > v_out + 15, "lóa phải sáng hơn mặt giấy"


def _file_hashes(root):
    import hashlib
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_generate_v2_end_to_end(tmp_path, page):
    """v2 chỉ thêm v10–v14: không đổi byte nào của v1, mask đúng tọa độ, tái lập được."""
    import pandas as pd
    from synthetic import generate as synth_generate

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    for i in range(3):
        img_mock = page.copy()
        cv2.putText(img_mock, f"DOC_{i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(str(src_dir / f"doc_{i:02d}.png"), img_mock)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(CONFIG.read_text(encoding="utf-8").replace("variants_per_image: 10", "variants_per_image: 2"),
                        encoding="utf-8")

    out_dir = tmp_path / "out"
    synth_generate.generate(config_path=cfg_path, src_dir=src_dir, out_dir=out_dir, version="v1")
    before = _file_hashes(out_dir)

    res = synth_generate.generate(config_path=cfg_path, src_dir=src_dir, out_dir=out_dir, version="v2")
    assert res["new_samples"] == 3 * 5
    after = _file_hashes(out_dir)
    assert all(after[k] == v for k, v in before.items()), "v2 không được sửa file nào của v1"

    m2 = pd.read_csv(out_dir / "manifest_v2.csv")
    assert len(m2) == 3 * 2 + 3 * 5
    assert set(m2[m2.version_added == "v2"].variant_index) == {10, 11, 12, 13, 14}
    assert (m2[m2.version_added == "v1"].glare_level == 0).all()

    for lp in sorted(out_dir.glob("*/labels/*_v1[0-4].json")):
        gt = json.loads(lp.read_text(encoding="utf-8"))
        sp, sid = lp.parent.parent, gt["sample_id"]
        assert set(gt["degradation_levels"]) == {"perspective", "rotate", "blur", "shadow",
                                                 "glare", "stamp", "pen", "jpeg_noise"}
        page_mask = cv2.imread(str(sp / "page_mask" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        for name in ("glare_mask", "stamp_mask", "pen_mask"):
            mk = cv2.imread(str(sp / name / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
            assert mk.shape == page.shape[:2] and set(np.unique(mk)) <= {0, 255}
            assert gt["masks"][f"has_{name.split('_')[0]}"] == bool(mk.any())
            assert not (mk > 0)[page_mask == 0].any(), f"{name} nằm ngoài tờ giấy"
        expected = {"single_glare": {"glare"}, "single_stamp": {"stamp"}, "single_pen": {"pen"},
                    "single_jpeg_noise": {"jpeg_noise"}}.get(gt["recipe_name"])
        if expected:
            assert set(gt["applied_degradations"]) == expected
        else:
            assert {"stamp", "pen", "perspective", "glare", "jpeg_noise"} <= set(gt["applied_degradations"])
        clean = cv2.imread(str(sp / "clean" / f"{sid}.png"))
        src_name = sid.rsplit("_v", 1)[0] + ".png"
        assert np.array_equal(clean, cv2.imread(str(src_dir / src_name))), "ảnh sạch không được có dấu / bút"
        exposed = sp / "clean_exposed" / f"{sid}.png"
        if gt["variant_index"] in (10, 14):
            lo, hi = degrade.EXPOSURE_GAIN_RANGE
            assert lo <= gt["exposure_gain"] <= hi and gt["degradation_params"]["exposure"]["gain"] == gt["exposure_gain"]
            assert np.array_equal(cv2.imread(str(exposed)), degrade.exposure(clean, gt["exposure_gain"]))
        else:
            assert gt["exposure_gain"] == 1.0 and gt["clean_exposed"] is None and not exposed.exists()

    # tái lập: sinh lại v2 vào thư mục khác → ảnh hỏng giống hệt
    out_dir_b = tmp_path / "out_b"
    synth_generate.generate(config_path=cfg_path, src_dir=src_dir, out_dir=out_dir_b, version="v1")
    synth_generate.generate(config_path=cfg_path, src_dir=src_dir, out_dir=out_dir_b, version="v2")
    for p in out_dir.glob("*/images/*_v1[0-4].png"):
        assert np.array_equal(cv2.imread(str(p)), cv2.imread(str(out_dir_b / p.relative_to(out_dir))))


def test_generate_v2_requires_v1(tmp_path, page):
    from synthetic import generate as synth_generate
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    cv2.imwrite(str(src_dir / "doc_00.png"), page)
    with pytest.raises(FileNotFoundError):
        synth_generate.generate(src_dir=src_dir, out_dir=tmp_path / "out", version="v2")


def test_v2_dataset_level_coverage():
    """Bộ 5 v2 thật: mỗi mức của từng loại hỏng mới đều có ở cả train, val, test; v1 vẫn đủ 1.000 mẫu."""
    manifest_csv = DATASETS[5] / "manifest_v2.csv"
    if not manifest_csv.exists():
        pytest.skip("Chưa có manifest_v2.csv để kiểm tra.")
    import pandas as pd
    df = pd.read_csv(manifest_csv)
    assert (df.version_added == "v1").sum() == 1000 and (df.version_added == "v2").sum() == 500
    for split in ("train", "val", "test"):
        sub = df[df.split == split]
        for col, levels in (("glare_level", {1, 2, 3}), ("stamp_level", {1, 2, 3}), ("pen_level", {1, 2}),
                            ("jpeg_noise_level", {1, 2, 3})):
            assert levels <= set(sub[col]), f"{split} thiếu mức {col}"
        assert (sub.glare_level == 0).any() and (sub.glare_level > 0).any(), "cần cả mẫu âm và dương cho lóa"
    lo, hi = degrade.EXPOSURE_GAIN_RANGE
    glare = df[df.glare_level > 0]
    assert glare.exposure_gain.between(lo, hi).all() and glare.rel_clean_exposed_path.notna().all()
    assert (df[df.glare_level == 0].exposure_gain == 1.0).all()
