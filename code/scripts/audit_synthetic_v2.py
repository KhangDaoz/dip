"""Kiểm tra Bộ 5 v2 sau khi sinh (python -m synthetic.generate --version v2) và ghi checksum phần mới.

Kiểm:
1. 6.000 file v1 vẫn khớp checksums_v1.sha256 (v2 không được sửa v1).
2. Đủ 500 mẫu v10–v14, mỗi mẫu 9 file (v10, v14: 10 file, thêm clean_exposed); mask nhị phân 0/255, cùng kích thước ảnh, nằm trong tờ giấy, cờ has_* khớp mask.
3. Công thức đúng loại hỏng; GT khoản thuốc items_degraded = biến đổi items_clean qua H.
4. Mỗi mức của lóa / dấu / bút / nén có ở cả train, val, test; có mẫu âm và dương cho lóa.
5. v10, v14: exposure_gain trong EXPOSURE_GAIN_RANGE, clean_exposed = clean × gain; v11–v13 không đổi phơi sáng.
   Ghi thêm độ sáng giấy trong / ngoài vùng lóa (single_glare): trong mask giấy phải lên ít nhất nửa đường tới 255.
Ghi: data/5_synthetic/checksums_v2.sha256 (file mới của v2), results/bo5_v2_audit/{audit.json, stats.csv, sheet_*.png}.

Chạy (từ thư mục code/): python scripts/audit_synthetic_v2.py
"""
import hashlib
import json

import cv2
import numpy as np
import pandas as pd

from common.experiment import results_dir
from common.io import read_image, write_image
from common.paths import DATASETS
from synthetic import degrade
from synthetic.generate import GLARE_VARIANTS, RECIPE_NAMES_V2, V2_MASKS, transform_items

ROOT = DATASETS[5]
FILE_DIRS = ("images", "labels", "clean", "clean_binary", "shadow_mask", "page_mask") + V2_MASKS
EXPECTED = {"single_glare": {"glare"}, "single_stamp": {"stamp"}, "single_pen": {"pen"},
            "single_jpeg_noise": {"jpeg_noise"},
            "combo_full_v2": {"stamp", "pen", "perspective", "rotate", "shadow", "glare", "blur", "jpeg_noise"}}


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_v1() -> int:
    bad = 0
    for line in (ROOT / "checksums_v1.sha256").read_text(encoding="utf-8").splitlines():
        digest, rel = line.strip().split(maxsplit=1)
        if sha256(ROOT / rel) != digest:
            bad += 1
    return bad


def psnr(a, b) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    out = results_dir("bo5_v2_audit")
    errors = []
    v1_bad = check_v1()
    if v1_bad:
        errors.append(f"{v1_bad} file v1 không còn khớp checksums_v1.sha256")

    m = pd.read_csv(ROOT / "manifest_v2.csv")
    v2 = m[m.version_added == "v2"]
    stats, new_files = [], []
    for row in v2.itertuples():
        sp, sid = ROOT / row.split, row.sample_id
        dirs = FILE_DIRS + (("clean_exposed",) if row.variant_index in GLARE_VARIANTS else ())
        paths = {d: sp / d / (f"{sid}.json" if d == "labels" else f"{sid}.png") for d in dirs}
        missing = [d for d, p in paths.items() if not p.exists()]
        if missing:
            errors.append(f"{sid}: thiếu {missing}")
            continue
        new_files.extend(paths.values())
        gt = json.loads(paths["labels"].read_text(encoding="utf-8"))
        w, h = gt["image_size"]
        page = read_image(paths["page_mask"], gray=True)
        for name in V2_MASKS:
            mk = read_image(paths[name], gray=True)
            if mk.shape != (h, w) or not set(np.unique(mk)) <= {0, 255}:
                errors.append(f"{sid}: {name} sai kích thước / không nhị phân")
            if (mk > 0)[page == 0].any():
                errors.append(f"{sid}: {name} nằm ngoài tờ giấy")
            if gt["masks"][f"has_{name.split('_')[0]}"] != bool(mk.any()):
                errors.append(f"{sid}: cờ has_{name.split('_')[0]} không khớp mask")
        if set(gt["applied_degradations"]) != EXPECTED[gt["recipe_name"]]:
            errors.append(f"{sid}: loại hỏng {gt['applied_degradations']} sai công thức {gt['recipe_name']}")
        if gt["recipe_name"] != RECIPE_NAMES_V2[gt["variant_index"]]:
            errors.append(f"{sid}: variant_index không khớp tên công thức")
        if transform_items(gt["items_clean"], np.array(gt["H"]), (h, w)) != gt["items_degraded"]:
            errors.append(f"{sid}: items_degraded không khớp items_clean qua H")
        img, clean = read_image(paths["images"]), read_image(paths["clean"])
        gain, lo_hi = gt["exposure_gain"], degrade.EXPOSURE_GAIN_RANGE
        paper_in = paper_out = None
        if row.variant_index in GLARE_VARIANTS:
            if not lo_hi[0] <= gain <= lo_hi[1] or row.exposure_gain != gain:
                errors.append(f"{sid}: exposure_gain {gain} ngoài khoảng {lo_hi} / lệch manifest")
            ref = read_image(paths["clean_exposed"])
            if not np.array_equal(ref, degrade.exposure(clean, gain)):
                errors.append(f"{sid}: clean_exposed khác clean × exposure_gain")
            if gt["recipe_name"] == "single_glare":
                mk = read_image(paths["glare_mask"], gray=True) > 0
                paper = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) >= 250
                v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
                paper_in, paper_out = float(np.median(v[mk & paper])), float(np.median(v[~mk & paper]))
                if paper_in < paper_out + 0.5 * (255 - paper_out) - 1:   # mask = B ≥ 0,5: giấy đi ít nhất nửa đường lên 255 (−1 làm tròn)
                    errors.append(f"{sid}: lóa không sáng hơn mặt giấy ({paper_in} vs {paper_out})")
            clean = ref   # PSNR so với ảnh sạch cùng phơi sáng
        elif gain != 1.0:
            errors.append(f"{sid}: v11–v13 không được đổi phơi sáng")
        stats.append({"sample_id": sid, "split": row.split, "recipe": gt["recipe_name"], **gt["degradation_levels"],
                      "psnr_vs_clean": round(psnr(img, clean), 2) if gt["recipe_name"] != "combo_full_v2" else None,
                      "exposure_gain": gain, "paper_v_in_glare": paper_in, "paper_v_out_glare": paper_out,
                      "glare_area_ratio": row.glare_area_ratio, "stamp_text_covered_ratio": row.stamp_text_covered_ratio,
                      "n_items": gt["n_items"]})

    coverage = {}
    for split in ("train", "val", "test"):
        sub = m[m.split == split]
        coverage[split] = {col: sorted(int(x) for x in sub[col].unique()) for col in
                           ("glare_level", "stamp_level", "pen_level", "jpeg_noise_level")}
        for col, levels in (("glare_level", {1, 2, 3}), ("stamp_level", {1, 2, 3}), ("pen_level", {1, 2}),
                            ("jpeg_noise_level", {1, 2, 3})):
            if not levels <= set(sub[col]):
                errors.append(f"{split}: thiếu mức {col}")

    lines = [f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}" for p in sorted(new_files)]
    lines += [f"{sha256(ROOT / n)}  {n}" for n in ("manifest_v2.csv", "manifest_v2.json")]
    # newline="\n": trên Windows write_text mặc định ghi CRLF → sha256sum -c (Linux / Colab) báo lỗi mọi file
    (ROOT / "checksums_v2.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    df = pd.DataFrame(stats)
    df.to_csv(out / "stats.csv", index=False, encoding="utf-8")
    summary = {
        "v1_checksum_mismatch": v1_bad,
        "v2_samples": int(len(v2)), "v2_files_checksummed": len(lines),
        "manifest_v2_rows": int(len(m)), "split_counts_v2": v2.split.value_counts().to_dict(),
        "level_coverage": coverage,
        "glare_negative_ratio_all": round(float((m.glare_level == 0).mean()), 3),
        "mean_by_recipe_level": {
            "glare_area_ratio": df[df.recipe == "single_glare"].groupby("glare").glare_area_ratio.mean().round(4).to_dict(),
            "paper_v_in_glare": df[df.recipe == "single_glare"].groupby("glare").paper_v_in_glare.mean().round(1).to_dict(),
            "paper_v_out_glare": df[df.recipe == "single_glare"].groupby("glare").paper_v_out_glare.mean().round(1).to_dict(),
            "psnr_glare_vs_clean_exposed": df[df.recipe == "single_glare"].groupby("glare").psnr_vs_clean.mean().round(2).to_dict(),
            "stamp_text_covered_ratio": df[df.recipe == "single_stamp"].groupby("stamp").stamp_text_covered_ratio.mean().round(4).to_dict(),
            "psnr_jpeg_noise": df[df.recipe == "single_jpeg_noise"].groupby("jpeg_noise").psnr_vs_clean.mean().round(2).to_dict(),
        },
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    (out / "audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ảnh lưới để duyệt bằng mắt: mỗi công thức 6 mẫu tập test, viền mask + box khoản thuốc
    colors = {"glare_mask": (0, 220, 255), "stamp_mask": (255, 0, 255), "pen_mask": (255, 128, 0)}
    for recipe in RECIPE_NAMES_V2.values():
        tiles = []
        for row in v2[(v2.recipe_name == recipe) & (v2.split == "test")].head(6).itertuples():
            sp, sid = ROOT / row.split, row.sample_id
            gt = json.loads((sp / "labels" / f"{sid}.json").read_text(encoding="utf-8"))
            im = read_image(sp / "images" / f"{sid}.png").copy()
            for name, c in colors.items():
                mk = (read_image(sp / name / f"{sid}.png", gray=True) > 0).astype(np.uint8)
                cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cnts, -1, c, 3)
            for it in gt["items_degraded"]:
                cv2.polylines(im, [np.round(np.array(it["polygon"])).astype(np.int32)], True, (0, 200, 0), 3)
            s = 400 / im.shape[1]
            im = cv2.resize(im, None, fx=s, fy=s)
            tile = np.full((max(im.shape[0], 600) + 30, 400, 3), 255, np.uint8)
            tile[30:30 + im.shape[0]] = im[:tile.shape[0] - 30]
            label = " ".join(f"{k[:5]}={v}" for k, v in gt["degradation_levels"].items() if v)
            cv2.putText(tile, f"{sid[-12:]} {label}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
            tiles.append(tile)
        if tiles:
            hmax = max(t.shape[0] for t in tiles)
            tiles = [np.vstack([t, np.full((hmax - t.shape[0], 400, 3), 255, np.uint8)]) for t in tiles]
            write_image(out / f"sheet_{recipe}.png", np.hstack(tiles))

    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=2, ensure_ascii=False))
    print(f"Lỗi: {len(errors)}", *errors[:20], sep="\n  ")
    print(f"Kết quả: {out}")


if __name__ == "__main__":
    main()
