"""Sinh bộ 5 → data/5_synthetic/. Chạy: python -m synthetic.generate [--out thư_mục] [--n-sources N]

Bộ 5 v2 chính thức (1.500 mẫu) đã có trên Drive — KHÔNG cần sinh lại để làm thí nghiệm. Script này tái lập ý tưởng:
mỗi ảnh gốc VAIPE sạch sinh nhiều biến thể, mỗi biến thể ghép một chuỗi phép hỏng (degrade.py); chia 60 / 10 / 30
THEO ẢNH GỐC (mọi biến thể của một ảnh gốc nằm cùng một tập, C8 · S41).
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from common.io import read_image, save_json, write_image
from common.paths import DATASETS
from . import degrade

RECIPES = [
    ["perspective", "rotate"], ["gaussian_blur"], ["motion_blur"], ["shadow"], ["glare"],
    ["stamp"], ["pen_blue"], ["jpeg"], ["gaussian_noise"], ["perspective", "shadow", "jpeg"],
]


def _apply(name, img, cfg, rng):
    d = cfg["degradations"]
    if name == "perspective":
        return degrade.perspective(img, float(rng.choice(d["perspective"]["levels"])), rng)
    if name == "rotate":
        return degrade.rotate(img, float(rng.choice(d["rotate"]["levels"])))
    if name == "gaussian_blur":
        return degrade.gaussian_blur(img, int(rng.choice(d["gaussian_blur"]["levels"])))
    if name == "motion_blur":
        return degrade.motion_blur(img, int(rng.choice(d["motion_blur"]["levels"])), rng)
    if name == "shadow":
        return degrade.shadow(img, int(rng.choice(d["shadow"]["levels"])), rng)
    if name == "glare":
        return degrade.glare(img, int(rng.choice(d["glare"]["levels"])), rng, tuple(d["glare"]["exposure_range"]))
    if name in ("stamp", "pen_blue", "pen_black"):
        return degrade.stamp_pen(img, int(rng.choice(d["stamp_pen"]["levels"])), rng, name)
    if name == "jpeg":
        return degrade.jpeg(img, int(rng.choice(d["jpeg"]["levels"])))
    if name == "gaussian_noise":
        return degrade.gaussian_noise(img, int(rng.choice(d["gaussian_noise"]["levels"])), rng)
    raise ValueError(name)


def generate(config_path: Path = Path(__file__).with_name("config.yaml"), out_dir: Path = None,
             n_sources: int = 100) -> None:
    """Đọc config, chia ảnh gốc 60 / 10 / 30 (biến thể cùng ảnh gốc chung một tập), ghi ảnh hỏng + gt.json + mask."""
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    rng = np.random.default_rng(cfg["seed"])
    out_dir = Path(out_dir or DATASETS[5].with_name("5_synthetic_regen"))
    srcs = sorted((DATASETS[1] / "train" / "images").glob("*.png"))
    srcs = [srcs[i] for i in rng.permutation(len(srcs))[:n_sources]]
    n_tr, n_va = int(cfg["split"]["train"] * len(srcs)), int(cfg["split"]["val"] * len(srcs))
    for si, src in enumerate(srcs):
        split = "train" if si < n_tr else "val" if si < n_tr + n_va else "test"
        clean = read_image(src)
        write_image(out_dir / split / "clean" / f"{src.stem}.png", clean)
        write_image(out_dir / split / "clean_binary" / f"{src.stem}.png", degrade.clean_binary(clean))
        for vi, recipe in enumerate(RECIPES):
            img, gt = clean.copy(), {"source": src.stem, "split": split, "recipe": recipe, "steps": []}
            for name in recipe:
                img, g = _apply(name, img, cfg, rng)
                for k, v in list(g.items()):
                    if isinstance(v, np.ndarray):
                        write_image(out_dir / split / k / f"{src.stem}_v{vi:02d}.png", v)
                        g[k] = f"{k}/{src.stem}_v{vi:02d}.png"
                gt["steps"].append({"name": name, **g})
            write_image(out_dir / split / "images" / f"{src.stem}_v{vi:02d}.png", img)
            save_json(out_dir / split / "labels" / f"{src.stem}_v{vi:02d}.json", gt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-sources", type=int, default=100)
    a = ap.parse_args()
    generate(out_dir=a.out, n_sources=a.n_sources)
