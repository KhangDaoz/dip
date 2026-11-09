"""K0 — Ground truth dòng / cột cho bộ 2 (57 ảnh gốc). Người làm: Khang. Kế hoạch: KeHoach.md mục 7.2, 7.4.

Nhãn YOLO của bộ 2: lớp + tâm / kích thước chuẩn hóa. Bỏ lớp `Report Type` (không thuộc thân bảng).
Dòng GT: hai cụm từ cùng dòng nếu chồng nhau theo chiều dọc ≥ 50 % chiều cao cụm thấp hơn; nối bằng union-find
(độc lập với cả hai thuật toán gom dòng, tránh lỗi của pilot cũ — KeHoach.md mục 7.4).
Cột GT: vai trò cụm từ (parameter, value, unit, reference value).
Kết quả: results/k0/gt.json (ảnh → words [x, y, w, h], role, row_id) và results/k0/review/*.jpg để SOÁT BẰNG MẮT.

Chạy (từ code/): python experiments/k0_gt.py
"""
import cv2
import numpy as np

from common.experiment import results_dir, save_config
from common.io import read_image, save_json, write_image
from common.paths import DATASETS
from common.seed import set_seed

EXP_ID = "k0"
NAMES = ["Report Type", "parameter", "reference value", "unit", "value"]
ROLE = {"parameter": "parameter", "value": "result", "unit": "unit", "reference value": "range"}
CONFIG = {"seed": 42, "row_overlap": 0.5, "splits": ["train", "valid", "test"]}
COLORS = {"parameter": (200, 80, 0), "result": (0, 0, 220), "unit": (0, 150, 0), "range": (160, 0, 160)}


def load_words(img_path, lab_path):
    img = read_image(img_path)
    H, W = img.shape[:2]
    words = []
    for line in open(lab_path, encoding="utf-8"):
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        if NAMES[c] == "Report Type":
            continue
        words.append({"bbox": [int(round((cx - w / 2) * W)), int(round((cy - h / 2) * H)),
                               int(round(w * W)), int(round(h * H))], "role": ROLE[NAMES[c]]})
    return img, words


def gt_rows(words, ratio):
    n = len(words)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        yi, hi = words[i]["bbox"][1], words[i]["bbox"][3]
        for j in range(i + 1, n):
            yj, hj = words[j]["bbox"][1], words[j]["bbox"][3]
            ov = min(yi + hi, yj + hj) - max(yi, yj)
            if ov >= ratio * min(hi, hj):
                parent[find(i)] = find(j)
    roots = sorted({find(i) for i in range(n)}, key=lambda r: min(words[i]["bbox"][1] for i in range(n) if find(i) == r))
    rid = {r: k for k, r in enumerate(roots)}
    return [rid[find(i)] for i in range(n)]


def main():
    set_seed(CONFIG["seed"])
    out = results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    gt = {}
    for split in CONFIG["splits"]:
        d = DATASETS[2] / split
        for img_path in sorted((d / "images").glob("*")):
            lab = d / "labels" / f"{img_path.stem}.txt"
            if not lab.exists():
                continue
            img, words = load_words(img_path, lab)
            if not words:
                continue
            rows = gt_rows(words, CONFIG["row_overlap"])
            for w, r in zip(words, rows):
                w["row"] = r
            key = f"{split}/{img_path.name}"
            gt[key] = {"split": split, "image": str(img_path.relative_to(DATASETS[2])), "words": words,
                       "n_rows": max(rows) + 1}
            vis = img.copy()
            for w in words:
                x, y, ww, hh = w["bbox"]
                cv2.rectangle(vis, (x, y), (x + ww, y + hh), COLORS[w["role"]], 1)
                cv2.putText(vis, str(w["row"]), (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[w["role"]], 1)
            write_image(out / "review" / f"{split}_{img_path.stem[:40]}.jpg", vis, jpeg_quality=85)
    save_json(out / "gt.json", gt)
    print(len(gt), "ảnh ·", sum(len(g["words"]) for g in gt.values()), "cụm từ ·",
          sum(g["n_rows"] for g in gt.values()), "dòng")


if __name__ == "__main__":
    main()
