"""S0 — Sinh ground truth khoản thuốc + vùng thuốc cho VAIPE (valid, test). Người làm: Thắng. Kế hoạch: KeHoach.md mục 3.3, 8.3.

Luật gom: mỗi box `drugname` mở một khoản. Box `quantity` / `usage` CÙNG HÀNG với một `drugname` (chồng nhau theo
chiều dọc ≥ 50 % chiều cao box) thuộc khoản đó; còn lại thuộc `drugname` gần nhất nằm phía trên. (Sắp theo tâm y là
sai: "SL" của thuốc 2 có tâm y hơi cao hơn tên thuốc 2 nên bị gán nhầm sang khoản 1.)
Khoản = DẢI NGANG: x trải hết bề ngang vùng thuốc, y từ đỉnh dòng tên thuốc tới đáy box thấp nhất của khoản — vì nhãn
VAIPE đôi khi thiếu box "SL" dù chữ có trên ảnh, nếu lấy khung bao box thì khoản bị hẹp giả.
Vùng thuốc = khung bao mọi box của 3 nhãn. Box VAIPE là [x_min, y_min, x_max, y_max] → đổi [x, y, w, h].

Kết quả:
  results/s0/gt_items_<split>.json        GT cho thí nghiệm S1, S3, S4 (trường "reviewed" = đã soát bằng mắt hay chưa)
  results/s0/review/<split>/*.jpg         ảnh vẽ khung để SOÁT BẰNG MẮT (Thắng soát ≥ 60 ảnh test)
  results/s0/review_<split>.csv           điền cột ok = 1 / 0 sau khi soát; chạy lại script để cập nhật "reviewed"

Chạy (từ code/): python experiments/s0_gt_vaipe.py
"""
import csv
import json

import cv2
import pandas as pd

from common.experiment import results_dir, save_config, save_table
from common.io import load_json, read_image, save_json, write_image
from common.paths import DATASETS
from common.seed import set_seed

EXP_ID = "s0"
CONFIG = {"seed": 42, "splits": ["valid", "test"], "labels": ["drugname", "quantity", "usage"], "review_width": 900}


def to_xywh(b):
    return [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]


def union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def items_from_labels(lab: list) -> dict:
    boxes = [{"label": e["label"], "bbox": to_xywh(e["box"]), "text": e["text"]}
             for e in lab if e["label"] in CONFIG["labels"]]
    drugs = sorted([b for b in boxes if b["label"] == "drugname"], key=lambda b: b["bbox"][1])
    if not drugs:
        return {"items": [], "region": None, "all_boxes": []}
    groups = [[d] for d in drugs]
    for b in boxes:
        if b["label"] == "drugname":
            continue
        x, y, w, h = b["bbox"]
        overlap = [max(0, min(y + h, d["bbox"][1] + d["bbox"][3]) - max(y, d["bbox"][1])) for d in drugs]
        j = max(range(len(drugs)), key=lambda k: overlap[k])
        if overlap[j] < 0.5 * h:   # không cùng hàng → tên thuốc gần nhất phía trên
            above = [k for k, d in enumerate(drugs) if d["bbox"][1] <= y]
            if not above:
                continue
            j = above[-1]
        groups[j].append(b)
    region = union([b["bbox"] for g in groups for b in g])
    out = []
    for g in groups:
        g.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        y0 = min(b["bbox"][1] for b in g)
        y1 = max(b["bbox"][1] + b["bbox"][3] for b in g)
        out.append({"bbox": [region[0], y0, region[2], y1 - y0], "bbox_labels": union([b["bbox"] for b in g]),
                    "name": g[0]["text"], "text": " ".join(b["text"] for b in g), "lines": g})
    lines_all = [{"label": e["label"], "bbox": to_xywh(e["box"]), "text": e["text"]} for e in lab]
    return {"items": out, "region": region, "all_boxes": lines_all}


def review_image(img, gt, width):
    vis = img.copy()
    x, y, w, h = gt["region"]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for i, it in enumerate(gt["items"]):
        x, y, w, h = it["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(vis, str(i + 1), (x - 25 if x > 30 else x + w + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
    ry0, ry1 = max(gt["region"][1] - 80, 0), min(gt["region"][1] + gt["region"][3] + 80, img.shape[0])
    vis = vis[ry0:ry1]
    return cv2.resize(vis, (width, int(vis.shape[0] * width / vis.shape[1])))


def main():
    set_seed(CONFIG["seed"])
    out = results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    rows = []
    for split in CONFIG["splits"]:
        d = DATASETS[1] / split
        review_csv = out / f"review_{split}.csv"
        reviewed = {}
        if review_csv.exists():   # giữ kết quả soát tay của lần trước
            for r in csv.DictReader(open(review_csv, encoding="utf-8-sig")):
                if r.get("ok", "").strip() in ("0", "1"):
                    reviewed[r["image_id"]] = r["ok"].strip() == "1"
        gts = {}
        for lab_path in sorted((d / "labels").glob("*.json")):
            gt = items_from_labels(load_json(lab_path))
            img_id = lab_path.stem
            if not gt["items"]:
                continue
            gt["reviewed"] = img_id in reviewed
            gt["review_ok"] = reviewed.get(img_id)
            gts[img_id] = gt
            img = read_image(d / "images" / f"{img_id}.png")
            write_image(out / "review" / split / f"{img_id}.jpg", review_image(img, gt, CONFIG["review_width"]),
                        jpeg_quality=80)
            rows.append({"split": split, "image_id": img_id, "n_items": len(gt["items"]),
                         "width": img.shape[1], "reviewed": gt["reviewed"], "ok": gt["review_ok"]})
        save_json(out / f"gt_items_{split}.json", gts)
        if not review_csv.exists():
            with open(review_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "n_items", "ok", "note"])
                for k, g in gts.items():
                    w.writerow([k, len(g["items"]), "", ""])
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID, "summary")
    print(df.groupby("split").agg(images=("image_id", "count"), items=("n_items", "sum"),
                                  reviewed=("reviewed", "sum")))


if __name__ == "__main__":
    main()
