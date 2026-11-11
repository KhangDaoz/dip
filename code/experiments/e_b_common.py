"""Phần dùng chung của E-B1, E-B2 (khối B, Trường): đọc dữ liệu VAIPE, cache dự đoán YOLO, chống chạy test nhiều lần."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR / "src"))

import cv2  # noqa: E402

VAIPE_DIR = REPO_DIR / "data" / "1_vaipe"
YOLO_YAML = REPO_DIR / "data" / "1_vaipe_yolo" / "vaipe.yaml"    # sinh bởi tools/to_yolo.py, có khai báo test
RESULTS = REPO_DIR / "results"
SPLITS = {"valid": 176, "test": 176}


def out_dir(exp_id: str) -> Path:
    d = RESULTS / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR, text=True).strip()
    except Exception:
        return "unknown"


def save_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_csv(path: Path, rows: list[dict]) -> Path:
    """CSV UTF-8 có BOM để mở bằng Excel không lỗi dấu (giống common.experiment.save_table)."""
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def load_split(split: str) -> list[dict]:
    """[{id, path, boxes [[x0,y0,x1,y1]], labels [str], texts [str]}] theo thứ tự tên file."""
    items = []
    for lab in sorted((VAIPE_DIR / split / "labels").glob("*.json")):
        img = VAIPE_DIR / split / "images" / f"{lab.stem}.png"
        ann = json.loads(lab.read_text(encoding="utf-8"))
        items.append({"id": lab.stem, "path": img,
                      "boxes": [a["box"] for a in ann], "labels": [a["label"] for a in ann],
                      "texts": [a.get("text", "") for a in ann]})
    if split in SPLITS and len(items) != SPLITS[split]:
        raise RuntimeError(f"{split}: {len(items)} ảnh, cần {SPLITS[split]}")
    return items


def read_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def yolo_predictions(split: str, items: list[dict], weights=None, min_conf: float = 0.01) -> dict:
    """Chạy YOLO một lần ở conf thấp, cache ra results/e_b1/yolo_pred_<split>.json; lọc theo conf về sau.
    Cache bị bỏ nếu trọng số đổi (so mtime + kích thước)."""
    from detect.yolo import DEFAULT_WEIGHTS, YoloDetector
    weights = Path(weights or DEFAULT_WEIGHTS)
    key = f"{weights.name}:{weights.stat().st_size}:{int(weights.stat().st_mtime)}:{min_conf}"
    cache = out_dir("e_b1") / f"yolo_pred_{split}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("key") == key:
            return data
    det = YoloDetector(weights, conf=min_conf)
    preds, times = {}, []
    for k, it in enumerate(items):
        img = read_bgr(it["path"])
        t = time.perf_counter()
        boxes, cls, score = det.predict_raw(img, conf=min_conf)
        times.append(time.perf_counter() - t)
        preds[it["id"]] = {"boxes": boxes.round(2).tolist(), "labels": [det.names[int(c)] for c in cls],
                           "scores": score.round(4).tolist()}
        if (k + 1) % 25 == 0:
            print(f"  yolo {split}: {k + 1}/{len(items)}", flush=True)
    times = times[1:] if len(times) > 1 else times      # bỏ ảnh đầu (khởi động)
    data = {"key": key, "device": det.device, "imgsz": det.imgsz, "iou": det.iou,
            "sec_per_image": sum(times) / len(times), "preds": preds}
    save_json(cache, data)
    return data


def filter_conf(pred: dict, conf: float):
    keep = [i for i, s in enumerate(pred["scores"]) if s >= conf]
    return [pred["boxes"][i] for i in keep], [pred["labels"][i] for i in keep], [pred["scores"][i] for i in keep]


def guard_test(exp_id: str, reason: str | None):
    """Kỷ luật đánh giá (plan.md mục 6): test chạy MỘT lần. Chạy lại phải ghi lý do, lý do được lưu vào test_runs.log."""
    log = out_dir(exp_id) / "test_runs.log"
    if log.exists() and not reason:
        sys.exit(f"{exp_id} đã chạy trên test (xem {log}). Chạy lại phải thêm --rerun-reason \"lý do\".")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\tcommit {commit_hash()}\t{reason or 'lần đầu'}\n")
