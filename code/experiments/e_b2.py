"""E-B2 — YOLO phân lớp đúng không? (khối B, Trường; plan.md mục 5.2, Truong_KhoiB.md B10)

Trên YOLOv8s 1024, conf chọn trên valid (results/e_b1/chosen.json):
  * P/R/F1 từng lớp @IoU 0,5 (ghép có phân lớp) + macro F1                 -> table_<split>.csv
  * Ma trận nhầm lẫn theo vị trí (ghép không phân lớp, có hàng/cột nền)     -> confusion_<split>.csv/.png
  * mAP50, mAP50-95 theo ultralytics model.val (không dùng để chọn mô hình)  -> config_<split>.json
  * 3 ảnh nhiều lỗi nhất: GT xanh lá, dự đoán đúng xanh dương, sai đỏ        -> errors_<split>/*.png

Chạy từ code/:  python experiments/e_b2.py --split valid    (chạy thử)
                python experiments/e_b2.py --split test     (SỐ LIỆU CUỐI, một lần)
"""
from __future__ import annotations

import argparse
from collections import Counter

import cv2
import numpy as np

from e_b_common import (YOLO_YAML, commit_hash, filter_conf, guard_test, load_split, out_dir, read_bgr, save_csv,
                        save_json, yolo_predictions)
from detect.textbox import CLASSES
from detect.yolo import DEFAULT_WEIGHTS
from metrics.detection import DetectionCounter, confusion_counts, match_greedy

EXP = "e_b2"


def map_ultralytics(split: str) -> dict:
    """mAP theo ultralytics; split 'val' của yaml là valid, 'test' là test."""
    from ultralytics import YOLO
    if not YOLO_YAML.exists():
        return {"error": f"thiếu {YOLO_YAML} — chạy tools/to_yolo.py"}
    m = YOLO(str(DEFAULT_WEIGHTS))
    met = m.val(data=str(YOLO_YAML), split="val" if split == "valid" else "test", imgsz=1024, batch=4,
                device="cpu", plots=False, verbose=False, project=str(out_dir(EXP)), name=f"ultra_{split}",
                exist_ok=True)
    return {"mAP50": float(met.box.map50), "mAP50_95": float(met.box.map),
            "per_class_mAP50": {CLASSES[i]: float(v) for i, v in zip(met.box.ap_class_index, met.box.ap50)}}


def plot_confusion(cm: np.ndarray, labels: list[str], path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    # chuẩn hóa theo hàng (nhãn GT) để so các lớp có số lượng khác nhau; ô ghi số đếm
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn chuẩn")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_errors(it: dict, boxes, labels, path):
    img = read_bgr(it["path"])
    pairs = match_greedy(it["boxes"], boxes, 0.5, it["labels"], labels)
    mg, mp = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
    for i, b in enumerate(it["boxes"]):                       # GT bị bỏ sót: xanh lá đậm
        if i not in mg:
            cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 160, 0), 2)
            cv2.putText(img, "GT:" + it["labels"][i], (int(b[0]), int(b[3]) + 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 160, 0), 1, cv2.LINE_AA)
    for j, b in enumerate(boxes):                             # dự đoán: đúng xanh dương mảnh, sai đỏ đậm
        ok = j in mp
        color = (200, 120, 0) if ok else (0, 0, 220)
        cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), color, 1 if ok else 2)
        if not ok:
            cv2.putText(img, labels[j], (int(b[0]), int(b[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                        cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["valid", "test"], required=True)
    ap.add_argument("--rerun-reason")
    ap.add_argument("--no-map", action="store_true", help="bỏ bước model.val (chậm trên CPU)")
    a = ap.parse_args(argv)

    import json
    chosen_p = out_dir("e_b1") / "chosen.json"
    if not chosen_p.exists() or "conf" not in json.loads(chosen_p.read_text(encoding="utf-8")):
        raise SystemExit("Chưa chọn conf trên valid: chạy experiments/e_b1.py --sweep-conf trước.")
    conf = json.loads(chosen_p.read_text(encoding="utf-8"))["conf"]
    if a.split == "test":
        guard_test(EXP, a.rerun_reason)

    items = load_split(a.split)
    data = yolo_predictions(a.split, items)
    counter = DetectionCounter(0.5, class_aware=True)
    conf_c, errors = Counter(), []
    for it in items:
        boxes, labels, _ = filter_conf(data["preds"][it["id"]], conf)
        pairs = counter.add(it["boxes"], boxes, it["labels"], labels)
        conf_c.update(confusion_counts(it["boxes"], it["labels"], boxes, labels))
        errors.append((len(it["boxes"]) + len(boxes) - 2 * len(pairs), it, boxes, labels))

    per = counter.per_class_result()
    rows = [{"class": c, **{k: per.get(c, {}).get(k, 0) for k in ("support", "tp", "fp", "fn", "precision", "recall",
                                                                  "f1")}} for c in CLASSES]
    macro = {k: float(np.mean([r[k] for r in rows])) for k in ("precision", "recall", "f1")}
    micro = counter.result()
    rows.append({"class": "macro", "support": sum(r["support"] for r in rows), "tp": "", "fp": "", "fn": "", **macro})
    rows.append({"class": "micro (cộng dồn)", "support": micro["tp"] + micro["fn"], "tp": micro["tp"],
                 "fp": micro["fp"], "fn": micro["fn"], "precision": micro["precision"], "recall": micro["recall"],
                 "f1": micro["f1"]})
    d = out_dir(EXP)
    save_csv(d / f"table_{a.split}.csv", rows)

    labs = list(CLASSES) + ["background"]
    cm = np.array([[conf_c.get((g, p), 0) for p in labs] for g in labs])
    save_csv(d / f"confusion_{a.split}.csv", [{"gt\\pred": g, **{p: int(cm[i, j]) for j, p in enumerate(labs)}}
                                              for i, g in enumerate(labs)])
    plot_confusion(cm, labs, d / f"confusion_{a.split}.png")

    err_dir = d / f"errors_{a.split}"
    err_dir.mkdir(exist_ok=True)
    worst = sorted(errors, key=lambda e: -e[0])[:3]
    for n_err, it, boxes, labels in worst:
        draw_errors(it, boxes, labels, err_dir / f"{it['id']}_err{n_err}.png")

    ultra = {} if a.no_map else map_ultralytics(a.split)
    save_json(d / f"config_{a.split}.json",
              {"split": a.split, "commit": commit_hash(), "conf": conf, "iou_thr": 0.5, "weights":
               "code/weights/yolo_vaipe.pt", "imgsz": data["imgsz"], "n_images": len(items), "ultralytics": ultra,
               "worst_images": [(it["id"], n) for n, it, _, _ in worst]})

    print(f"\nE-B2 trên {a.split} (conf {conf}):")
    print(f"  {'Lớp':<16} {'n':>5} {'P':>6} {'R':>6} {'F1':>6}")
    for r in rows:
        print(f"  {r['class']:<16} {r['support']:>5} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f}")
    if ultra:
        print(f"  mAP50 {ultra.get('mAP50', float('nan')):.3f} | mAP50-95 {ultra.get('mAP50_95', float('nan')):.3f}")
    print("  Nhầm lẫn lớn nhất (GT -> dự đoán):",
          [(g, p, n) for (g, p), n in conf_c.most_common() if g != p][:6])


if __name__ == "__main__":
    main()
