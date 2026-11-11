"""E-B1 — Học sâu hơn cổ điển bao nhiêu ở việc tìm vùng chữ? (khối B, Trường; plan.md mục 5.2, Truong_KhoiB.md B6–B10)

Bảng (P/R/F1 @IoU 0,5 cộng dồn cả tập; kèm F1 @IoU 0,3 để tham khảo):
  1. Cổ điển (Sauvola → xóa kẻ → giãn theo h → CC), không phân lớp
  2. YOLOv8s 1024, bỏ nhãn lớp khi ghép
  3. YOLOv8s 1024, bắt buộc đúng lớp
(Plan gốc có thêm YOLO 640; nhóm đã bỏ ngày 11/11.)

Các bước — chạy từ thư mục code/ bằng môi trường có ultralytics (.venv):
  python experiments/e_b1.py --grid          # valid: chọn a, b, xóa kẻ cho bản cổ điển  -> classical_grid.csv
  python experiments/e_b1.py --sweep-conf    # valid: chọn conf cho YOLO (max F1 không phân lớp) -> conf_sweep.csv
  python experiments/e_b1.py --split valid   # bảng trên valid (chạy thử, chạy bao nhiêu lần cũng được)
  python experiments/e_b1.py --split test    # SỐ LIỆU CUỐI — chỉ một lần (chạy lại phải có --rerun-reason)
Lựa chọn trên valid lưu ở results/e_b1/chosen.json; bước test đọc file này, không chọn lại gì.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from e_b_common import (commit_hash, filter_conf, guard_test, load_split, out_dir, read_bgr, save_csv, save_json,
                        yolo_predictions)
from detect.classical import ClassicalDetector, ClassicalParams
from metrics.detection import DetectionCounter

EXP = "e_b1"
GRID = {"a": [0.1, 0.2, 0.3, 0.5, 0.8], "b": [0.8, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        "remove_lines": [True, False]}   # mở rộng sau lần 1: tối ưu rơi vào biên a=0,3, b=3,0
CONF_GRID = [round(x, 2) for x in np.arange(0.05, 0.91, 0.05)]


def chosen_path():
    return out_dir(EXP) / "chosen.json"


def load_chosen() -> dict:
    import json
    p = chosen_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def update_chosen(**kw):
    c = load_chosen()
    c.update(kw)
    save_json(chosen_path(), c)


def run_grid():
    items = load_split("valid")
    counters = {combo: DetectionCounter() for combo in itertools.product(*GRID.values())}
    t0 = time.time()
    for k, it in enumerate(items):
        img = read_bgr(it["path"])
        for rl in GRID["remove_lines"]:
            det = ClassicalDetector(remove_lines=rl)
            binary, h = det.binarize(img)
            for a, b in itertools.product(GRID["a"], GRID["b"]):
                counters[(a, b, rl)].add(it["boxes"], det.boxes_from_binary(binary, h, a, b))
        if (k + 1) % 25 == 0:
            print(f"  grid: {k + 1}/{len(items)} ảnh ({time.time() - t0:.0f}s)", flush=True)
    rows = [{"a": a, "b": b, "remove_lines": rl, **counters[(a, b, rl)].result()} for a, b, rl in counters]
    rows.sort(key=lambda r: -r["f1"])
    save_csv(out_dir(EXP) / "classical_grid.csv", rows)
    best = rows[0]
    update_chosen(classical={**ClassicalParams().to_dict(), "a": best["a"], "b": best["b"],
                             "remove_lines": best["remove_lines"]}, classical_valid_f1=best["f1"])
    print("Top 5 (valid):")
    for r in rows[:5]:
        print(f"  a={r['a']} b={r['b']} xóa_kẻ={r['remove_lines']}: P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f}")


def run_sweep_conf():
    items = load_split("valid")
    data = yolo_predictions("valid", items)
    rows = []
    for conf in CONF_GRID:
        c = DetectionCounter()
        for it in items:
            boxes, _, _ = filter_conf(data["preds"][it["id"]], conf)
            c.add(it["boxes"], boxes)
        rows.append({"conf": conf, **c.result()})
    save_csv(out_dir(EXP) / "conf_sweep.csv", rows)
    best = max(rows, key=lambda r: (r["f1"], -abs(r["conf"] - 0.25)))   # hòa thì gần mặc định 0,25
    update_chosen(conf=best["conf"], conf_valid_f1=best["f1"])
    for r in rows:
        mark = "  <-- chọn" if r is best else ""
        print(f"  conf {r['conf']:.2f}: P {r['precision']:.4f} R {r['recall']:.4f} F1 {r['f1']:.4f}{mark}")


def run_table(split: str, rerun_reason: str | None):
    chosen = load_chosen()
    if "conf" not in chosen or "classical" not in chosen:
        raise SystemExit("Chưa chọn tham số trên valid: chạy --grid và --sweep-conf trước.")
    if split == "test":
        guard_test(EXP, rerun_reason)
    items = load_split(split)
    conf = chosen["conf"]

    # 1. Cổ điển
    det = ClassicalDetector(ClassicalParams(**chosen["classical"]))
    c50, c30, times = DetectionCounter(0.5), DetectionCounter(0.3), []
    for it in items:
        img = read_bgr(it["path"])
        t = time.perf_counter()
        boxes = [tb.box for tb in det.detect(img)]
        times.append(time.perf_counter() - t)
        c50.add(it["boxes"], boxes)
        c30.add(it["boxes"], boxes)
    cp = chosen["classical"]
    name = f"Cổ điển (Sauvola{' + xóa kẻ' if cp['remove_lines'] else ''} + giãn {cp['a']}h×{cp['b']}h + CC)"
    rows = [{"method": name, "class_aware": False, **c50.result(),
             "f1_iou30": c30.result()["f1"], "sec_per_image": float(np.mean(times))}]

    # 2–3. YOLO
    data = yolo_predictions(split, items)
    for aware in (False, True):
        c50, c30 = DetectionCounter(0.5, aware), DetectionCounter(0.3, aware)
        for it in items:
            boxes, labels, _ = filter_conf(data["preds"][it["id"]], conf)
            kw = {"gt_labels": it["labels"], "pred_labels": labels} if aware else {}
            c50.add(it["boxes"], boxes, **kw)
            c30.add(it["boxes"], boxes, **kw)
        name = "YOLOv8s 1024, " + ("bắt buộc đúng lớp" if aware else "bỏ nhãn lớp")
        rows.append({"method": name, "class_aware": aware, **c50.result(), "f1_iou30": c30.result()["f1"],
                     "sec_per_image": data["sec_per_image"]})

    save_csv(out_dir(EXP) / f"table_{split}.csv", rows)
    save_json(out_dir(EXP) / f"config_{split}.json",
              {"split": split, "commit": commit_hash(), "conf": conf, "classical": chosen["classical"],
               "yolo": {"weights": "code/weights/yolo_vaipe.pt", "imgsz": data["imgsz"], "iou_nms": data["iou"],
                        "device_timing": data["device"]},
               "iou_thr": 0.5, "n_images": len(items), "n_gt_boxes": sum(len(it["boxes"]) for it in items)})
    print(f"\nE-B1 trên {split} ({len(items)} ảnh, conf {conf}):")
    print(f"  {'Phương án':<42} {'P':>6} {'R':>6} {'F1':>6} {'F1@.3':>6} {'s/ảnh':>6}")
    for r in rows:
        print(f"  {r['method']:<42} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} "
              f"{r['f1_iou30']:6.3f} {r['sec_per_image']:6.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--grid", action="store_true", help="chọn tham số bản cổ điển trên valid")
    g.add_argument("--sweep-conf", action="store_true", help="chọn conf YOLO trên valid")
    g.add_argument("--split", choices=["valid", "test"], help="chạy bảng E-B1")
    ap.add_argument("--rerun-reason", help="bắt buộc khi chạy lại trên test")
    a = ap.parse_args(argv)
    if a.grid:
        run_grid()
    elif a.sweep_conf:
        run_sweep_conf()
    else:
        run_table(a.split, a.rerun_reason)


if __name__ == "__main__":
    main()
