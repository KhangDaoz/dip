"""E-B0 / E-B1 / E-B2 — khối B phát hiện vùng chữ (plan §5.2).

E-B0 (valid): (a) quét ngưỡng conf của YOLO; (b) lưới tham số a, v, xóa đường kẻ của detector cổ điển.
              Chọn cấu hình tối đa F1 trên valid — test KHÔNG dùng để chọn.
E-B1 (test):  cổ điển vs YOLO, KHÔNG xét lớp — P/R/F1 @IoU 0,5 (ghép tham lam, cộng dồn cả tập).
E-B2 (test):  YOLO 6 lớp — P/R/F1 từng lớp, ma trận nhầm lẫn, mAP50 / mAP50-95 (ultralytics model.val).

    python experiments/e_b_detection.py                  # đủ B0 + B1 + B2
    python experiments/e_b_detection.py --skip-grid      # dùng a, v mặc định của detect/classical.py
Kết quả: results/e_b/{b0_yolo_conf_valid.csv, b0_classical_grid_valid.csv, b1_test.csv, b2_per_class_test.csv,
         b2_confusion_test.csv, b2_confusion.png, b2_map_test.json, config.json}
"""
import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd

from common.contracts import BOX_LABELS
from common.experiment import results_dir, save_config, save_figure, save_table
from common.io import read_image
from common.paths import VAIPE_YOLO
from detect import classical as C
from detect.yolo import CONF, IMGSZ, detect_yolo, load_yolo
from metrics.detection import confusion_matrix, evaluate
from vaipe.dataset import list_images

EXP_ID = "e_b"


def run(images, fn):
    t = time.perf_counter()
    preds = [fn(read_image(im.image_path)) for im in images]
    return preds, (time.perf_counter() - t) / max(1, len(images))


def agnostic(boxes):
    for b in boxes:
        b.label = None
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--grid-images", type=int, default=60, help="số ảnh valid dùng cho lưới cổ điển")
    args = ap.parse_args()
    out = results_dir(EXP_ID)
    cfg = {"iou_match": 0.5, "imgsz": IMGSZ}
    valid, test = list_images("valid"), list_images("test")

    # ---- E-B0a: ngưỡng conf YOLO trên valid (dự đoán 1 lần ở conf thấp rồi lọc)
    raw, _ = run(valid, lambda im: detect_yolo(im, conf=0.05))
    gts = [im.annotations for im in valid]
    rows = []
    for c in np.round(np.arange(0.05, 0.86, 0.05), 2):
        p = [[b for b in bs if b.score >= c] for bs in raw]
        r = evaluate(gts, p, by_class=True)
        rows.append({"conf": c, "P": r["all"][0], "R": r["all"][1], "F1": r["all"][2],
                     "F1_mean6": float(np.mean([r[l][2] for l in BOX_LABELS]))})
    b0 = pd.DataFrame(rows)
    save_table(b0, EXP_ID, "b0_yolo_conf_valid")
    best_conf = float(b0.loc[b0["F1_mean6"].idxmax(), "conf"])
    print(f"[E-B0a] conf tốt nhất trên valid (lưới 0,05): {best_conf}; code đang dùng CONF={CONF}")
    cfg["yolo_conf"] = CONF

    # ---- E-B0b: lưới tham số cổ điển trên một phần valid
    best = {"a": C.A, "v": C.V, "lines": C.LINES}
    if not args.skip_grid:
        sub = valid[:args.grid_images]
        sub_gt = [im.annotations for im in sub]
        imgs = [read_image(im.image_path) for im in sub]
        rows = []
        for a, v, lines in itertools.product([0.5, 1.0, 1.6, 2.0, 2.5, 3.0, 4.0], [0.0, 0.2, 0.3, 0.5], [True, False]):
            p = [C.detect_classical(im, a=a, v=v, lines=lines) for im in imgs]
            r = evaluate(sub_gt, p)["all"]
            rows.append({"a": a, "v": v, "lines": lines, "P": r[0], "R": r[1], "F1": r[2]})
        grid = pd.DataFrame(rows).sort_values("F1", ascending=False)
        save_table(grid, EXP_ID, "b0_classical_grid_valid")
        top = grid.iloc[0]
        best = {"a": float(top["a"]), "v": float(top["v"]), "lines": bool(top["lines"])}
        print(f"[E-B0b] cổ điển tốt nhất trên {len(sub)} ảnh valid: {best}, F1={top['F1']:.3f}")
    cfg["classical"] = best

    # ---- E-B1: test, không xét lớp
    gts = [im.annotations for im in test]
    yolo, t_yolo = run(test, lambda im: detect_yolo(im))
    clas, t_clas = run(test, lambda im: C.detect_classical(im, **best))
    rows = []
    for name, preds, t in (("Cổ điển (Sauvola + giãn)", clas, t_clas), ("YOLOv8s", yolo, t_yolo)):
        r = evaluate(gts, [agnostic([type(b)(**b.to_dict()) for b in bs]) for bs in preds])
        rows.append({"method": name, "P": r["all"][0], "R": r["all"][1], "F1": r["all"][2],
                     "TP": r["counts"].get("tp", 0), "FP": r["counts"].get("fp", 0), "FN": r["counts"].get("fn", 0), "s_per_img": t})
    b1 = pd.DataFrame(rows)
    save_table(b1, EXP_ID, "b1_test")
    print("\n[E-B1] test, không xét lớp\n", b1.round(4).to_string(index=False))

    # ---- E-B2: YOLO theo lớp + ma trận nhầm lẫn + mAP
    r = evaluate(gts, yolo, by_class=True)
    b2 = pd.DataFrame([{"class": l, "P": r[l][0], "R": r[l][1], "F1": r[l][2],
                        "n_gt": r["counts"].get(f"{l}/tp", 0) + r["counts"].get(f"{l}/fn", 0)} for l in BOX_LABELS]
                      + [{"class": "all", "P": r["all"][0], "R": r["all"][1], "F1": r["all"][2],
                          "n_gt": r["counts"].get("tp", 0) + r["counts"].get("fn", 0)}])
    save_table(b2, EXP_ID, "b2_per_class_test")
    M, names = confusion_matrix(gts, yolo)
    pd.DataFrame(M, index=[f"gt_{n}" for n in names], columns=[f"pred_{n}" for n in names]).to_csv(
        out / "b2_confusion_test.csv", encoding="utf-8-sig")
    print("\n[E-B2] YOLO theo lớp (test)\n", b2.round(4).to_string(index=False))

    m = load_yolo().val(data=str(VAIPE_YOLO / "data.yaml"), split="test", imgsz=IMGSZ, batch=8, plots=False,
                        verbose=False, project=str(out), name="val_test", exist_ok=True)
    maps = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
            "mAP50_per_class": {BOX_LABELS[i]: float(v) for i, v in enumerate(m.box.ap50)}}
    (out / "b2_map_test.json").write_text(json.dumps(maps, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[E-B2] mAP50 = {maps['mAP50']:.4f}, mAP50-95 = {maps['mAP50_95']:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.log1p(M), cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8, color="black" if M[i, j] < M.max() / 3 else "white")
    ax.set_xlabel("Dự đoán (bg = không có box)")
    ax.set_ylabel("Đáp án (bg = box thừa)")
    fig.tight_layout()
    save_figure(fig, EXP_ID, "b2_confusion")
    save_config(cfg, EXP_ID)


if __name__ == "__main__":
    main()
