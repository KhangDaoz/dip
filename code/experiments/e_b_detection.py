"""E-B1 / E-B2 — phát hiện vùng chữ (plan §5.2). Người làm: Trường.

E-B0 (valid): quét conf YOLO, chọn conf tối đa F1 có lớp → dùng cho test.
E-B1 (test):  cổ điển vs YOLO (bỏ nhãn lớp), P/R/F1 @IoU 0,5.
E-B2 (test):  YOLO 6 lớp — P/R/F1 từng lớp + ma trận nhầm lẫn (mAP50 lấy từ notebook train / model.val).

    python experiments/e_b_detection.py                 # đủ B0 + B1 + B2
    python experiments/e_b_detection.py --conf 0.25     # bỏ quét, dùng conf cố định
"""
import argparse

import _v2  # noqa: F401
import numpy as np
import pandas as pd

from common.io import read_image, save_json
from detect.classical import detect_classical
from detect.yolo import detect_yolo
from metrics.detection import confusion_matrix, evaluate
from vaipe.dataset import list_images


def predict_all(images, fn):
    return [fn(read_image(img.image_path)) for img in images]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None, help="mặc định = imgsz lúc train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = _v2.out_dir("e_b")
    cfg = {"iou": 0.5, "imgsz": args.imgsz, "commit": _v2.git_hash()}

    # E-B0: chọn conf trên valid (dự đoán với conf thấp 1 lần, lọc theo ngưỡng)
    conf = args.conf
    if conf is None:
        valid = list_images("valid")[:args.limit]
        raw = predict_all(valid, lambda im: detect_yolo(im, conf=0.05, imgsz=args.imgsz))
        gts = [img.annotations for img in valid]
        rows = []
        for c in np.arange(0.05, 0.8, 0.05):
            r = evaluate(gts, [[b for b in p if b.score >= c] for p in raw], by_class=True)
            rows.append({"conf": round(float(c), 2), "P": r["all"][0], "R": r["all"][1], "F1": r["all"][2]})
        df0 = pd.DataFrame(rows)
        df0.to_csv(out / "b0_conf_valid.csv", index=False, encoding="utf-8-sig")
        conf = float(df0.loc[df0["F1"].idxmax(), "conf"])
        print(df0.to_string(index=False), f"\n→ chọn conf = {conf}")
    cfg["conf"] = conf

    test = list_images("test")[:args.limit]
    gts = [img.annotations for img in test]
    yolo = predict_all(test, lambda im: detect_yolo(im, conf=conf, imgsz=args.imgsz))
    classical = predict_all(test, detect_classical)

    # E-B1: không phân lớp
    rows = []
    for name, preds in (("classical (Sauvola+giãn)", classical), (f"YOLOv8s {args.imgsz or 'train-imgsz'}", yolo)):
        r = evaluate(gts, preds, by_class=False)
        rows.append({"method": name, "P": r["all"][0], "R": r["all"][1], "F1": r["all"][2], **r["counts"]})
    df1 = pd.DataFrame(rows)
    df1.to_csv(out / "b1_agnostic_test.csv", index=False, encoding="utf-8-sig")
    print("\nE-B1\n", df1.to_string(index=False))

    # E-B2: theo lớp + ma trận nhầm lẫn
    r = evaluate(gts, yolo, by_class=True)
    df2 = pd.DataFrame([{"class": k, "P": v[0], "R": v[1], "F1": v[2]} for k, v in r.items() if k != "counts"])
    df2.to_csv(out / "b2_per_class_test.csv", index=False, encoding="utf-8-sig")
    M, names = confusion_matrix(gts, yolo)
    pd.DataFrame(M, index=[f"gt_{n}" for n in names], columns=[f"pred_{n}" for n in names]).to_csv(
        out / "b2_confusion_test.csv", encoding="utf-8-sig")
    print("\nE-B2\n", df2.to_string(index=False))
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(np.log1p(M), cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Đáp án")
        fig.tight_layout()
        fig.savefig(out / "b2_confusion.png", dpi=200)
    except Exception as e:
        print("bỏ qua vẽ hình:", e)
    save_json(out / "config.json", cfg)


if __name__ == "__main__":
    main()
