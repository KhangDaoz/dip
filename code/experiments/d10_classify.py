"""D10 — Phân loại 3 lớp (đơn thuốc / phiếu XN / khác) bằng đặc trưng + kNN có từ chối. Người làm: Hiếu.
Kế hoạch: KeHoach.md mục 5.1 (H9–H11), 5.2 (D10), 5.3.

Dữ liệu: mọi ảnh đi qua bước nắn ① (preprocess, bỏ qua phân loại) rồi CLAHE như trong pipeline.
  Train : bộ 1 train (200 ảnh ngẫu nhiên) · bộ 2 train (39) · bộ 3 train (200 ngẫu nhiên)
  Test  : bộ 1 test (100) · bộ 2 valid + test (18) · bộ 3 test (100) · bộ 3 open-set (100, lớp lạ không có lúc train)
  Chính : bộ 4 tự chụp (83 ảnh, nhãn theo thư mục) — kiểm tra "học thuộc nguồn" (KeHoach.md mục 3.2)
Phương án: đề xuất = HOG + GLCM + mật độ + đường kẻ + thống kê dòng; đối chứng = lược đồ màu 8×8×8 (C8 · S13).
k và độ đo khoảng cách chọn bằng 5-fold trên train (C8 · S41). Độ đo: Accuracy, Macro-F1, ma trận nhầm lẫn (C8 · S48–49).
Kết quả: results/d10/ ; mô hình cho pipeline: code/models/doc_knn.pkl

Chạy (từ code/): python experiments/d10_classify.py
"""
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classification import classify, features
from common.experiment import results_dir, save_config, save_figure, save_table
from common.io import read_image, save_json
from common.paths import DATASETS
from common.seed import set_seed
from preprocessing import enhance
from preprocessing.pipeline import preprocess

EXP_ID = "d10"
CONFIG = {"seed": 42, "n_train": {"prescription": 200, "other": 200}, "n_test": 100,
          "ks": [1, 3, 5, 7], "metrics": ["euclidean", "cosine", "chi2"], "reject_percentile": 95,
          "classes": ["prescription", "lab_report", "other"]}


def list_images(rng):
    def pick(paths, n):
        paths = sorted(paths)
        return [paths[i] for i in rng.permutation(len(paths))[:n]] if n and len(paths) > n else paths
    d1, d2, d3, d4 = DATASETS[1], DATASETS[2], DATASETS[3], DATASETS[4]
    lab3 = pd.read_csv(d3 / "labels.csv")
    rvl = lambda split: [d3 / f for f in lab3[lab3["split"] == split]["file"]]
    sets = {
        "train": [(p, "prescription") for p in pick((d1 / "train" / "images").glob("*.png"), 200)]
                 + [(p, "lab_report") for p in sorted((d2 / "train" / "images").glob("*"))]
                 + [(p, "other") for p in pick(rvl("train"), 200)],
        "test": [(p, "prescription") for p in pick((d1 / "test" / "images").glob("*.png"), CONFIG["n_test"])]
                + [(p, "lab_report") for s in ("valid", "test") for p in sorted((d2 / s / "images").glob("*"))]
                + [(p, "other") for p in pick(rvl("test"), CONFIG["n_test"])],
        "openset": [(p, "other") for p in pick(rvl("openset"), CONFIG["n_test"])],
        "set4": [(p, c) for c in CONFIG["classes"] for p in sorted((d4 / c).glob("*")) if p.suffix.lower() in
                 (".jpg", ".jpeg", ".png")],
    }
    return sets


def featurize(path, tmp):
    img = read_image(path)
    pre = preprocess(img, "f", tmp, doc_type="other")          # chỉ nắn + chỉnh nghiêng, không phân loại
    from common.paths import REPO_DIR
    rect = read_image(REPO_DIR / pre.rectified_path)
    gray = enhance.clahe(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY))
    return features.extract_all(gray), features.color_histogram(rect), features.extract_normalized(rect)


def evaluate(model, X, y, name, method, reject=True):
    pred, conf = classify.predict_with_reject(model, X, reject=reject)
    acc = float(np.mean(pred == y))
    f1 = classify.macro_f1(y, pred)
    cm = pd.crosstab(pd.Series(y, name="thật"), pd.Series(pred, name="dự đoán"))
    return {"set": name, "method": method, "n": len(y), "accuracy": acc, "macro_f1": f1}, cm, pred


def main():
    rng = set_seed(CONFIG["seed"])
    out = results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    sets = list_images(rng)
    feats = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, items in sets.items():
            F, Hc, Fn, y, paths = [], [], [], [], []
            for p, c in items:
                try:
                    f, hc, fn = featurize(p, tmp)
                except Exception as e:
                    print("bỏ", p.name, e)
                    continue
                F.append(f), Hc.append(hc), Fn.append(fn), y.append(c), paths.append(str(p))
            feats[name] = (np.array(F), np.array(Hc), np.array(Fn), np.array(y), paths)
            print(name, len(y))
    np.savez_compressed(out / "features.npz", **{f"{k}_{i}": v[i] for k, v in feats.items() for i in range(4)})

    ytr = feats["train"][3]
    rows, cms = [], {}
    methods = (("hog_glcm_layout", 0), ("color_hist", 1), ("normalized_binary_layout", 2))
    for method, Xi in methods:
        X = feats["train"][Xi]
        sel = classify.choose_k_cv(X, ytr, CONFIG["ks"], CONFIG["metrics"])
        save_json(out / f"cv_{method}.json", sel)
        model = classify.train_knn(X, ytr, sel["k"], sel["metric"], CONFIG["reject_percentile"])
        for name in ("test", "openset", "set4"):
            for reject in (True, False):
                r, cm, pred = evaluate(model, feats[name][Xi], feats[name][3], name, method, reject)
                r.update(k=sel["k"], metric=sel["metric"], reject=reject)
                if name == "openset":
                    r["rejected_or_other"] = float(np.mean(pred == "other"))
                rows.append(r)
                tag = f"{method}_{name}" + ("" if reject else "_noreject")
                cms[tag] = cm
                cm.to_csv(out / f"cm_{tag}.csv", encoding="utf-8-sig")
        sel_best = sel
        if method == "normalized_binary_layout":
            classify.save_model(model)   # mô hình pipeline: đặc trưng trên ảnh đã chuẩn hóa (ít học thuộc nguồn)
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID)
    print(df.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (method, _) in zip(axes, methods):
        cm = cms[f"{method}_set4"].reindex(index=CONFIG["classes"], columns=CONFIG["classes"], fill_value=0)
        ax.imshow(cm.values, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(cm.values[i, j]), ha="center", va="center")
        ax.set_xticks(range(3), CONFIG["classes"], rotation=20)
        ax.set_yticks(range(3), CONFIG["classes"])
        ax.set_title(f"{method} — bộ 4")
        ax.set_xlabel("dự đoán")
        ax.set_ylabel("thật")
    fig.tight_layout()
    save_figure(fig, EXP_ID, "confusion_set4")


if __name__ == "__main__":
    main()
