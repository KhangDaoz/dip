"""D1 — IQA như bộ phát hiện mờ / lóa
Người làm: Hiếu. Kế hoạch: team/KeHoach_Hieu.md (mục 3.2).

Dữ liệu: bộ 6 — chọn ngưỡng trên val, báo cáo trên test.
Phần mờ (bộ 6 v1): nhãn "mờ" = blur_level ≥ 1 (chính) và ≥ 2 (mờ đủ ảnh hưởng đọc chữ).
    Điểm độ nét tính trên ảnh đã resize cạnh dài 1000 px (như assess_quality); điểm thấp = mờ.
Phần lóa: chờ bộ 6 v2 có mask lóa (04/11) — CONFIG glare_* để sẵn.
Độ đo: ROC, AUC; F1, precision, recall trên test tại ngưỡng chọn trên val (F1 lớn nhất).
Đầu ra: results/d1/table.csv (điểm từng mẫu), results/d1/summary.csv, results/d1/figure.png (ROC test)

Chạy: python experiments/d1_iqa_roc.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

import bo6
from common.experiment import save_config, save_figure, save_table
from common.seed import set_seed
from preprocessing.quality import BLUR_THRESHOLD, blur_score, resize_long_side

EXP_ID = "d1"
CONFIG = {"seed": 42, "blur_methods": ["laplacian", "tenengrad", "fft"], "long_side": 1000,
          "positive_levels": [1, 2], "select_split": "val", "report_split": "test",
          "glare_v": [220, 230, 240, 250], "glare_s": [20, 30, 40, 60]}
LABELS = {"laplacian": "Phương sai Laplacian", "tenengrad": "Tenengrad", "fft": "Năng lượng FFT cao tần"}


def load_data(config: dict) -> pd.DataFrame:
    return bo6.manifest((config["select_split"], config["report_split"]))


def run(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for _, r in data.iterrows():
        small, _ = resize_long_side(bo6.degraded(r), config["long_side"])
        row = {"sample_id": r.sample_id, "split": r.split, "blur_level": r.blur_level, "blur_kind": r.blur_kind}
        for m in config["blur_methods"]:
            row[m] = blur_score(small, m)
        rows.append(row)
    return pd.DataFrame(rows)


def _best_threshold(score: np.ndarray, y: np.ndarray) -> float:
    """Ngưỡng t (mờ khi score < t) cho F1 lớn nhất; t đặt giữa hai điểm liên tiếp."""
    s = np.sort(np.unique(score))
    cands = np.concatenate([[s[0] - 1e-9], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-9]])
    f1 = [f1_score(y, score < t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1))])


def summarize(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    sel, rep = df[df.split == config["select_split"]], df[df.split == config["report_split"]]
    out = []
    for level in config["positive_levels"]:
        for m in config["blur_methods"]:
            y_sel, y_rep = (sel.blur_level >= level).values, (rep.blur_level >= level).values
            t = _best_threshold(sel[m].values, y_sel)
            pred = rep[m].values < t
            out.append({"positive": f"blur_level>={level}", "method": m, "n_pos_test": int(y_rep.sum()),
                        "auc_test": roc_auc_score(y_rep, -rep[m].values), "threshold_val": t,
                        "f1_test": f1_score(y_rep, pred), "precision_test": precision_score(y_rep, pred, zero_division=0),
                        "recall_test": recall_score(y_rep, pred)})
    # Ngưỡng đang dùng trong quality.py, để so sánh
    y_rep = (rep.blur_level >= 1).values
    pred = rep["laplacian"].values < BLUR_THRESHOLD
    out.append({"positive": "blur_level>=1", "method": "laplacian (ngưỡng hiện tại)", "n_pos_test": int(y_rep.sum()),
                "auc_test": np.nan, "threshold_val": BLUR_THRESHOLD, "f1_test": f1_score(y_rep, pred),
                "precision_test": precision_score(y_rep, pred, zero_division=0), "recall_test": recall_score(y_rep, pred)})
    return pd.DataFrame(out).round(4)


def plot(df: pd.DataFrame, config: dict):
    rep = df[df.split == config["report_split"]]
    y = rep.blur_level >= 1
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for m in config["blur_methods"]:
        fpr, tpr, _ = roc_curve(y, -rep[m])
        ax.plot(fpr, tpr, label=f"{LABELS[m]} (AUC={roc_auc_score(y, -rep[m]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Tỉ lệ dương tính giả")
    ax.set_ylabel("Tỉ lệ phát hiện đúng ảnh mờ")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    set_seed(CONFIG["seed"])
    save_config(CONFIG, EXP_ID)
    df = run(load_data(CONFIG), CONFIG)
    print("Bảng:", save_table(df, EXP_ID))
    summary = summarize(df, CONFIG)
    save_table(summary, EXP_ID, "summary")
    print(summary.to_string(index=False))
    print("Hình:", save_figure(plot(df, CONFIG), EXP_ID))


if __name__ == "__main__":
    main()
