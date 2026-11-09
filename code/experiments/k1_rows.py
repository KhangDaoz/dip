"""K1 — So sánh gom dòng bằng horizontal projection với tâm y cố định.

Chạy từ thư mục code/: python experiments/k1_rows.py
GT pilot được duyệt lưu local tại results/khang_gt_pilot/ (không commit dataset).
"""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from common.experiment import save_config, save_figure, save_table
from common.paths import RESULTS_DIR
from common.seed import set_seed
from table_extraction.rows import rows_fixed, rows_projection

EXP_ID = "k1"
CONFIG = {
    "seed": 42,
    "methods": ["projection", "fixed"],
    "y_tol": [0.3, 0.5, 0.7],
    "ground_truth": "results/khang_gt_pilot/candidate_ground_truth.json",
    "input_boxes": "reviewed Dataset 2 role boxes (isolates row grouping from OCR/detection)",
    "metrics": ["pairwise_precision", "pairwise_recall", "pairwise_f1", "exact_row_accuracy"],
}


def load_data(config: dict):
    """Đọc dữ liệu từ DATASETS[...]; trả về list mẫu (ảnh + ground truth)."""
    gt_path = RESULTS_DIR / "khang_gt_pilot" / "candidate_ground_truth.json"
    if not gt_path.is_file():
        raise FileNotFoundError(
            f"Thiếu pilot GT: {gt_path.relative_to(RESULTS_DIR.parent)}. "
            "Tạo/đặt results/khang_gt_pilot/candidate_ground_truth.json trước khi chạy K1."
        )
    records = json.loads(gt_path.read_text(encoding="utf-8"))
    data = []
    for sample in records["samples"]:
        binary_path = RESULTS_DIR.parent / sample["phase2_binary_preview"]
        binary = cv2.imread(str(binary_path), cv2.IMREAD_GRAYSCALE)
        if binary is None:
            raise FileNotFoundError(f"Không đọc được ảnh nhị phân pilot: {binary_path}")
        words, ground_truth = [], []
        for row in sample["rows_proposed"]:
            row_group = []
            for fields in row["fields_by_role"].values():
                for field in fields:
                    x1, y1, x2, y2 = field["bbox_rectified_xyxy"]
                    row_group.append(len(words))
                    words.append([x1, y1, x2 - x1, y2 - y1])
            if row_group:
                ground_truth.append(sorted(row_group))
        data.append({
            "sample_id": sample["sample_id"],
            "binary": binary,
            "words": words,
            "ground_truth": ground_truth,
        })
    if not data:
        raise ValueError("Pilot GT không có mẫu nào")
    return data


def _pairwise_scores(predicted: list, ground_truth: list, n_items: int) -> tuple[float, float, float]:
    pred_label = [-1] * n_items
    gt_label = [-1] * n_items
    for label, group in enumerate(predicted):
        for index in group:
            pred_label[index] = label
    for label, group in enumerate(ground_truth):
        for index in group:
            gt_label[index] = label
    tp = fp = fn = 0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            same_pred = pred_label[i] == pred_label[j] and pred_label[i] >= 0
            same_gt = gt_label[i] == gt_label[j] and gt_label[i] >= 0
            tp += int(same_pred and same_gt)
            fp += int(same_pred and not same_gt)
            fn += int(same_gt and not same_pred)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _exact_row_accuracy(predicted: list, ground_truth: list) -> float:
    pred = {frozenset(group) for group in predicted}
    gt = {frozenset(group) for group in ground_truth}
    if not gt:
        return 1.0
    return len(pred & gt) / len(gt)


def run(data, config: dict) -> pd.DataFrame:
    """Chạy từng phương án trên từng mẫu, mỗi dòng của DataFrame = (mẫu, phương án, độ đo...)."""
    results = []
    for sample in data:
        for method in config["methods"]:
            tolerances = config["y_tol"] if method == "fixed" else [None]
            for y_tol in tolerances:
                if method == "projection":
                    predicted = rows_projection(sample["binary"], sample["words"])
                    param = "adaptive_char_height"
                elif method == "fixed":
                    predicted = rows_fixed(sample["words"], y_tol=y_tol)
                    param = y_tol
                else:
                    raise ValueError(f"Phương pháp K1 không hỗ trợ: {method}")
                precision, recall, f1 = _pairwise_scores(
                    predicted, sample["ground_truth"], len(sample["words"]))
                results.append({
                    "sample_id": sample["sample_id"],
                    "method": method,
                    "parameter": param,
                    "gt_rows": len(sample["ground_truth"]),
                    "predicted_rows": len(predicted),
                    "pairwise_precision": precision,
                    "pairwise_recall": recall,
                    "pairwise_f1": f1,
                    "exact_row_accuracy": _exact_row_accuracy(predicted, sample["ground_truth"]),
                })
    return pd.DataFrame(results)


def plot(df: pd.DataFrame):
    """Vẽ hình đưa vào báo cáo; trả về matplotlib Figure hoặc None."""
    summary = df.groupby(["method", "parameter"], dropna=False)[
        ["pairwise_precision", "pairwise_recall", "pairwise_f1", "exact_row_accuracy"]
    ].mean().reset_index()
    labels = ["Projection" if row.method == "projection" else f"Fixed y_tol={row.parameter}"
              for row in summary.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(summary))
    ax.bar(x, summary["pairwise_f1"], color=["#2673b8" if m == "projection" else "#e48b36"
                                               for m in summary["method"]])
    ax.set_xticks(list(x), labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pairwise F1 (trung bình 4 ảnh)")
    ax.set_title("K1 — Độ chính xác gom cụm từ vào cùng hàng")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(summary["pairwise_f1"]):
        ax.text(i, value + 0.015, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def save_summary(df: pd.DataFrame) -> Path:
    path = RESULTS_DIR / EXP_ID / "summary.csv"
    summary = df.groupby(["method", "parameter"], dropna=False)[
        ["gt_rows", "predicted_rows", "pairwise_precision", "pairwise_recall",
         "pairwise_f1", "exact_row_accuracy"]
    ].mean().reset_index()
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main():
    set_seed(CONFIG["seed"])
    save_config(CONFIG, EXP_ID)
    df = run(load_data(CONFIG), CONFIG)
    print("Bảng:", save_table(df, EXP_ID))
    print("Tổng hợp:", save_summary(df))
    print(df.groupby(["method", "parameter"], dropna=False)[
        ["pairwise_precision", "pairwise_recall", "pairwise_f1", "exact_row_accuracy"]
    ].mean().to_string(index=False))
    fig = plot(df)
    if fig is not None:
        print("Hình:", save_figure(fig, EXP_ID))


if __name__ == "__main__":
    main()
