"""S1 — Tách khoản thuốc: phương pháp đề xuất (Dilation + quy tắc) vs các phương án gom
Người làm: Thắng. Kế hoạch: team/KeHoach_Thang.md (mục 4.2, bảng tham số 4.3).

Dữ liệu: ground truth khoản thuốc results/t04/t04_gt_30_images.json, chỉ dùng 15 ảnh VAIPE (bộ 1), bỏ ảnh `exclude_from_eval`.
15 ảnh Roboflow đơn thuốc Ấn Độ trong file GT bị bỏ qua: bộ này đã loại khỏi dữ liệu nhóm từ 08/11.
Bản đầu: cắt theo vùng thuốc GT (lỗi khoanh vùng đo riêng ở S4), nhị phân Otsu trên ảnh xám gốc.
Chọn tham số trên ảnh split `valid`, báo kết quả trên ảnh còn lại (`test`, `test_labeled`).
Độ đo: F1 khoản thuốc (khớp khi IoU > 0.5, theo KeHoach_Thang mục 4.2), IoU trung bình của các cặp khớp
Đầu ra: results/s1/table.csv (từng ảnh × cấu hình), results/s1/sweep.csv (quét tham số trên valid),
        results/s1/summary.csv (bảng chính, tập test), results/s1/figure.png, results/s1/config.json

Chạy: python experiments/s1_items.py
"""
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from common.experiment import save_config, save_figure, save_table
from common.io import load_json, read_image
from common.paths import REPO_DIR, RESULTS_DIR
from common.seed import set_seed
from prescription_segmentation import items, lines

EXP_ID = "s1"
CONFIG = {
    "seed": 42,
    "gt": "t04/t04_gt_30_images.json",
    "binarize": "otsu",
    "crop": "gt_region",
    "select_split": "valid",
    "iou_threshold": 0.5,
    "line_methods": ["projection", "components"],
    # phương pháp → lưới tham số. Bảng 4.3 dự kiến SE dọc 0,5–1,5 × chiều cao chữ; mở rộng xuống 0,1 vì khoảng trống
    # giữa hai dòng trong một khoản chỉ ~0,15–0,25 × chiều cao chữ (VAIPE). Thụt lề 0,5–3 × chiều cao chữ.
    "methods": {
        "rules": {"se_factor": [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5], "indent_tol": [0.5, 1.0, 2.0, 3.0]},
        "dilation": {"se_factor": [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5]},
        "per_line": {},
        "fixed_gap": {"gap_px": [5, 10, 15, 20]},
        "fixed": {"lines_per_item": [2, 3]},
    },
}

LINE_METHODS = {"projection": lines.lines_projection, "components": lines.lines_components}


def _grid(params: dict) -> list:
    combos = [{}]
    for name, values in params.items():
        combos = [{**c, name: v} for c in combos for v in values]
    return combos


def group(method: str, line_boxes: list, binary, p: dict) -> list:
    if method == "rules":
        return items.group_items_rules(line_boxes, binary=binary, se_factor=p["se_factor"], indent_tol=p["indent_tol"])
    if method == "dilation":
        return items.group_items_dilation(line_boxes, binary=binary, se_factor=p["se_factor"])
    if method == "per_line":
        return items.group_items_fixed(line_boxes, lines_per_item=1)
    if method == "fixed_gap":
        return items.group_items_fixed_gap(line_boxes, gap_px=p["gap_px"])
    if method == "fixed":
        return items.group_items_fixed(line_boxes, lines_per_item=p["lines_per_item"])
    raise ValueError(method)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def match(pred: list, gt: list, thr: float):
    """Ghép tham lam theo IoU lớn nhất, khớp khi IoU > thr; trả về (số cặp khớp, list IoU của các cặp khớp)."""
    used, ious = set(), []
    for p in pred:
        cands = [(iou(p, g), j) for j, g in enumerate(gt) if j not in used]
        best = max(cands, default=(0.0, -1))
        if best[0] > thr:
            used.add(best[1])
            ious.append(best[0])
    return len(ious), ious


def load_data(config: dict):
    """Đọc ground truth khoản thuốc; trả về list mẫu (ảnh nhị phân vùng thuốc + bbox khoản theo tọa độ vùng)."""
    data = []
    for e in load_json(RESULTS_DIR / config["gt"]):
        if e.get("exclude_from_eval") or "vaipe" not in e["origin_dataset"]:
            continue
        x, y, w, h = e["medicine_region_bbox"]
        gray = read_image(REPO_DIR / e["image_path"], gray=True)[y:y + h, x:x + w]
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        gt = [[it["bbox"][0] - x, it["bbox"][1] - y, it["bbox"][2], it["bbox"][3]] for it in e["items"]]
        source = "vaipe" if "vaipe" in e["origin_dataset"] else "roboflow"
        split = "valid" if e["split_role"] == "valid" else "test"
        data.append({"id": e["candidate_id"], "source": source, "split": split, "binary": binary, "gt": gt})
    return data


def run(data, config: dict) -> pd.DataFrame:
    """Mỗi dòng = (mẫu, cách tách dòng, phương án gom, bộ tham số, độ đo)."""
    rows = []
    for s in data:
        for ln_name in config["line_methods"]:
            line_boxes = LINE_METHODS[ln_name](s["binary"])
            for method, params in config["methods"].items():
                for p in _grid(params):
                    pred = group(method, line_boxes, s["binary"], p)
                    tp, ious = match(pred, s["gt"], config["iou_threshold"])
                    rows.append({"id": s["id"], "source": s["source"], "split": s["split"], "lines": ln_name,
                                 "method": method, "params": str(p), **p, "n_lines": len(line_boxes),
                                 "n_pred": len(pred), "n_gt": len(s["gt"]), "tp": tp, "iou_sum": sum(ious)})
    return pd.DataFrame(rows)


def prf(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Micro P / R / F1 + IoU trung bình của cặp khớp, gộp theo keys."""
    g = df.groupby(keys)[["tp", "n_pred", "n_gt", "iou_sum"]].sum().reset_index()
    g["precision"] = g.tp / g.n_pred.clip(lower=1)
    g["recall"] = g.tp / g.n_gt
    g["f1"] = 2 * g.precision * g.recall / (g.precision + g.recall).clip(lower=1e-9)
    g["mean_iou_matched"] = g.iou_sum / g.tp.clip(lower=1)
    return g.drop(columns="iou_sum").round(3)


def select_params(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Quét tham số trên split chọn (gộp 2 nguồn); cột `chosen` đánh dấu bộ tham số F1 cao nhất của mỗi phương án."""
    sweep = prf(df[df.split == config["select_split"]], ["lines", "method", "params"])
    best = sweep.loc[sweep.groupby(["lines", "method"]).f1.idxmax(), ["lines", "method", "params"]]
    best["chosen"] = True
    return sweep.merge(best, how="left").fillna({"chosen": False})


def summarize(df: pd.DataFrame, sweep: pd.DataFrame) -> pd.DataFrame:
    """Bảng chính: tham số đã chọn trên valid, đánh giá trên test theo từng nguồn và gộp."""
    chosen = sweep[sweep.chosen][["lines", "method", "params"]]
    test = df[df.split == "test"].merge(chosen)
    by_source = prf(test, ["lines", "method", "params", "source"])
    overall = prf(test, ["lines", "method", "params"]).assign(source="all")
    return pd.concat([by_source, overall]).sort_values(["lines", "source", "f1"], ascending=[True, True, False])


def plot(df: pd.DataFrame):
    """F1 trên valid theo hệ số SE dọc — hình cho bảng tham số & lý do chọn."""
    valid = df[(df.split == CONFIG["select_split"]) & (df.method.isin(["rules", "dilation"]))]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, ln_name in zip(axes, CONFIG["line_methods"]):
        sub = valid[valid["lines"] == ln_name]
        for method in ["rules", "dilation"]:
            m = sub[sub.method == method]
            if method == "rules":
                tol = prf(m, ["indent_tol"]).sort_values("f1").indent_tol.iloc[-1]
                m = m[m.indent_tol == tol]
                label = f"Dilation + quy tắc (thụt lề {tol:g}h)"
            else:
                label = "Chỉ Dilation"
            curve = prf(m, ["se_factor"])
            ax.plot(curve.se_factor, curve.f1, marker="o", label=label)
        ax.set_title(f"Tách dòng: {ln_name}")
        ax.set_xlabel("SE dọc (× chiều cao chữ)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"F1 khoản thuốc ({CONFIG['select_split']})")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    set_seed(CONFIG["seed"])
    save_config(CONFIG, EXP_ID)
    df = run(load_data(CONFIG), CONFIG)
    print("Bảng:", save_table(df, EXP_ID))
    sweep = select_params(df, CONFIG)
    print("Quét tham số:", save_table(sweep, EXP_ID, "sweep"))
    summary = summarize(df, sweep)
    print("Tổng hợp:", save_table(summary, EXP_ID, "summary"))
    print(summary.to_string(index=False))
    fig = plot(df)
    if fig is not None:
        print("Hình:", save_figure(fig, EXP_ID))


if __name__ == "__main__":
    main()
