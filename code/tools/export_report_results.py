from __future__ import annotations

import csv
import datetime
import json
import shutil
import subprocess
import unicodedata
from pathlib import Path

from common.paths import REPO_DIR, RESULTS_DIR, vaipe_split

DEST = REPO_DIR / "report" / "latex" / "result"

FILES = [
    ("data", "s0", []),
    ("e_a2_deskew", "e_a2", ["table.csv", "error_by_angle.png", "config.json"]),
    ("e_a3_runtime", "e_a3", ["table.csv", "rectify_methods.csv", "config.json"]),
    ("fig_enhance", "fig_a", ["a_steps.png", "a_levels.png", "a_shadow.png"]),
    ("e_b_detection", "e_b", ["b0_yolo_conf_valid.csv", "b0_classical_grid_valid.csv", "b1_test.csv",
                              "b2_per_class_test.csv", "b2_confusion_test.csv", "b2_map_test.json",
                              "b2_confusion.png", "config.json"]),
    ("e_c1_ocr", "e_c1", ["table.csv", "c3_errors.png", "config.json"]),
    ("e_d1_rules", "E-D1", ["table.csv", "bad_cases.csv", "config.json"]),
    ("e_d2_endtoend", "E-D2", ["table.csv", "config.json", "prompt.txt"]),
    ("e_x1_ablation", "e_x1", ["table.csv", "config.json"]),
    ("e_a1_robustness", "e_a1", ["table.csv", "robustness.png", "config.json"]),
    ("runtime", "handoff", ["dl_c3_timing.json", "dl_c2_timing.json", "hybrid_timing.json", "classical_timing.json"]),
]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fmt(v, nd=4):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x.is_integer():
        return str(int(x)) if "." not in str(v) else f"{x:.1f}"
    return f"{x:.{nd}f}" if abs(x) < 1000 else f"{x:.0f}"


def md_table(rows: list[dict], cols: list[str] | None = None, nd: int = 4) -> str:
    if not rows:
        return "_(trống)_\n"
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join([" --- "] * len(cols)) + "|"]
    out += ["| " + " | ".join(fmt(r.get(c, ""), nd) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def data_stats() -> dict | None:
    path = vaipe_split("test") / "items_gt.json"
    if not path.exists():
        return None
    from assemble.lexicon import Lexicon
    from assemble.rules import clean_name
    vocab = {n.casefold() for n in Lexicon.from_split("train").names}
    gt = json.loads(path.read_text(encoding="utf-8"))
    names = [clean_name(it["name"]) for v in gt.values() for it in v["items"]]
    oov = [n for n in names if n.casefold() not in vocab]
    return {"n_images": len(gt), "n_reviewed": sum(bool(v.get("reviewed")) for v in gt.values()),
            "n_items": len(names), "n_names": len(set(names)), "lexicon_size": len(vocab),
            "oov_items": len(oov), "oov_items_pct": 100 * len(oov) / max(1, len(names)),
            "oov_names": len(set(oov)), "oov_names_pct": 100 * len(set(oov)) / max(1, len(set(names)))}


def c1_exact_match() -> list[dict]:
    src = RESULTS_DIR / "e_c1"
    norm = lambda s: " ".join(unicodedata.normalize("NFC", s or "").split())
    rows = read_csv(src / "table.csv")
    for r in rows:
        key = {"C0_tesseract": "C0", "C1_tesseract_sauvola": "C1", "C2_pretrained": "C2", "C3_finetune": "C3"}.get(r["engine"])
        p = src / f"preds_{key}.csv"
        if key and p.exists():
            preds = read_csv(p)
            r["exact_match"] = sum(norm(x["pred"]) == norm(x["gt"]) for x in preds) / len(preds)
    return rows


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR, text=True).strip()
    except Exception:
        commit = "?"
    md = [f"# Kết quả cho báo cáo\n\nSinh bởi `code/tools/export_report_results.py` lúc "
          f"{datetime.datetime.now():%d/%m/%Y %H:%M}, commit `{commit}`. Mọi số liệu trên **176 ảnh kiểm tra** "
          f"trừ khi ghi khác. Chạy lại script để cập nhật.\n"]
    status = {}

    for dst, src, files in FILES:
        found = [f for f in files if (RESULTS_DIR / src / f).exists()]
        if found:
            (DEST / dst).mkdir(parents=True, exist_ok=True)
            for f in found:
                shutil.copy2(RESULTS_DIR / src / f, DEST / dst / f)
        status[dst] = bool(found) or dst == "data"

    st = data_stats()
    md.append("## Dữ liệu (đáp án khoản thuốc, từ điển)\n")
    if st:
        (DEST / "data").mkdir(exist_ok=True)
        (DEST / "data" / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
        md.append(f"- Tập kiểm tra: {st['n_images']} ảnh ({st['n_reviewed']} ảnh đã soát tay), {st['n_items']} khoản thuốc, "
                  f"{st['n_names']} tên thuốc khác nhau.\n- Từ điển tên thuốc (tập huấn luyện): {st['lexicon_size']} tên.\n"
                  f"- Ngoài từ điển (OOV): {st['oov_items']} khoản ({st['oov_items_pct']:.1f} %), "
                  f"{st['oov_names']} tên ({st['oov_names_pct']:.1f} %).\n")
    else:
        md.append("_chưa có items_gt.json_\n")

    def section(title, dst, body):
        md.append(f"## {title}\n")
        md.append(body() if status.get(dst) else "_chưa có — chưa chạy thí nghiệm_\n")
        md.append(f"\nFile: `result/{dst}/`\n" if status.get(dst) else "")

    section("E-A2 — Sai số chỉnh nghiêng (khối A)", "e_a2_deskew",
            lambda: md_table(read_csv(DEST / "e_a2_deskew" / "table.csv")) +
            "\nGóc thật = ±(a + u), u ~ U[0; 0,5)°. Đơn vị sai số: độ. Hình: `error_by_angle.png`.\n")
    section("E-A3 — Thời gian khối A (giây/ảnh, 50 ảnh mỗi mức)", "e_a3_runtime",
            lambda: md_table(read_csv(DEST / "e_a3_runtime" / "table.csv"),
                             ["level"] + [c for c in read_csv(DEST / "e_a3_runtime" / "table.csv")[0] if c.endswith("_mean")]) +
            "\nPhương pháp tìm góc giấy của A1:\n\n" + md_table(read_csv(DEST / "e_a3_runtime" / "rectify_methods.csv"), nd=0))
    section("Hình minh họa khối A", "fig_enhance",
            lambda: "- `a_steps.png`: ảnh L4 qua 5 bước A1–A5\n- `a_levels.png`: một ảnh ở L0 / L2 / L4\n"
                    "- `a_shadow.png`: ảnh có bóng, nền ước lượng bằng phép đóng, kết quả chia nền\n")

    def eb():
        b1 = read_csv(DEST / "e_b_detection" / "b1_test.csv")
        b2 = read_csv(DEST / "e_b_detection" / "b2_per_class_test.csv")
        mp = json.loads((DEST / "e_b_detection" / "b2_map_test.json").read_text(encoding="utf-8"))
        for r in b2:
            r["mAP50"] = mp["mAP50_per_class"].get(r["class"], mp["mAP50"] if r["class"] == "all" else "")
        cfg = json.loads((DEST / "e_b_detection" / "config.json").read_text(encoding="utf-8"))
        return ("**E-B1 — không xét lớp, IoU ≥ 0,5**\n\n" + md_table(b1) +
                "\n**E-B2 — YOLOv8s theo lớp** (dòng `all` = cộng dồn mọi vùng)\n\n" + md_table(b2) +
                f"\nmAP50 = {mp['mAP50']:.4f}, mAP50-95 = {mp['mAP50_95']:.4f}. Ngưỡng YOLO {cfg.get('yolo_conf')}; "
                f"cổ điển chọn trên valid: {cfg.get('classical')}. Ma trận nhầm lẫn: `b2_confusion.png`.\n")
    section("E-B — Phát hiện vùng chữ (khối B)", "e_b_detection", eb)

    def ec():
        rows = c1_exact_match()
        with open(DEST / "e_c1_ocr" / "table.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return (md_table(rows, ["engine", "CER", "WER", "exact_match", "CER_drugname", "CER_quantity", "CER_usage",
                                "CER_diagnose", "CER_date", "ms_per_crop", "n"]) +
                "\nVùng cắt theo khung chuẩn. `exact_match` = tỉ lệ vùng đọc đúng nguyên chuỗi. Tesseract 8 luồng CPU; "
                "VietOCR GPU RTX 3050, batch 64. `c3_errors.png`: 3 vùng C3 đọc khác nhãn (2 ca do nhãn sai).\n")
    section("E-C1 — Nhận dạng chữ (khối C)", "e_c1_ocr", ec)

    section("E-D1 — Luật ghép khoản trên khung + chữ chuẩn (khối D)", "e_d1_rules",
            lambda: md_table(read_csv(DEST / "e_d1_rules" / "table.csv")))
    section("E-D2 — Bảng chính đầu-cuối", "e_d2_endtoend",
            lambda: md_table(read_csv(DEST / "e_d2_endtoend" / "table.csv")) +
            "\n`f1_item` có từ điển, `f1_item_nolex` không từ điển; [ci_low, ci_high] = CI 95 % bootstrap. "
            "Dòng `*_gt_text` = khung dự đoán + chữ chuẩn.\n")
    section("E-X1 — Ablation: lỗi đến từ khối nào", "e_x1_ablation",
            lambda: md_table(read_csv(DEST / "e_x1_ablation" / "table.csv")))
    def ea1():
        rows = []
        for r in read_csv(DEST / "e_a1_robustness" / "table.csv"):
            p = RESULTS_DIR / "e_a1" / f"{r['level']}_{r['enhance']}.jsonl"
            if r["level"] == "L0" and r["enhance"] == "none" and not p.exists():
                p = RESULTS_DIR / "handoff" / "dl_c3.jsonl"
            if not p.exists():
                continue
            recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            rows.append({"level": r["level"], "enhance": r["enhance"],
                         "box_per_img": sum(len(x["boxes"]) for x in recs) / len(recs),
                         "imgs_with_date_box": sum(any(b["class"] == "date" for b in x["boxes"]) for x in recs),
                         "n": len(recs)})
        with open(DEST / "e_a1_robustness" / "diagnostics.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return (md_table(read_csv(DEST / "e_a1_robustness" / "table.csv")) + "\nHình: `robustness.png`.\n\n"
                "**Chẩn đoán** (`diagnostics.csv`): số vùng YOLO tìm được mỗi ảnh và số ảnh có vùng `date`.\n\n" +
                md_table(rows, nd=1) +
                "\nKhối A làm giảm F1. Quan sát trên ảnh: phép đóng với phần tử cấu trúc vuông 81×81 (A3) để lại mảng xám "
                "hình khối và vệt sọc ở rìa vệt bóng, chữ nhạt và mảnh hơn; YOLO (huấn luyện trên ảnh gốc) nhận thêm "
                "nhiều vùng giả và bỏ sót vùng ngày. Chưa tách riêng ảnh hưởng của từng bước A1–A5.\n")
    section("E-A1 — Độ bền khi ảnh suy giảm", "e_a1_robustness", ea1)

    def rt():
        rows = []
        for f in sorted((DEST / "runtime").glob("*_timing.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"preset": f.stem.replace("_timing", ""), "n": d["n"],
                         "B_s_per_img": d.get("s_per_img_B"), "C_s_per_img": d.get("s_per_img_C")})
        return (md_table(rows) + "\nKhối B + C trên 176 ảnh test (GPU RTX 3050 cho YOLO / VietOCR, Tesseract CPU). "
                "Khối A xem E-A3; khối D ≈ 0,01–0,03 s/ảnh.\n")
    section("Thời gian khối B + C theo pipeline", "runtime", rt)

    (DEST / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    done = [k for k, v in status.items() if v]
    print(f"→ {DEST}\n  có: {done}\n  chưa có: {[k for k, v in status.items() if not v]}")


if __name__ == "__main__":
    main()
