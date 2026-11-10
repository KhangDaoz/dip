"""K0 — Ground truth dòng / cột cho bộ 2 (57 ảnh gốc). Người làm: Khang. Kế hoạch: KeHoach.md mục 7.2, 7.4.

Nhãn YOLO của bộ 2: lớp + tâm / kích thước chuẩn hóa. Bỏ lớp `Report Type` (không thuộc thân bảng).
Dòng GT: hai cụm từ cùng dòng nếu chồng nhau theo chiều dọc ≥ 50 % chiều cao cụm thấp hơn; nối bằng union-find
(độc lập với cả hai thuật toán gom dòng, tránh lỗi của pilot cũ — KeHoach.md mục 7.4).
Cột GT: vai trò cụm từ (parameter, value, unit, reference value).
Text GT (cho K5 — CER): bộ 2 không có nhãn chữ. Chữ trong từng hộp do LLM (Claude) phiên âm từ ẢNH GỐC màu,
lưu ở experiments/k0_text_gt.json (ảnh → {chỉ số dòng trong file nhãn YOLO: text}); null = không đọc được;
"uncertain" = ô LLM đọc không chắc (soát trước); "bad_labels" = hộp nhãn gốc lệch khỏi chữ, toàn bộ text = null.
Text là nhãn ĐỀ XUẤT, người phải soát (KeHoach.md mục 7.4) — sửa thẳng vào k0_text_gt.json rồi chạy lại.
Soát tay (experiments/k0_review.json): "keep" = ảnh đủ cơ sở làm GT sau khi soát bằng mắt; "fixes" sửa dòng
theo quy tắc ghi trong file — "merge" [a, b] đưa hộp a vào dòng của hộp b (tên 2 dòng, ảnh nghiêng),
"exclude" bỏ hộp khỏi GT dòng / cột (dòng mô tả như Peripheral Smear, tiêu đề nhóm không có kết quả).
Kết quả: results/k0/gt.json = GT CHÍNH THỨC, chỉ ảnh "keep", đã áp "fixes" (ảnh → words [x, y, w, h], role,
row, lid, text, text_uncertain; hộp bị exclude chuyển sang "excluded_words"); results/k0/gt_all.json = cả 55 ảnh
(đã áp fixes nếu có, kèm cờ "keep"); results/k0/review/*.jpg để SOÁT BẰNG MẮT: trái = hộp tô màu theo vai trò +
[số dòng GT], phải = text theo dòng GT, dạng "lid: text" (nền vàng + "?" = không chắc, xám ∅ = null,
cuối bảng = hộp bị exclude); ảnh không thuộc "keep" có tiền tố "x_".

Chạy (từ code/): python experiments/k0_gt.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from common.experiment import results_dir, save_config
from common.io import read_image, save_json, write_image
from common.paths import DATASETS
from common.seed import set_seed

EXP_ID = "k0"
NAMES = ["Report Type", "parameter", "reference value", "unit", "value"]
ROLE = {"parameter": "parameter", "value": "result", "unit": "unit", "reference value": "range"}
CONFIG = {"seed": 42, "row_overlap": 0.5, "splits": ["train", "valid", "test"],
          "text_gt": "experiments/k0_text_gt.json", "text_source": "LLM (Claude) phiên âm từ ảnh gốc, người soát"}
COLORS = {"parameter": (200, 80, 0), "result": (0, 0, 220), "unit": (0, 150, 0), "range": (160, 0, 160)}
TEXT_GT = Path(__file__).resolve().parent / "k0_text_gt.json"
REVIEW = Path(__file__).resolve().parent / "k0_review.json"


def load_words(img_path, lab_path):
    img = read_image(img_path)
    H, W = img.shape[:2]
    words = []
    for lid, line in enumerate(open(lab_path, encoding="utf-8")):
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        if NAMES[c] == "Report Type":
            continue
        words.append({"bbox": [int(round((cx - w / 2) * W)), int(round((cy - h / 2) * H)),
                               int(round(w * W)), int(round(h * H))], "role": ROLE[NAMES[c]],
                      "lid": lid})   # chỉ số dòng trong file nhãn = khóa nối với text GT
    return img, words


def review_image(img, words, text_entry, excluded=()):
    """Ảnh soát: trái = hộp (màu theo vai trò) + số dòng GT; phải = text từng dòng GT."""
    S = 2
    vis = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    for w in excluded:   # hộp bị bỏ khỏi GT dòng / cột: xám
        x, y, ww, hh = (v * S for v in w["bbox"])
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), (160, 160, 160), 1)
    for w in words:
        x, y, ww, hh = (v * S for v in w["bbox"])
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), COLORS[w["role"]], 1)
        cv2.putText(vis, f"{w['row']}", (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[w["role"]], 1)
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
    unc = set(text_entry.get("uncertain", [])) if text_entry else set()
    lines = []
    if text_entry and text_entry.get("bad_labels"):
        lines.append([("!! HỘP NHÃN GỐC LỆCH KHỎI CHỮ — text = null, nên loại ảnh khỏi K5", (200, 0, 0))])
    for r in range(max(w["row"] for w in words) + 1):
        seg = [(f"[{r}]", (0, 0, 0))]
        for w in sorted((w for w in words if w["row"] == r), key=lambda w: w["bbox"][0]):
            t = w.get("text")
            col = (150, 150, 150) if t is None else tuple(COLORS[w["role"]][::-1])
            mark = "?" if w["lid"] in unc else ""
            seg.append((f"{mark}{w['lid']}: {'∅' if t is None else t}", col))
        lines.append(seg)
    if excluded:
        lines.append([("bỏ khỏi GT dòng/cột:", (120, 120, 120))] +
                     [(f"{w['lid']}: {w.get('text')}", (150, 150, 150)) for w in excluded])
    panel_w, lh = 900, 18
    # xuống dòng khi một dòng GT dài hơn bảng
    rows_px, cur = [], []
    for seg in lines:
        x = 0
        for txt, col in seg:
            tw = font.getlength(txt + "   ")
            if x and x + tw > panel_w - 10:
                rows_px.append(cur); cur, x = [], 30
            cur.append((x, txt, col)); x += tw
        rows_px.append(cur); cur = []
    panel = Image.new("RGB", (panel_w, max(vis.shape[0], lh * len(rows_px) + 10)), "white")
    d = ImageDraw.Draw(panel)
    for i, row in enumerate(rows_px):
        for x, txt, col in row:
            if txt.startswith("?"):   # ô LLM đọc không chắc: nền vàng để soát trước
                d.rectangle([5 + x - 2, 5 + i * lh, 5 + x + font.getlength(txt) + 2, 5 + i * lh + lh - 2],
                            fill=(255, 235, 0))
            d.text((5 + x, 5 + i * lh), txt, font=font, fill=col)
    right = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    left = np.full((right.shape[0], vis.shape[1], 3), 255, np.uint8)
    left[:vis.shape[0]] = vis
    return np.hstack([left, right])


def gt_rows(words, ratio):
    n = len(words)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        yi, hi = words[i]["bbox"][1], words[i]["bbox"][3]
        for j in range(i + 1, n):
            yj, hj = words[j]["bbox"][1], words[j]["bbox"][3]
            ov = min(yi + hi, yj + hj) - max(yi, yj)
            if ov >= ratio * min(hi, hj):
                parent[find(i)] = find(j)
    roots = sorted({find(i) for i in range(n)}, key=lambda r: min(words[i]["bbox"][1] for i in range(n) if find(i) == r))
    rid = {r: k for k, r in enumerate(roots)}
    return [rid[find(i)] for i in range(n)]


def apply_fixes(words, fix):
    """Áp sửa tay của k0_review.json: merge [a, b] → a vào dòng của b; exclude → tách ra. Đánh lại số dòng theo y."""
    by_lid = {w["lid"]: w for w in words}
    for a, b in fix.get("merge", []):
        by_lid[a]["row"] = by_lid[b]["row"]
    drop = set(fix.get("exclude", []))
    missing = (drop | {x for m in fix.get("merge", []) for x in m}) - set(by_lid)
    if missing:
        raise ValueError(f"k0_review.json nhắc tới lid không có trong nhãn: {sorted(missing)}")
    kept = [w for w in words if w["lid"] not in drop]
    excluded = [w for w in words if w["lid"] in drop]
    order = sorted({w["row"] for w in kept},
                   key=lambda r: np.median([w["bbox"][1] + w["bbox"][3] / 2 for w in kept if w["row"] == r]))
    rid = {r: k for k, r in enumerate(order)}
    for w in kept:
        w["row"] = rid[w["row"]]
    for w in excluded:
        w.pop("row", None)
    return kept, excluded


def main():
    set_seed(CONFIG["seed"])
    out = results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    gt = {}
    texts = json.load(open(TEXT_GT, encoding="utf-8")) if TEXT_GT.exists() else {}
    review = json.load(open(REVIEW, encoding="utf-8")) if REVIEW.exists() else {"keep": None, "fixes": {}}
    keep = set(review["keep"]) if review["keep"] is not None else None
    for p in (out / "review").glob("*.jpg"):   # tên ảnh soát phụ thuộc keep → xóa bản cũ
        p.unlink()
    for split in CONFIG["splits"]:
        d = DATASETS[2] / split
        for img_path in sorted((d / "images").glob("*")):
            lab = d / "labels" / f"{img_path.stem}.txt"
            if not lab.exists():
                continue
            img, words = load_words(img_path, lab)
            if not words:
                continue
            rows = gt_rows(words, CONFIG["row_overlap"])
            for w, r in zip(words, rows):
                w["row"] = r
            key = f"{split}/{img_path.name}"
            te = texts.get(key)
            unc = set(te["uncertain"]) if te else set()
            for w in words:
                w["text"] = te["text"].get(str(w["lid"])) if te else None
                w["text_uncertain"] = w["lid"] in unc
            fix = review["fixes"].get(key, {})
            words, excluded = apply_fixes(words, fix)
            is_keep = keep is None or key in keep
            gt[key] = {"split": split, "image": str(img_path.relative_to(DATASETS[2])), "words": words,
                       "excluded_words": excluded, "n_rows": max(w["row"] for w in words) + 1,
                       "text_bad_labels": bool(te and te["bad_labels"]), "keep": is_keep,
                       "review_note": fix.get("note")}
            # đánh số thứ tự (giữ nguyên số đã dùng khi soát): cắt tên còn 40 ký tự từng làm 3 ảnh trùng tên
            prefix = "" if is_keep else "x_"
            write_image(out / "review" / f"{prefix}{len(gt):02d}_{split}_{img_path.stem[:40]}.jpg",
                        review_image(img, words, te, excluded), jpeg_quality=85)
    save_json(out / "gt_all.json", gt)
    official = {k: v for k, v in gt.items() if v["keep"]}
    save_json(out / "gt.json", official)
    for name, d in (("gt_all", gt), ("gt (chính thức)", official)):
        ws = [w for g in d.values() for w in g["words"]]
        print(f"{name}: {len(d)} ảnh · {len(ws)} cụm từ · {sum(g['n_rows'] for g in d.values())} dòng · "
              f"{sum(len(g['excluded_words']) for g in d.values())} hộp bị exclude · text: "
              f"{sum(w['text'] is not None for w in ws)} có chữ, {sum(w['text_uncertain'] for w in ws)} không chắc")


if __name__ == "__main__":
    main()
