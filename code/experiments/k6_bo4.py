"""K6 — Dựng bảng trên phiếu xét nghiệm Việt Nam chụp thật (bộ 4): minh họa ĐỊNH TÍNH. Người làm: Khang.
Kế hoạch: team/KeHoach.md mục 7.2 (K6), 3.2 (bộ 4 không gán nhãn chi tiết — không có số liệu).

Chọn ngẫu nhiên (seed 42) 5 phiếu trong data/4_self_captured/lab_report/, chạy ① → ② → cấu trúc ③ (tham số cuối, không OCR),
vẽ lên ảnh đã nắn: dải dòng (màu xen kẽ) + ranh giới cột (vạch đỏ). Người xem tự nhận xét ca đúng / sai.
Kết quả: results/k6/<id>_k6.jpg. ẢNH CÓ THÔNG TIN BỆNH NHÂN — results/ không commit; muốn đưa vào báo cáo / slide phải
cắt về vùng thân bảng hoặc che họ tên, ngày sinh, mã bệnh nhân, số điện thoại (kế hoạch mục 3.2).

Chạy (từ code/): python experiments/k6_bo4.py
"""
import cv2
import numpy as np

from cleaning.pipeline import clean
from common.experiment import results_dir, save_config
from common.io import read_image, write_image
from common.paths import DATASETS, REPO_DIR
from common.seed import set_seed
from preprocessing.pipeline import preprocess
from table_extraction.pipeline import structure

EXP_ID = "k6"
CONFIG = {"seed": 42, "n": 5}


def draw(img, s):
    vis = img.copy()
    over = vis.copy()
    words = s["words"]
    for k, r in enumerate(s["rows"]):
        y0 = min(words[i][1] for i in r)
        y1 = max(words[i][1] + words[i][3] for i in r)
        x0 = min(words[i][0] for i in r)
        x1 = max(words[i][0] + words[i][2] for i in r)
        cv2.rectangle(over, (x0, y0), (x1, y1), (255, 200, 0) if k % 2 else (0, 220, 120), -1)
    vis = cv2.addWeighted(over, 0.35, vis, 0.65, 0)
    H = vis.shape[0]
    t = max(2, vis.shape[1] // 500)
    for a, b in s["columns"]:
        for x in (a, b):
            cv2.line(vis, (int(x), 0), (int(x), H - 1), (0, 0, 255), t)
    for x, y, w, h in words:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (80, 80, 80), 1)
    return vis


def main():
    rng = set_seed(CONFIG["seed"])
    out = results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    paths = sorted((DATASETS[4] / "lab_report").glob("*.jpg"))
    pick = [paths[i] for i in sorted(rng.choice(len(paths), CONFIG["n"], replace=False))]
    work = out / "work"
    for p in pick:
        pre = preprocess(read_image(p), p.stem, work, doc_type="lab_report")
        cl = clean(read_image(REPO_DIR / pre.rectified_path), "lab_report", p.stem, work)
        s = structure(read_image(REPO_DIR / cl.binary_path, gray=True))
        write_image(out / f"{p.stem}_k6.jpg", draw(read_image(REPO_DIR / pre.rectified_path), s), jpeg_quality=85)
        print(f"{p.stem}: góc nghiêng {pre.skew_deg:+.2f}°, h = {s['h']:.0f} px, {len(s['rows'])} dòng, "
              f"{len(s['columns'])} cột")


if __name__ == "__main__":
    main()
