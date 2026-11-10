"""S5 — Bóc tách khoản thuốc trên đơn thật chụp bằng điện thoại (bộ 4): minh họa ĐỊNH TÍNH. Người làm: Thắng.
Kế hoạch: KeHoach.md mục 8.3 (S5), 3.2 (bộ 4 không gán nhãn chi tiết — không có số liệu).

Chọn ngẫu nhiên (seed 42) 5 đơn trong data/4_self_captured/prescription/, chạy ① (ép loại prescription) → ② (xóa dấu /
bút) → ④ (tham số cuối, không OCR), vẽ lên ảnh đã nắn: vùng thuốc (khung xanh dương), dòng (vạch xám), khoản thuốc
(mỗi khoản một màu). Người xem nhận xét ca đúng / sai.
Kết quả: results/s5/<id>_s5.jpg, summary.csv. ẢNH CÓ THÔNG TIN BỆNH NHÂN — results/ không commit.

Chạy (từ code/): python experiments/s5_bo4.py
"""
import cv2
import pandas as pd

from cleaning.pipeline import clean
from common.experiment import results_dir, save_config, save_table
from common.io import read_image, write_image
from common.paths import DATASETS, REPO_DIR
from common.seed import set_seed
from preprocessing.pipeline import preprocess
from prescription_segmentation.pipeline import segment

EXP_ID = "s5"
CONFIG = {"seed": 42, "n": 5}
COLORS = [(0, 170, 0), (0, 120, 255), (200, 0, 200), (0, 200, 200), (255, 120, 0), (120, 60, 0)]


def draw(img, seg):
    vis = img.copy()
    t = max(2, img.shape[1] // 400)
    for x, y, w, h in seg["lines"]:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (150, 150, 150), 1)
    over = vis.copy()
    for k, (x, y, w, h) in enumerate(seg["items"]):
        cv2.rectangle(over, (x, y), (x + w, y + h), COLORS[k % len(COLORS)], -1)
    vis = cv2.addWeighted(over, 0.3, vis, 0.7, 0)
    for k, (x, y, w, h) in enumerate(seg["items"]):
        cv2.rectangle(vis, (x, y), (x + w, y + h), COLORS[k % len(COLORS)], t)
    x, y, w, h = seg["region"]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), t)
    return vis


def main():
    rng = set_seed(CONFIG["seed"])
    out = results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    paths = sorted((DATASETS[4] / "prescription").glob("*.jpg"))
    pick = [paths[i] for i in sorted(rng.choice(len(paths), CONFIG["n"], replace=False))]
    work = out / "work"
    rows = []
    for p in pick:
        pre = preprocess(read_image(p), p.stem, work, doc_type="prescription")
        rect = read_image(REPO_DIR / pre.rectified_path)
        cl = clean(rect, "prescription", p.stem, work)
        seg = segment(read_image(REPO_DIR / cl.binary_path, gray=True))
        write_image(out / f"{p.stem}_s5.jpg", draw(rect, seg), jpeg_quality=85)
        rows.append({"id": p.stem, "corner_method": pre.corner_method, "skew_deg": pre.skew_deg,
                     "h": round(seg["h"], 1), "lines": len(seg["lines"]), "items": len(seg["items"])})
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID, "summary")
    print(df.to_string())


if __name__ == "__main__":
    main()
