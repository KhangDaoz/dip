"""T5 — An toàn: phiếu XN có giữ đủ chữ đỏ cảnh báo không? Người làm: Trường. Kế hoạch: KeHoach.md mục 6.1 (R3–R6), 6.2 (T5).

KIỂM TRA ĐỊNH TÍNH trên bộ 4 (không có nhãn chi tiết, KeHoach.md mục 3.2). Chọn bằng mắt 5 phiếu có mực đỏ thật
(chữ đỏ cảnh báo / tiêu đề đỏ / dấu đỏ) trong 48 phiếu: lab_report 003, 004, 028, 041, 048. Mỗi phiếu chạy ① (ép loại
lab_report) → ② (clean, phiếu XN KHÔNG xóa dấu). Đếm tự động để hỗ trợ nhận xét bằng mắt:
  * red_px        số pixel mask đỏ (R3, HSV)
  * stamp_cc      số thành phần bị R4 coi là DẤU (diện tích lớn + độ tròn cao) — chỉ những thành phần này bị xóa ở đơn thuốc
  * red_kept      tỉ lệ pixel chữ đỏ (red_text) còn là mực trong ảnh nhị phân sau ②
  * erased_if_rx  tỉ lệ pixel đỏ SẼ bị xóa nếu (nhầm) chạy nhánh đơn thuốc R6 = |stamp| / |red|
Kết quả: results/t5/{<id>_t5.jpg (lưới trước / sau), table.csv, config.json}. ẢNH CÓ THÔNG TIN BỆNH NHÂN — results/
không commit; hình đưa vào báo cáo chỉ cắt vùng kết quả / kết luận, không có họ tên.

Chạy (từ code/): python experiments/t5_red_text.py
"""
import cv2
import numpy as np
import pandas as pd

from cleaning.pipeline import clean
from common.experiment import results_dir, save_config, save_table
from common.io import read_image, write_image
from common.paths import DATASETS, REPO_DIR
from common.seed import set_seed
from preprocessing.pipeline import preprocess

EXP_ID = "t5"
CONFIG = {"seed": 42, "ids": ["003", "004", "028", "041", "048"], "view_long_side": 1100}


def overlay(img, red_text, stamp_mask):
    v = (0.55 * img).astype(np.uint8) + 115
    v[red_text > 0] = (0, 200, 0)        # chữ đỏ được GIỮ: xanh lá
    v[stamp_mask > 0] = (200, 0, 200)    # bị coi là dấu: tím
    return v


def fit(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    out = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return out if out.ndim == 3 else cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def main():
    set_seed(CONFIG["seed"])
    out = results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    work = out / "work"
    rows = []
    for i in CONFIG["ids"]:
        p = DATASETS[4] / "lab_report" / f"sc_lab_report_{i}.jpg"
        pre = preprocess(read_image(p), p.stem, work, doc_type="lab_report")
        rect = read_image(REPO_DIR / pre.rectified_path)
        cl = clean(rect, "lab_report", p.stem, work)
        rt = read_image(REPO_DIR / cl.masks["red_text"], gray=True)
        st = read_image(REPO_DIR / cl.masks["stamp"], gray=True)
        b = read_image(REPO_DIR / cl.binary_path, gray=True)
        red = np.maximum(rt, st) > 0
        n_red = int(red.sum())
        n_st = int(cv2.connectedComponents((st > 0).astype(np.uint8))[0] - 1)
        rows.append({"id": p.stem, "red_px": n_red, "stamp_cc": n_st,
                     "red_kept": float(((b > 0) & (rt > 0)).sum() / max((rt > 0).sum(), 1)),
                     "erased_if_rx": float((st > 0).sum() / max(n_red, 1)),
                     "components": len(cl.components)})
        s = CONFIG["view_long_side"]
        grid = np.hstack([fit(rect, s), fit(overlay(rect, rt, st), s), fit(255 - b, s)])
        write_image(out / f"{p.stem}_t5.jpg", grid, jpeg_quality=85)
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
