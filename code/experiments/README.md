# Thí nghiệm v2 (plan.md §5, §6) — chạy từ `code/`

Chuẩn bị: data VAIPE ở `data/1_vaipe/`, trọng số ở `code/weights/` (xem `weights/README.md`), rồi:

```bash
python tools/build_items_gt.py --split all       # đáp án khoản thuốc (bắt buộc cho E-D/E-X/E-A1)
python tools/make_degraded.py                    # ảnh L0/L2/L4 cho E-A1
```

| Thứ tự | Lệnh | Người | Ra | Thời gian ước tính |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `python experiments/e_d_items.py --pipelines dl_c2 --limit 10` | ai cũng được | thử nhanh, kiểm tra trọng số | 2' |
| 2 | `python experiments/e_b_detection.py` | Trường | `results/e_b/` E-B0 (chọn conf trên valid), E-B1, E-B2 + ma trận nhầm lẫn | 10–20' |
| 3 | `python experiments/e_c1_ocr.py --limit 3000` | Khang | `results/e_c1/table.csv` C0–C3 | 20–40' (Tesseract chậm) |
| 4 | `python experiments/e_d_items.py` | Thắng | `results/e_d1/`, **`results/e_d2/table.csv` (bảng chính)** | 30–60' |
| 5 | `python experiments/e_x1_ablation.py --ocr C3` | Hiếu | `results/e_x1/table.csv` | 15' (tái dùng không được cache của E-D2) |
| 6 | `python experiments/e_a1_robustness.py --ocr C3` | Hiếu | `results/e_a1/` bảng + `robustness.png` | 30–60' |
| 7 | `python experiments/e_a2_deskew.py` | Hiếu | `results/e_a2/table.csv` | 5' |

Dự đoán từng ảnh được cache (`results/<mã>/<biến thể>/<id>.json`): chạy lại chỉ tính phần thiếu; đổi cấu hình thì
xóa thư mục cache tương ứng. Chưa có C3 → thay `--ocr C2` / `--pipelines dl_c2` (plan §8.4).
Kỷ luật: chọn tham số (conf YOLO, τ từ điển) trên **valid**; test chạy một lần với cấu hình đã chốt.

---

# (Cũ, v1 — không còn chạy được vì import module đã chuyển sang `legacy/`)

# Thí nghiệm

Mỗi thí nghiệm là **một file** `<mã>_<tên>.py`, ghi kết quả vào `results/<mã>/` (CSV, hình, `config.json`). Copy [`_template.py`](_template.py), đổi `EXP_ID` và `CONFIG`, rồi viết `load_data`, `run`, `plot`.

Danh sách mã thí nghiệm, dữ liệu, độ đo và mức ưu tiên lấy theo [kế hoạch toàn nhóm](../../team/KeHoach.md):

| Người | Mã | Mục kế hoạch |
| :--- | :--- | :--- |
| Hiếu | D1–D6, D9, D10 | 5.2 |
| Trường | T1–T6, T8 (ablation) | 6.2, 10 |
| Khang | K0–K6 | 7.2 |
| Thắng | S1–S6 | 8.3 |

Phần ③ (Khang) — chạy theo thứ tự, từ `code/`:

| Script | Làm gì | Đọc | Ghi |
| :--- | :--- | :--- | :--- |
| `k0_gt.py` | GT dòng / cột / chữ bộ 2; áp quyết định soát tay | nhãn bộ 2, `k0_text_gt.json`, `k0_review.json` | `results/k0/gt.json` (32 ảnh chính thức), `gt_all.json`, `review/` |
| `k2_table.py` | K1 gom dòng, K2 tìm cột + tách cụm từ, K4 $h$ thích nghi — 5-fold | `results/k0/gt.json` | `results/k2/` |
| `k3_skew.py` | K3 gom dòng khi ảnh nghiêng thêm 0→5°, có / không chỉnh nghiêng ① | như trên | `results/k3/` |
| `k5_ocr.py` | K5 OCR theo ô vs cả trang (CER) — cần Tesseract 5 | như trên | `results/k5/` |

Sửa GT: chữ sai → `k0_text_gt.json`; dòng sai / ảnh loại → `k0_review.json`; rồi chạy lại `k0_gt.py` và các script sau nó.

Quy tắc (kế hoạch mục 3.2):
* Chọn tham số trên **valid**, báo kết quả trên **test**; tập test chỉ dùng một lần ở cuối.
* Seed cố định (`common.seed.set_seed`); mọi tham số nằm trong `CONFIG`, tự lưu ra `config.json` để lập bảng "tham số và lý do chọn".
* Không viết cứng đường dẫn: dùng `common.paths.DATASETS[số bộ]`.
* Chạy từ thư mục `code/`: `python experiments/k2_columns.py`.
* Đo thời gian bằng `common.experiment.timer`.
