# Thí nghiệm

Mỗi thí nghiệm là **một file** `<mã>_<tên>.py`, ghi kết quả vào `results/<mã>/` (CSV, hình, `config.json`). Copy [`_template.py`](_template.py), đổi `EXP_ID` và `CONFIG`, rồi viết `load_data`, `run`, `plot`.

Danh sách mã thí nghiệm, dữ liệu, độ đo và mức ưu tiên lấy theo [kế hoạch toàn nhóm](../../team/KeHoach.md):

| Người | Mã | Mục kế hoạch |
| :--- | :--- | :--- |
| Hiếu | D1–D6, D9, D10 | 5.2 |
| Trường | T1–T6, T8 (ablation) | 6.2, 10 |
| Khang | K0–K6 | 7.2 |
| Thắng | S1–S6 | 8.3 |

Quy tắc (kế hoạch mục 3.2):
* Chọn tham số trên **valid**, báo kết quả trên **test**; tập test chỉ dùng một lần ở cuối.
* Seed cố định (`common.seed.set_seed`); mọi tham số nằm trong `CONFIG`, tự lưu ra `config.json` để lập bảng "tham số và lý do chọn".
* Không viết cứng đường dẫn: dùng `common.paths.DATASETS[số bộ]`.
* Chạy từ thư mục `code/`: `python experiments/k2_columns.py`.
* Đo thời gian bằng `common.experiment.timer`.
