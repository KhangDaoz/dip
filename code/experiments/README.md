# Thí nghiệm

Mỗi thí nghiệm = **một file** `<mã>_<tên>.py`, ghi kết quả vào `results/<mã>/` (CSV + PNG + config.json).
Copy một file mẫu bên dưới để tạo thí nghiệm mới, đổi `EXP_ID`, `CONFIG`, rồi viết `load_data` / `run` / `plot`.

| File mẫu | Người | Ghi chú |
| :--- | :--- | :--- |
| `d1_iqa_roc.py` | Hiếu | D1 → D10 |
| `t1_shadow.py` | Trường | T1 → T8 |
| `t2_binarization.py` | Trường | T2: 5 cách nhị phân hóa × hai nhánh, chọn trên val; xem [t2_binarization.md](t2_binarization.md) |
| `k1_rows.py` | Khang | K1 → K7 |
| `k0_dataset3_ground_truth.py` | Khang | K0: sinh ground truth phrase-level dòng/cột từ nhãn bộ 3; mặc định lấy 20 nhóm ảnh nguồn để review |
| `k0_ocr_benchmark.py` | Khang | So sánh CER/WER và tốc độ Tesseract/PaddleOCR trên 50 crop đã có transcript người dùng xác nhận |
| `s1_items.py` | Thắng | S1 → S6 |

Quy tắc:
* Seed cố định (`CONFIG["seed"]`), mọi tham số nằm trong `CONFIG` → tự lưu ra `config.json` (dùng cho bảng "tham số & lý do chọn").
* Không viết cứng đường dẫn: dùng `common.paths.DATASETS[số bộ]`.
* Chạy từ thư mục `code/`: `python experiments/d1_iqa_roc.py`.
* Thời gian chạy (D9, T7, K7, S6): dùng `common.experiment.timer`.

T1 khử bóng đã có script chạy đầy đủ; xem [t1_shadow.md](t1_shadow.md) để biết cách chọn tham số trên val, đánh giá test và trỏ tới dataset đã tải bằng `--data-dir`.

## Benchmark OCR bộ 3

Dataset `data/dataset3_50/` có 50 crop độc lập, transcript `text_gt` đã xác nhận trong manifest. Chạy từ `code/`:

```bash
python experiments/k0_ocr_benchmark.py --dataset ../data/dataset3_50 \
  --out ../results/ocr_benchmark_dataset3_50_line --engines tesseract paddle
```

Script từ chối mẫu chưa được người dùng xác nhận, ghi dự đoán từng crop vào `predictions.csv` và tổng hợp CER theo toàn corpus, WER, độ tin cậy, độ trễ và CER theo loại vùng vào `summary.json`. Mặc định dùng `ocr_line` (Tesseract PSM 7); đo riêng một lượt warm-up rồi lấy độ trễ trên 50 mẫu. PaddleOCR/VietOCR cache instance model trong tiến trình; không cache kết quả theo từng crop.

T1–T2 hiện dùng bộ ⑥ v2 tại `data/6_synthetic/`; xem lệnh v2 trong hai hướng dẫn trên. Kết quả mới: `results/t1_v2/`, `results/t2_v2/`. Phép đo sinh bóng đơn từ clean v2 theo seed cố định, giữ split nguồn vì mẫu bóng đơn có sẵn không phủ test; kết quả v1 được lưu riêng để truy vết.

Sau bản sửa lóa `fed83ebc`, T1–T2 đã lưu được tái kiểm chứng bằng các cấu hình khóa, không thay config/số liệu lịch sử. Khi dùng T1 lịch sử, truyền `--compatibility-evidence` theo hướng dẫn T2; khi chạy cặp mới, T1 lưu contract đầu vào và T2 kiểm contract đó. Hướng dẫn bàn giao hiện tại: `results/t1_t2_repro_audit/HANDOFF_V2_1.md`. Gói v2.1 kèm ZIP gốc làm bằng chứng, không kèm dataset; không có cờ bỏ qua checksum.
