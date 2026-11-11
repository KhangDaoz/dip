# Trích xuất thông tin đơn thuốc tiếng Việt (VAIPE)

Đồ án Xử lý ảnh số. Đầu vào là ảnh đơn thuốc, đầu ra là JSON gồm danh sách thuốc (tên, số lượng, đơn vị, cách dùng), chẩn đoán và ngày kê đơn.

```
ảnh ─► [A] enhance ─► [B] detect ─► [C] recognize ─► [D] assemble ─► JSON
       tiền xử lý     YOLOv8s       VietOCR          luật + từ điển
       cổ điển        6 lớp         (Tesseract)      ghép khoản thuốc
```

| Khối | Việc làm | Kỹ thuật (chương học phần) |
| :-: | :--- | :--- |
| A | nắn phối cảnh, chỉnh nghiêng, khử bóng, khử nhiễu, CLAHE | Canny, Hough, biến đổi hình học (ch. 2, 3, 5, 7) |
| B | phát hiện và phân loại vùng chữ: `drugname`, `quantity`, `usage`, `diagnose`, `date`, `other` | YOLOv8s; đối chứng cổ điển Sauvola + hình thái học (ch. 5, 7, 8) |
| C | nhận dạng chữ trên từng vùng | VietOCR fine-tune (chính), VietOCR gốc, Tesseract (đối chứng) |
| D | ghép vùng chữ thành từng khoản thuốc, sửa lỗi OCR, khớp từ điển tên thuốc | luật trên chuỗi + rapidfuzz; VLM zero-shot để đối chứng |

Dữ liệu: bộ VAIPE Prescription (train 820 / valid 176 / test 176 ảnh). Báo cáo và slide nằm ngoài thư mục này.

## Chạy nhanh trên ảnh mẫu

Cần Python 3.11 (3.10 đến 3.13 đều được). Thư mục `weights/` phải có `yolo_vaipe.pt` và `vietocr_finetuned.pth`.

```bash
cd code
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install torch torchvision              # GPU: thêm --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-dl.txt
pip install --no-deps -r requirements-vietocr.txt
pip install -e .

python run_pipeline.py samples/            # kết quả ở out/<tên ảnh>.json và out/<tên ảnh>_overlay.png
```

10 ảnh mẫu chạy hết khoảng 46 giây trên máy phát triển.

Nếu `weights/` trống, chạy `python tools/download_weights.py` để tải từ Drive của nhóm và kiểm tra SHA-256 (`--check` chỉ kiểm tra mã băm).

Tùy chọn của `run_pipeline.py`:

```bash
python run_pipeline.py anh.png --enhance full          # bật khối A: none | geom | light | full
python run_pipeline.py samples/ --preset classical     # đối chứng cổ điển, cần Tesseract 5 + gói vie
python run_pipeline.py samples/ --no-lexicon           # tắt khớp từ điển tên thuốc
```

Preset: `dl_c3` (mặc định, YOLO + VietOCR fine-tune), `dl_c2` (VietOCR gốc), `hybrid` (YOLO + Tesseract), `classical` (cổ điển hoàn toàn).

## Giao diện demo

```bash
python app.py              # mở http://127.0.0.1:7860
```

Tải ảnh lên để xem overlay vùng chữ, ảnh sau tiền xử lý, bảng khoản thuốc và JSON. Giao diện cho chọn preset, mức tiền xử lý và bật/tắt từ điển.

## Ảnh mẫu

`samples/` có 10 ảnh: 6 ảnh gốc từ tập test, 2 ảnh suy giảm (`rx_L2_*`, `rx_L4_*`: nghiêng, mờ, bóng, nhiễu) và 2 ảnh VAIPE khác (`rx_vaipe_*`).

## Cấu trúc

```
code/
├── run_pipeline.py   CLI
├── app.py            demo Gradio
├── src/
│   ├── common/       box, hợp đồng dữ liệu, đọc/ghi ảnh, seed
│   ├── vaipe/        đọc nhãn VAIPE, phép làm suy giảm ảnh
│   ├── enhance/      khối A
│   ├── detect/       khối B (yolo.py, classical.py)
│   ├── recognize/    khối C (vietocr_engine.py, tesseract_engine.py)
│   ├── assemble/     khối D (rules, group, lexicon, vlm)
│   ├── metrics/      P/R/F1 phát hiện, CER/WER, F1 khoản thuốc
│   └── pipeline/     ghép A → B → C → D
├── experiments/      các thí nghiệm
├── tools/            chuyển định dạng dữ liệu, tải trọng số, sinh ảnh suy giảm
├── notebooks/        huấn luyện YOLO và fine-tune VietOCR (Kaggle)
├── configs/          cấu hình VietOCR
├── weights/          trọng số (không đưa lên git)
├── samples/          ảnh mẫu
└── tests/            pytest (phần thiếu dữ liệu tự bỏ qua)
```

## Chạy lại thí nghiệm (tùy chọn)

Cần dữ liệu VAIPE tại `data/1_vaipe/{train,valid,test}/{images,labels}` (không kèm theo bài nộp; tập VAIPE công khai trên Kaggle: `tommyngx/vaipepill2022`). Chạy từ `code/`, kết quả ghi vào `results/<mã>/`.

```bash
python tools/to_yolo.py && python tools/to_vietocr.py && python tools/make_degraded.py
```

| Lệnh | Thí nghiệm |
| :--- | :--- |
| `experiments/e_a2_deskew.py`, `e_a3_runtime.py`, `fig_a_enhance.py` | A: sai số chỉnh nghiêng, thời gian chạy, hình minh họa |
| `experiments/e_b_detection.py` | B: cổ điển so với YOLO, theo từng lớp, mAP |
| `experiments/e_c1_ocr.py` | C: Tesseract so với VietOCR trên vùng cắt chuẩn |
| `experiments/e_d1.py` | D: luật ghép trên đầu vào chuẩn |
| `tools/make_handoff.py`, rồi `experiments/e_d2.py` | bảng chính đầu-cuối trên tập test |
| `experiments/e_x1_ablation.py` | ablation: thay từng khối bằng đáp án |
| `experiments/e_a1_robustness.py` | độ bền với ảnh suy giảm, có và không khối A |

Tham số chọn trên tập valid, tập test chỉ chạy một lần sau khi chốt cấu hình. Dòng VLM của `e_d2.py` cần file `code/.env` (mẫu: `.env.example`).

## Kiểm thử

```bash
pytest
```
