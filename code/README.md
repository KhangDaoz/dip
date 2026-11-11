# Code — Trích xuất thông tin đơn thuốc VAIPE

Kế hoạch: [plan.md](../plan.md). Pipeline 4 khối, mỗi người một khối:

```
ảnh ─► [A] enhance/ ─► [B] detect/ ─► [C] recognize/ ─► [D] assemble/ ─► PrescriptionResult (JSON)
       Hiếu             Trường          Khang              Thắng
       tiền xử lý       YOLOv8 6 lớp    VietOCR            ghép khoản thuốc
       cổ điển          (+ cổ điển)     (+ Tesseract)      (+ VLM đối chứng)
```

Hợp đồng dữ liệu giữa các khối: [`src/common/contracts.py`](src/common/contracts.py) — box `[x0, y0, x1, y1]` (xyxy,
pixel, giống nhãn VAIPE), 6 lớp `drugname, quantity, usage, diagnose, date, other`. Đổi hợp đồng phải báo cả nhóm và
sửa test trong cùng commit.

## Cấu trúc

```
code/
├── src/
│   ├── common/      boxes (IoU, đổi định dạng, ghép tham lam), contracts, io, paths, seed, experiment
│   ├── vaipe/       đọc nhãn VAIPE 6 lớp (train 820 / valid 176 / test 176)
│   ├── enhance/     A — nắn phối cảnh, chỉnh nghiêng, khử bóng, khử nhiễu, CLAHE; preset none/geom/light/full
│   ├── detect/      B — yolo.py
│   ├── recognize/   C — vietocr_engine.py
│   ├── assemble/    D — (chưa có)
│   └── metrics/     P/R/F1 @IoU + ma trận nhầm lẫn, CER/WER, F1 khoản thuốc + bootstrap CI
├── legacy/          code pipeline v1 — KHÔNG sửa; enhance/, metrics/, tools/ và các đối chứng cổ điển import từ đây
├── tools/           to_yolo, to_vietocr, make_degraded (L0/L2/L4), download_weights
├── notebooks/       train_yolo, finetune_vietocr (Kaggle)
├── configs/         vietocr_vgg_transformer.yml (image_max_width 1024 — khớp lúc fine-tune)
├── weights/         yolo_vaipe.pt, vietocr_finetuned.pth — không commit, tải bằng tools/download_weights.py
├── samples/         ảnh mẫu cho demo
└── tests/
```

Kết quả thí nghiệm ghi vào `results/<mã>/` ở gốc repo (không commit).

## Cài đặt

Python 3.11 khuyến nghị (3.13 cũng chạy được).

```bash
cd code
python -m venv .venv
# Windows: .venv\Scripts\activate        macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CPU: .../whl/cpu
pip install -r requirements-dl.txt
pip install --no-deps -r requirements-vietocr.txt    # vietocr ghim thư viện cũ → cài không kèm phụ thuộc
pip install -e .
python tools/download_weights.py                      # trọng số từ Drive nhóm, kiểm tra SHA-256
pytest                                                # phần thiếu data / trọng số tự SKIP
```

Chỉ cần khối A và độ đo (không học sâu): `pip install -r requirements.txt && pip install -e .`.

## Dữ liệu

`data/1_vaipe/{train,valid,test}/{images,labels}` — tải từ Drive nhóm, không commit (xem [data/README.md](../data/README.md)).

```bash
python tools/to_yolo.py          # nhãn YOLO + data.yaml
python tools/to_vietocr.py       # crops + nhãn cho VietOCR
python tools/make_degraded.py    # ảnh test suy giảm L0/L2/L4 cho thí nghiệm độ bền
```

Kỷ luật đánh giá: chọn tham số trên **valid**, test chỉ chạy một lần khi đã chốt cấu hình.

## Quy ước

- Ảnh BGR `uint8`; đọc / ghi qua `common.io` (đường dẫn có dấu tiếng Việt trên Windows).
- Mask / ảnh nhị phân: mực = 255, nền = 0.
- Seed 42; mỗi thí nghiệm ghi `results/<mã>/{config.json, table.csv}`.
- Mỗi người chỉ sửa module của mình; sửa `common/` phải báo cả nhóm.
