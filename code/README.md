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
│   ├── vaipe/       đọc nhãn VAIPE 6 lớp (train 820 / valid 176 / test 176); degrade (phép làm hỏng ảnh)
│   ├── enhance/     A — nắn phối cảnh, chỉnh nghiêng, khử bóng, khử nhiễu, CLAHE (+ binarize: Sauvola, Otsu)
│   ├── detect/      B — yolo.py (YOLOv8s 6 lớp), classical.py (đối chứng: Sauvola + giãn + thành phần liên thông)
│   ├── recognize/   C — vietocr_engine.py (C2 gốc, C3 tinh chỉnh), tesseract_engine.py (C0 xám, C1 Sauvola)
│   ├── assemble/    D — rules, group (ghép khoản), lexicon (từ điển), vlm (đối chứng), load, scoring
│   ├── metrics/     P/R/F1 @IoU + ma trận nhầm lẫn, CER/WER, F1 khoản thuốc + bootstrap CI
│   └── pipeline/    ghép A → B → C → D, 4 preset: classical, hybrid, dl_c2, dl_c3 (chính)
├── run_pipeline.py  CLI: ảnh / thư mục → out/<ảnh>.json + out/<ảnh>_overlay.png
├── experiments/     thí nghiệm E-* (mục "Thí nghiệm" bên dưới)
├── tools/           to_yolo, to_vietocr, make_degraded (L0/L2/L4), download_weights, build_items_gt, make_handoff
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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # GPU (driver CUDA ≥ 13); CPU: .../whl/cpu
pip install -r requirements-dl.txt
pip install --no-deps -r requirements-vietocr.txt    # vietocr ghim thư viện cũ → cài không kèm phụ thuộc
pip install -e .
python tools/download_weights.py                      # trọng số từ Drive nhóm, kiểm tra SHA-256
pytest                                                # phần thiếu data / trọng số tự SKIP
```

Chỉ cần khối A và độ đo (không học sâu): `pip install -r requirements.txt && pip install -e .`.

## Giao diện demo (Gradio)

```bash
python app.py            # http://127.0.0.1:7860 — tải ảnh lên, chạy A → B → C → D
python app.py --share    # link công khai tạm thời
```

Chọn được preset (dl_c3 / dl_c2 / hybrid / classical), mức tiền xử lý khối A và bật/tắt từ điển. Kết quả: overlay,
ảnh sau khối A, bảng khoản thuốc, bảng vùng chữ B + C, JSON tải về.

## Dữ liệu

`data/1_vaipe/{train,valid,test}/{images,labels}` — tải từ Drive nhóm, không commit (xem [data/README.md](../data/README.md)).

```bash
python tools/to_yolo.py          # nhãn YOLO + data.yaml
python tools/to_vietocr.py       # crops + nhãn cho VietOCR
python tools/make_degraded.py    # ảnh test suy giảm L0/L2/L4 cho thí nghiệm độ bền
```

Kỷ luật đánh giá: chọn tham số trên **valid**, test chỉ chạy một lần khi đã chốt cấu hình.

## Chạy pipeline

```bash
python run_pipeline.py samples/                       # dl_c3: YOLOv8s + VietOCR tinh chỉnh + khối D
python run_pipeline.py samples/ --preset classical    # đối chứng cổ điển (cần Tesseract 5 + vie)
python run_pipeline.py anh.png --enhance full         # bật tiền xử lý khối A
```

## Thí nghiệm

Chạy từ `code/`, kết quả ở `results/<mã>/`. Đáp án khoản thuốc cần file đáp án v1 đã soát tay (`results/s0/gt_items_<split>.json`, Drive nhóm).

| Thứ tự | Lệnh | Thí nghiệm | Cần GPU |
| :-: | :--- | :--- | :-: |
| 1 | `python tools/make_degraded.py` · `python tools/build_items_gt.py --split test` | ảnh suy giảm, đáp án khoản thuốc | |
| 2 | `python experiments/e_a2_deskew.py` · `e_a3_runtime.py` · `fig_a_enhance.py` | A: sai số chỉnh nghiêng, thời gian, hình minh họa | |
| 3 | `python experiments/e_b_detection.py` | B: chọn tham số trên valid; cổ điển vs YOLO; YOLO theo lớp + mAP | ✓ |
| 4 | `python experiments/e_c1_ocr.py` | C: C0/C1/C2/C3 trên vùng cắt chuẩn | ✓ |
| 5 | `python experiments/e_d1.py` | D: luật ghép trên khung + chữ chuẩn | |
| 6 | `python tools/make_handoff.py --preset dl_c3 dl_c2 hybrid classical` | file bàn giao B+C cho E-D2 | ✓ |
| 7 | `python experiments/e_d2.py --variant dl_c3=../results/handoff/dl_c3.jsonl ... --check dl_c3 [--vlm]` | **bảng chính** đầu-cuối (VLM cần `code/.env`) | |
| 8 | `python experiments/e_x1_ablation.py` | ablation: thay B / C bằng đáp án | ✓ |
| 9 | `python experiments/e_a1_robustness.py` | độ bền L0/L2/L4 × có / không khối A | ✓ |

## Quy ước

- Ảnh BGR `uint8`; đọc / ghi qua `common.io` (đường dẫn có dấu tiếng Việt trên Windows).
- Mask / ảnh nhị phân: mực = 255, nền = 0.
- Seed 42; mỗi thí nghiệm ghi `results/<mã>/{config.json, table.csv}`.
- Mỗi người chỉ sửa module của mình; sửa `common/` phải báo cả nhóm.
