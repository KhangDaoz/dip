# Code — Pipeline xử lý ảnh MedTrack

Khung code (skeleton) cho đồ án Xử lý ảnh số. Mọi hàm đã có **tên, tham số mặc định và docstring theo [kế hoạch toàn nhóm](../team/KeHoach.md)**; thân hàm là `raise NotImplementedError`. **Mỗi người chỉ sửa module của mình.** Ai muốn đổi `common/` hoặc hợp đồng dữ liệu thì phải báo cả nhóm.

```
① Hiếu ──PreprocessResult──► ② Trường ──CleaningResult──► ③ Khang  (phiếu XN)  ──LabReportResult──►
 preprocessing/                cleaning/                     table_extraction/
 classification/                                          ④ Thắng (đơn thuốc) ──PrescriptionResult─►
                                                             prescription_segmentation/
                  Dùng chung: common/ · ocr/ (công cụ đo CER) · synthetic/ = bộ 5 (Thắng)
```

## 1. Cài đặt

Yêu cầu Python **3.10 → 3.13**. Riêng phần OCR nên dùng **3.11** (xem mục 6).

```bash
cd code
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # import được preprocessing, cleaning, ... từ bất kỳ đâu
pytest                    # phải XANH: test của hàm chưa viết hiện SKIPPED
```

## 2. Ai làm gì

Cột "Bước" là mã bước trong kế hoạch toàn nhóm; docstring mỗi hàm ghi bước và slide bài giảng tương ứng.

| Module | Người | File | Bước (kế hoạch) | Thí nghiệm | Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `preprocessing/` | Hiếu | `quality`, `denoise`, `glare`, `corners`, `rectify`, `deskew`, `enhance`, `pipeline` | H1–H9 (mục 5.1) | D1–D6, D9 | `test_preprocessing.py` |
| `classification/` | Hiếu | `features`, `classify` | H10–H11 | D10 | `test_classification.py` |
| `cleaning/` | Trường | `background`, `ink`, `stamp`, `inpaint`, `binarize`, `postprocess`, `components`, `metrics`, `pipeline` | R1–R9 (mục 6.1) | T1–T6, T8 | `test_cleaning.py` |
| `table_extraction/` | Khang | `geometry`, `rows`, `columns`, `cells`, `postprocess`, `flags`, `grid` (mở rộng), `pipeline` | K1–K9 (mục 7.1) | K0–K6 | `test_table.py` |
| `prescription_segmentation/` | Thắng | `region`, `lines`, `items`, `fields`, `pipeline` | S-B0–S-B5 (mục 8.1) | S1–S6 | `test_prescription.py` |
| `synthetic/` *(bộ 5)* | Thắng | `degrade`, `generate`, `config.yaml` | mục 8.2 | — | `test_synthetic.py` |
| `ocr/` *(công cụ)* | Khang | `engines`, `metrics` | — | — | `test_ocr.py` |
| `common/` | Cả nhóm | `contracts`, `io`, `paths`, `seed`, `experiment`, `testing` | — | — | `test_contracts.py` |

Cách làm một hàm:

1. Thay `raise NotImplementedError(...)` bằng code thật.
2. Chạy `pytest tests/test_<module>.py`: test của hàm đó chuyển từ SKIPPED sang PASSED (hoặc FAILED nếu sai).
3. Thêm hàm, tham số thoải mái. **Không đổi tên hoặc kiểu trả về** của hàm người khác gọi: `*/pipeline.py`, `ocr/*`, `synthetic/degrade.py`.

## 3. Quy ước bắt buộc

| Hạng mục | Quy ước |
| :--- | :--- |
| Ảnh màu | `uint8`, shape `(H, W, 3)`, kênh **BGR** |
| Ảnh nhị phân / mask | `uint8`, `(H, W)`, **mực / vùng quan tâm = 255, nền = 0** |
| bbox | `[x, y, w, h]`, số nguyên, gốc ở góc trên-trái |
| Tọa độ của ②③④ | Tính trên **ảnh đã nắn** của ①; chiếu về ảnh gốc bằng `PreprocessResult.to_source()` |
| 4 góc | Thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái |
| Kích thước SE, ngưỡng | Theo **chiều cao chữ $h$** của từng ảnh, không theo pixel cố định (kế hoạch mục 4.1, C2 · S26) |
| Đọc / ghi ảnh | `common.io.read_image` / `write_image` (`cv2.imread` không đọc được đường dẫn có dấu trên Windows) |
| Đường dẫn | `common.paths` (`DATA_DIR`, `DATASETS[1..5]`, `RESULTS_DIR`, `OUT_DIR`), không viết cứng |
| Seed | `common.seed.set_seed(42)` ở đầu mọi thí nghiệm |
| CER / WER | `ocr.metrics`; báo cáo cả tập dùng `corpus_cer` (tổng lỗi / tổng ký tự) |

## 4. Hợp đồng dữ liệu — `src/common/contracts.py`

Mỗi mắt xích trả về một `dataclass` có `.to_dict()`, `.from_dict()`, `.validate()`. Ví dụ JSON đầy đủ ở `tests/test_contracts.py`.

| Hợp đồng | Người gửi → nhận | Trường bắt buộc |
| :--- | :--- | :--- |
| `PreprocessResult` | Hiếu → Trường | `rectified_path` (BGR đã nắn + chỉnh nghiêng, **chưa CLAHE**), `homography` (ảnh gốc → ảnh đã nắn, gồm cả xoay), `corners_src`, `corner_method`, `skew_deg`, `doc_type`, `doc_confidence`, `quality` |
| `CleaningResult` | Trường → Khang, Thắng | `clean_gray_path`, `binary_path`, `masks` (`stamp`, `handwriting`, `red_text`), `components[]` (`bbox`, `label`, `confidence`), `params` |
| `LabReportResult` | Khang → app | `report_type`, `rows[]` (`bbox`, `parameter`, `result`, `unit`, `range`, `flag`, ...) |
| `PrescriptionResult` | Thắng → app | `patient`, `diagnosis`, `items[]` (`bbox`, `name`, `strength`, `dose`, `days`, ...) |

Giá trị hợp lệ: `doc_type` ∈ {`prescription`, `lab_report`, `other`}; `corner_method` ∈ {`contour`, `hough`, `minrect`, `full_image`}; `label` ∈ {`printed`, `handwritten`, `stamp`, `other`} (gán bằng **luật**, không dùng ML); `flag` ∈ {`LOW`, `NORMAL`, `HIGH`, `UNKNOWN`}.

**Muốn đổi hợp đồng:** báo nhóm, sửa `contracts.py` và `tests/test_contracts.py` **trong cùng một commit**.

## 5. Chạy

```bash
pytest                                                 # toàn bộ test
pytest tests/test_table.py -v                          # test một module
python run_pipeline.py samples/                        # đầu-cuối ① → ② → ③/④
python run_pipeline.py anh.jpg --doc-type lab_report   # ép loại tài liệu khi bộ phân loại chưa xong
python experiments/k2_columns.py                       # một thí nghiệm → results/k2/
python -m synthetic.generate                           # sinh bộ 5 → data/5_synthetic/
```

`run_pipeline.py` in trạng thái từng bước của từng ảnh (`ok` / `chưa làm` / `LỖI`), JSON ghi vào `out/`. Thí nghiệm mới: copy [`experiments/_template.py`](experiments/_template.py), xem [experiments/README.md](experiments/README.md).

## 6. Thư viện thêm

| File | Khi nào cài | Ghi chú |
| :--- | :--- | :--- |
| `requirements.txt` | Ngay | OpenCV, scikit-image, scikit-learn, pandas, matplotlib, pytest |
| `requirements-ocr.txt` | Khi đo CER | Tesseract 5 **cài riêng** ([bản Windows](https://github.com/UB-Mannheim/tesseract/wiki)), thêm gói ngôn ngữ `vie`. PaddleOCR / VietOCR có thể chưa hỗ trợ Python 3.13 → dùng venv 3.11 |

## 7. Dữ liệu và kết quả (không commit)

> [!IMPORTANT]
> **Dataset của nhóm:** [Google Drive MedTrack Dataset](https://drive.google.com/drive/folders/1PtO0ret01AW0mF504ayRFE7GoE2Ez4_G?usp=drive_link). Tải về và giải nén vào `data/` ở gốc repo đúng cấu trúc dưới đây; không tự tải nguồn lẻ.

`data/`, `results/`, `out/` nằm ở **gốc repo** và đã có trong `.gitignore`. Chi tiết từng bộ: [data/README.md](../data/README.md).

```
data/
├── 1_vaipe/                               # bộ 1 — đơn thuốc VN
├── 2_roboflow_report_keywords/            # bộ 2 — phiếu xét nghiệm, 57 ảnh gốc (39 / 11 / 7)
├── 2_roboflow_report_keywords_augmented/  # bản tăng cường của bộ 2, không dùng cho thí nghiệm
├── 3_other_rvl_cdip/                      # bộ 3 — lớp "khác"
├── 4_self_captured/                       # bộ 4 — ảnh tự chụp, CHỈ để test, đã che thông tin cá nhân
└── 5_synthetic/                           # bộ 5 — tổng hợp từ ảnh sạch bộ 1
```

`code/samples/` chứa 5–10 ảnh mẫu **có commit** (đã che thông tin cá nhân), dùng cho demo và mốc tích hợp.
