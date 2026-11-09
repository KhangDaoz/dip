# Code — Pipeline xử lý ảnh MedTrack

Khung code chung cho đồ án Xử lý ảnh (DIP). **Mỗi người chỉ sửa module của mình.** Riêng `common/` và hợp đồng dữ liệu, ai muốn đổi thì phải báo cả nhóm.

```
① Hiếu ──PreprocessResult──► ② Trường ──CleaningResult──► ③ Khang  (phiếu XN)  ──LabReportResult──► app
 preprocessing/                cleaning/                     table_extraction/
 classification/                                          ④ Thắng (đơn thuốc) ──PrescriptionResult─► app
                                                             prescription_segmentation/
                               Dùng chung: common/ · ocr/ (Khang) · synthetic/ = bộ 5 (Thắng)
```

## 1. Cài đặt (làm một lần)

Yêu cầu Python **3.10 → 3.13**. Riêng phần OCR nên dùng **3.11**, xem mục 6.

```bash
cd code
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # để import được preprocessing, cleaning, ... từ bất kỳ đâu
pytest                    # phải XANH: test của hàm chưa viết sẽ hiện SKIPPED
```

## 2. Ai làm gì

| Module | Người | File | Thí nghiệm | Test |
| :--- | :--- | :--- | :--- | :--- |
| `preprocessing/` | Hiếu | `quality`, `glare`, `corners`, `rectify`, `deskew`, `enhance`, `pipeline` | D1–D9 | `test_preprocessing.py` |
| `classification/` | Hiếu | `features`, `classify` | D10 | `test_classification.py` |
| `cleaning/` | Trường | `background`, `ink`, `inpaint`, `binarize`, `components`, `pipeline` | T1–T8 | `test_cleaning.py` |
| `ocr/` *(dùng chung)* | Khang | `engines` (cần viết), `metrics` (**đã có** `cer`, `wer`, `corpus_cer`) | — | `test_ocr_metrics.py` |
| `table_extraction/` | Khang | `geometry`, `rows`, `columns`, `cells`, `flags`, `pipeline` | K1–K7 | `test_table.py` |
| `synthetic/` *(bộ 5)* | Thắng | `degrade`, `generate`, `config.yaml` | — | `test_synthetic.py` |
| `prescription_segmentation/` | Thắng | `region`, `lines`, `items`, `fields`, `pipeline` | S1–S6 | `test_prescription.py` |
| `common/` | Cả nhóm | `contracts`, `io`, `paths`, `seed`, `experiment`, `testing` | — | `test_contracts.py` |

Mỗi hàm đã có **tên, tham số mặc định lấy theo kế hoạch, và docstring mô tả đầu vào / đầu ra**. Thân hàm hiện là `raise NotImplementedError`. Cách làm việc với một hàm:

1. Mở file của mình, thay `raise NotImplementedError(...)` bằng code thật.
2. Chạy `pytest tests/test_<module>.py`. Test của hàm đó sẽ chuyển từ SKIPPED sang PASSED (hoặc FAILED nếu sai).
3. Muốn thêm hàm hay tham số thì cứ thêm. **Không đổi tên hoặc kiểu trả về** của hàm mà người khác gọi (`*/pipeline.py`, `ocr/*`, `synthetic/degrade.py`).

## 3. Quy ước bắt buộc

| Hạng mục | Quy ước |
| :--- | :--- |
| Ảnh màu | `uint8`, shape `(H, W, 3)`, kênh **BGR** (theo OpenCV) |
| Ảnh nhị phân / mask | `uint8`, `(H, W)`, **mực / vùng quan tâm = 255, nền = 0** |
| bbox | `[x, y, w, h]`, số nguyên, gốc ở góc trên-trái |
| Tọa độ của ②③④ | Tính trên **ảnh đã nắn** của ①. Chiếu về ảnh gốc bằng `PreprocessResult.to_source()` |
| 4 góc | Thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái |
| Đọc / ghi ảnh | Dùng `common.io.read_image` / `write_image`. `cv2.imread` **không đọc được đường dẫn có dấu tiếng Việt** trên Windows |
| Đường dẫn | Lấy từ `common.paths` (`DATA_DIR`, `DATASETS[1..5]`, `RESULTS_DIR`, `OUT_DIR`), không viết cứng |
| Seed | `common.seed.set_seed(42)` ở đầu mọi thí nghiệm |
| CER / WER | Dùng `ocr.metrics`. Khi báo cáo cả tập, dùng `corpus_cer` (tổng lỗi / tổng ký tự) |

## 4. Hợp đồng dữ liệu — `src/common/contracts.py`

Mỗi mắt xích trả về một `dataclass`, có `.to_dict()`, `.from_dict()` và `.validate()`. Ví dụ JSON đầy đủ nằm trong `tests/test_contracts.py`, lấy đúng từ kế hoạch của từng người.

| Hợp đồng | Người gửi → nhận | Trường bắt buộc |
| :--- | :--- | :--- |
| `PreprocessResult` | Hiếu → Trường | `rectified_path`, `homography` (ảnh gốc → ảnh đã nắn), `corners_src`, `corner_method`, `skew_deg`, `doc_type`, `doc_confidence`, `quality` |
| `CleaningResult` | Trường → Khang, Thắng | `clean_gray_path`, `binary_path`, `masks`, `components[]` (`bbox`, `label`, `confidence`), `params` |
| `LabReportResult` | Khang → app | `report_type`, `rows[]` (`parameter`, `result`, `unit`, `range`, `flag`, `loinc`, ...) |
| `PrescriptionResult` | Thắng → app | `patient`, `diagnosis`, `items[]` (`name`, `strength`, `dose`, `days`, `note`, ...) |

Giá trị hợp lệ: `doc_type` ∈ {`prescription`, `lab_report`, `other`}; `corner_method` ∈ {`contour`, `hough`, `minrect`, `full_image`}; `label` ∈ {`printed`, `handwritten`, `stamp`, `other`}; `flag` ∈ {`LOW`, `NORMAL`, `HIGH`, `UNKNOWN`}.

**Muốn đổi hợp đồng:** báo trong nhóm chat, rồi sửa `contracts.py` và `tests/test_contracts.py` **trong cùng một commit**.

## 5. Chạy

```bash
pytest                                        # toàn bộ test
pytest tests/test_cleaning.py -v              # test một module
python run_pipeline.py samples/               # đầu-cuối ① → ② → ③/④ (mốc tích hợp 05/11)
python run_pipeline.py anh.jpg --doc-type lab_report   # ép loại tài liệu khi bộ phân loại chưa xong
python experiments/d1_iqa_roc.py              # một thí nghiệm → results/d1/
python -m synthetic.generate                  # sinh bộ 5 → data/5_synthetic/
```

`run_pipeline.py` in ra trạng thái từng bước của từng ảnh (`ok` / `chưa làm` / `LỖI`). Kết quả JSON ghi vào `out/`.

Thí nghiệm: copy file mẫu của mình trong `experiments/` (`d1_`, `t1_`, `k1_`, `s1_`), xem [experiments/README.md](experiments/README.md).

## 6. Thư viện thêm

| File | Khi nào cài | Ghi chú |
| :--- | :--- | :--- |
| `requirements.txt` | Ngay | OpenCV, scikit-image, scikit-learn, pandas, matplotlib, imagehash, pytest |
| `requirements-ocr.txt` | Khi làm OCR (Khang trước 01/11, sau đó cả nhóm) | Tesseract 5 phải **cài riêng** ([bản Windows](https://github.com/UB-Mannheim/tesseract/wiki)), chọn thêm gói ngôn ngữ `vie`. PaddleOCR / VietOCR có thể chưa hỗ trợ Python 3.13 → dùng venv Python 3.11 |

## 7. Dữ liệu và kết quả (không commit)

> [!IMPORTANT]
> **Link Google Drive tải toàn bộ Dataset của nhóm:** [Google Drive Folder MedTrack Dataset](https://drive.google.com/drive/folders/1PtO0ret01AW0mF504ayRFE7GoE2Ez4_G?usp=drive_link)  
> **Yêu cầu:** Mọi thành viên bắt buộc tải dataset từ link Drive trên về và giải nén vào đúng thư mục `data/` ở gốc repo theo cấu trúc bên dưới (không tự tải nguồn lẻ bên ngoài để đảm bảo tính đồng bộ dữ liệu).

`data/`, `results/`, `out/` nằm ở **gốc repo** và đã có trong `.gitignore`. Cấu trúc `data/` thống nhất như sau (xem `common/paths.py`):

```
data/
├── 1_vaipe/                     # bộ 1 (đơn thuốc VN, đã lọc bỏ ảnh viên thuốc)
├── 2_roboflow_report_keywords/  # bộ 2 (phiếu xét nghiệm) — 57 ảnh gốc
├── 2_roboflow_report_keywords_augmented/  # bản tăng cường của bộ 2, chỉ dùng khi cần
├── 3_other_rvl_cdip/            # bộ 3 (Hiếu) — sinh bằng: python scripts/fetch_rvl_cdip.py (đã có trên Drive)
├── 4_self_captured/             # bộ 4 — ĐÃ CHE thông tin cá nhân trước khi chép vào
└── 5_synthetic/                 # bộ 5 (Thắng sinh)
```

`code/samples/` chứa 5–10 ảnh mẫu **có commit**, dùng cho demo và mốc tích hợp. Chỉ bỏ vào đây ảnh đã che thông tin cá nhân.
