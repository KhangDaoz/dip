# Code — Trích xuất đơn thuốc VAIPE (v2)

> **v2 (11/11, theo [plan.md](../plan.md)):** pipeline học sâu A→B→C→D cho đơn thuốc VAIPE. Code v1 (phiếu XN, bộ 1–5)
> chuyển sang `legacy/`, dùng lại làm đối chứng cổ điển. Các script cũ trong `experiments/` (D*, T*, K*, S*) còn import
> module v1 nên **không chạy được nữa**; thí nghiệm v2 (E-*) viết mới.

```
ảnh ─► [A] enhance/ ─► [B] detect/ ─► [C] recognize/ ─► [D] assemble/ ─► PrescriptionResult (JSON)
       deskew, shadow,   yolo (chính)     C0/C1 Tesseract     classify_rules (D1), group (D2),
       denoise, clahe    classical        C2/C3 VietOCR       lexicon (D3), vlm (D4 đối chứng)
                         Ghép: src/pipeline.py · Độ đo: src/metrics/ · Hợp đồng: src/common/contracts.py
```

### Chạy nhanh v2

```bash
cd code
pip install -r requirements-dl.txt && pip install -e .
pytest                                                   # phần cần trọng số / data tự SKIP
python tools/to_yolo.py && python tools/to_vietocr.py    # nhãn YOLO + crops VietOCR
python tools/build_items_gt.py --split all --review 10   # đáp án khoản thuốc + 10 ảnh để soát
python run_pipeline.py samples/ --preset classical        # chạy được ngay, không cần trọng số
python run_pipeline.py samples/ --preset dl_c2            # YOLO + VietOCR pretrained
python run_pipeline.py samples/                           # pipeline chính: YOLO + VietOCR fine-tune (C3)
python run_pipeline.py samples/ --vlm                     # đối chứng VLM (đặt VLM_API_KEY trước)
python app.py                                             # demo Gradio
```

| Preset | B | C | Phân lớp |
| :--- | :--- | :--- | :--- |
| `classical` | Sauvola + giãn | Tesseract C0 | luật D1 |
| `hybrid` | YOLO | Tesseract C0 | YOLO |
| `dl` (chính) | YOLO | VietOCR C3 | YOLO |
| `dl_c2` | YOLO | VietOCR C2 | YOLO |

Trọng số (không commit): `weights/yolo_vaipe.pt` từ `notebooks/train_yolo.ipynb`, `weights/vietocr_finetuned.pth`
từ `notebooks/finetune_vietocr.ipynb` (Kaggle). Ablation E-X1: `Pipeline(detector="gt", ocr="C3")(img, id, gt=annotations)`.

---

# (Cũ, v1) Code — Pipeline xử lý ảnh MedTrack

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
pytest                                                 # toàn bộ test (25 test)
python run_pipeline.py samples/                        # đầu-cuối ① → ② → ③/④ trên 10 ảnh mẫu
python run_pipeline.py anh.jpg --doc-type prescription # ép loại tài liệu (bỏ qua bộ phân loại)
python experiments/s1_items.py                         # một thí nghiệm → results/s1/
```

**Đầu ra mong đợi của `run_pipeline.py`:** với mỗi ảnh, in trạng thái từng bước (`ok` / `chưa làm` / `LỖI`) và ghi vào `out/`:

| File | Nội dung |
| :--- | :--- |
| `<id>_rect.png` | ① ảnh đã nắn phối cảnh + chỉnh nghiêng |
| `<id>_1_preprocess.json` | ① ma trận H, 4 góc, cách tìm góc, góc nghiêng, điểm mờ / lóa, loại tài liệu + độ tin cậy |
| `<id>_clean.png`, `<id>_bin.png` | ② ảnh xám đã khử bóng, ảnh nhị phân (Sauvola, mực trắng) |
| `<id>_stamp.png`, `<id>_hw.png`, `<id>_red.png` | ② mask dấu đỏ, nét bút xanh, chữ đỏ được giữ |
| `<id>_2_clean.json` | ② đường dẫn ảnh, thành phần liên thông có nhãn, tham số |
| `<id>_3_result.json` | ③ các dòng chỉ số (bbox, tên, kết quả, cờ) hoặc ④ các khoản thuốc (bbox, tên, liều) |
| `<id>_overlay.png` | ảnh đã nắn có vẽ khung dòng chỉ số / khoản thuốc (dùng cho demo) |

Bộ phân loại dùng mô hình `models/doc_knn.pkl` (do `experiments/d10_classify.py` tạo). OCR là tùy chọn: thiếu Tesseract thì bbox vẫn đầy đủ, trường chữ để trống.

**Thí nghiệm** (kết quả vào `results/<mã>/`; chạy theo thứ tự vì có phụ thuộc):

| Người | Lệnh (từ `code/`) | Mục KeHoach |
| :--- | :--- | :--- |
| Thắng | `s0_gt_vaipe.py` → `s1_items.py` → `s4_region.py` → `s2_robustness.py`; `s6_bo5_check.py` (sinh `ignore_band` cho T2); `s3_ocr.py` (OCR); `s5_bo4.py` (bộ 4, định tính) | 8.3 |
| Trường | `t1_background.py`; `t2_binarize.py` (sau S6) → `t3_degradation.py`; `t4_stamp.py`; `t2cer_binarize_ocr.py` (OCR); `t5_red_text.py` (bộ 4, định tính); `t8_ablation.py` (OCR, mục 10) | 6.2 |
| Khang | `k0_gt.py` → `k2_table.py` (K1, K2, K4, 5-fold) → `k3_skew.py`; `k5_ocr.py` (OCR); `k6_bo4.py` (bộ 4, định tính) | 7.2 |
| Hiếu | `d1_iqa.py`, `d2_corners.py` (D2 + D3), `d4_deskew.py`, `d5_glare.py`, `d9_runtime.py`, `d10_classify.py`; `d6_enhance.py` (OCR, dùng bộ chạy của T8) | 5.2 |

Các thí nghiệm có OCR (T2-CER, S3, D6, K5, T8) cần Tesseract 5 + gói `vie`, `eng`; hàm dùng chung ở `experiments/_ocr_common.py`. T8 và D6 mất ≈ 20–30 phút mỗi script với 4 tiến trình.

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
├── 4_self_captured/                       # bộ 4 — ảnh tự chụp, CHỈ để test (chưa xác nhận đã che thông tin cá nhân)
└── 5_synthetic/                           # bộ 5 — tổng hợp từ ảnh sạch bộ 1
```

`code/samples/` chứa 10 ảnh mẫu **có commit**, dùng cho demo và mốc tích hợp: 2 đơn VAIPE (thông tin cá nhân đã che sẵn từ nguồn), 2 đơn bộ 5 có nghiêng / phối cảnh / bóng / dấu, 4 phiếu xét nghiệm bộ 2 (CC BY 4.0), 2 tài liệu khác RVL-CDIP. Chưa dùng ảnh bộ 4 vì chưa xác nhận đã che thông tin.
