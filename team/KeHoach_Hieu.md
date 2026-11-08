# KẾ HOẠCH CHI TIẾT — HIẾU

> **Vai trò:** Nhóm trưởng  
> **Phần phụ trách:** Mắt xích ① của pipeline — Thu nhận ảnh, Tiền xử lý & Phân loại tài liệu  
> **Việc nhóm trưởng:** khung báo cáo, ghép báo cáo / slide / video, đóng gói `.zip` và nộp bài ở cả 2 môn — xem mục 5  
> **Áp dụng cho:** Đồ án Xử lý ảnh (DIP, hạn **11/11/2026**) · Bài tập lớn Học máy (ML) · Mobile App MedTrack (module `01-core`)  
> **Tài liệu liên quan:** [PhanCong_TongHop.md](PhanCong_TongHop.md) · [DeTai_XuLyAnh_ChiTiet.md](../dip/docs/DeTai_XuLyAnh_ChiTiet.md) · [KeHoach_BTL_MachineLearning.md](../ml/docs/KeHoach_BTL_MachineLearning.md)

Mục tiêu của tài liệu: đưa phần của Hiếu từ mức "chắc ăn nhưng ít điểm mới" lên **mức tối đa** ở cả hai môn, bằng cách bổ sung các phần còn thiếu sau:

| Điểm yếu hiện tại | Cách khắc phục trong kế hoạch này |
| :--- | :--- |
| DIP: nắn phối cảnh, deskew, CLAHE là kỹ thuật quen thuộc, có sẵn trong notebook thực hành | Đánh giá như một **bài toán đo lường** trên bộ 6 (ground truth chính xác), so sánh nhiều phương pháp, có **phương án dự phòng** khi không tìm được 4 góc, thêm **khử lóa**, và **so sánh với app scan thương mại** |
| DIP: IQA (đo mờ, đo lóa) chỉ là ngưỡng đặt tay | Đánh giá IQA như **bộ phát hiện** có ROC/AUC; chọn ngưỡng bằng dữ liệu, không đặt tay |
| ML: nhãn có sẵn theo nguồn nên bài toán dễ, dễ bị nghi học thuộc nguồn | **Chứng minh và khắc phục** học thuộc nguồn (source probe, chuẩn hóa ảnh, Grad-CAM), đánh giá tài liệu lạ (open-set) bằng AUROC, calibration, kiểm định thống kê |
| ML: ghép báo cáo dễ làm qua loa | Quy trình ghép có khung, mốc thời gian và checklist |

---

## 1. VỊ TRÍ CỦA HIẾU TRONG PIPELINE

```mermaid
flowchart LR
    IN["📷 Ảnh bất kỳ<br>người dùng tải lên"] --> Q["IQA<br>đo mờ · đo lóa"]
    Q -->|ảnh xấu| RE["Báo chụp lại"]
    Q -->|đạt| G["Khử lóa<br>(nếu lóa nhẹ)"]
    G --> P["Tìm 4 góc<br>contour → Hough → minAreaRect"]
    P --> W["Nắn phối cảnh<br>xuất ma trận H"]
    W --> D["Chỉnh nghiêng<br>Hough / Radon / Projection"]
    D --> C["CLAHE"]
    C --> K{"Phân loại 3 lớp<br>+ ngưỡng từ chối"}
    K -->|Đơn thuốc / Phiếu XN| T["→ Trường (làm sạch)"]
    K -->|Khác / không chắc| X["⛔ Lưu làm tệp đính kèm"]
```

**Hợp đồng dữ liệu gửi cho Trường** (✅ chốt với Trường ngày 31/10; code: `code/src/common/contracts.py`, class `PreprocessResult`). Quy ước:

* `rectified_path`: ảnh **màu BGR, đã nắn phối cảnh và chỉnh nghiêng, chưa CLAHE**. CLAHE chỉ chạy trên bản sao dùng cho phân loại. Trường giữ nguyên kích thước và hệ tọa độ của ảnh này khi làm sạch.
* `homography`: **toàn bộ** biến đổi ảnh gốc → `rectified_path`, **gồm cả phép xoay chỉnh nghiêng** ($H = R \cdot H_{persp}$); `skew_deg` chỉ để báo cáo. Muốn chiếu một box trên ảnh đã nắn về ảnh gốc (màn hình xác nhận của Trường) thì nhân với **nghịch đảo** của `H` — hàm `PreprocessResult.to_source()`.
* `corners_src` theo thứ tự **trên-trái, trên-phải, dưới-phải, dưới-trái** trên ảnh gốc; `corner_method = "full_image"` khi không tìm được 4 góc (dùng cả ảnh, chỉ chỉnh nghiêng).
* `doc_type = "other"` thì pipeline dừng, không gửi sang Trường.

Kiểm tra bằng test: `code/tests/test_preprocessing.py` (`test_preprocess_homography_includes_deskew`, `test_preprocess_saves_image_before_clahe`).

```json
{
  "image_id": "rx_000123",
  "rectified_path": "out/rx_000123_rect.png",
  "homography": [[1.02, 0.03, -12.1], [0.01, 0.98, -8.4], [0.0, 0.0, 1.0]],
  "corners_src": [[34, 51], [980, 40], [1002, 1390], [20, 1401]],
  "corner_method": "contour",
  "skew_deg": -2.4,
  "doc_type": "prescription",
  "doc_confidence": 0.94,
  "quality": {"blur_var": 182.5, "glare_ratio": 0.012, "is_blurry": false, "has_glare": false, "glare_removed": false}
}
```

---

## 2. DỮ LIỆU HIẾU DÙNG

| Bộ | Vai trò với Hiếu | Việc cần làm |
| :--- | :--- | :--- |
| **1** VAIPE (đơn thuốc VN, **ảnh phẳng, sạch**) | Lớp `prescription`; **nguồn ảnh sạch phù hợp nhất cho bộ 6** (giữ đúng tỉ lệ, độ phân giải cao) | Không cần gán nhãn |
| **2** Roboflow Prescription (đơn Ấn Độ – Practo, **ảnh chụp thật**, nhiều đơn viết tay) | Lớp `prescription`; **ảnh chụp thật để thử nắn phẳng, chỉnh nghiêng** (D4, D6) | Không cần gán nhãn; gom 3 bản tăng cường của cùng ảnh gốc |
| **3** Roboflow Report Keywords (phiếu CBC, ảnh chụp) | Lớp `lab_report`; ảnh chụp thật nghiêng | Không cần gán nhãn; gom 3 bản tăng cường của cùng ảnh gốc |
| **4** Tài liệu khác | Lớp `other` | ✅ **Đã lấy (31/10)** bằng `code/scripts/fetch_rvl_cdip.py`: tập con RVL-CDIP gồm các lớp `form`, `invoice`, `letter`, `memo`, `email`, `news article`, `scientific report`, `questionnaire` — mỗi lớp 100 ảnh (50/25/25), tổng 800 ảnh; thêm `openset/` 200 ảnh lớp `budget`, `resume` **không dùng khi train** (mục 4.5c). Ghi chú license: RVL-CDIP chỉ dùng cho nghiên cứu |
| **5** Ảnh tự chụp | **Tập test độc lập** cho cả nắn phẳng và phân loại | Chụp ≥ 15 ảnh: đơn thuốc, phiếu XN, giấy khám, hóa đơn, CCCD đã che; gán nhãn lớp; **thêm 10 ảnh "loại lạ"** (giấy ra viện, kết quả siêu âm, biên lai) để test open-set |
| **6** Benchmark tổng hợp (Thắng sinh) | Ground truth cho sai số nắn, PSNR/SSIM, phát hiện mờ/lóa | ✅ Đã chốt với Thắng (31/10) — xem checklist nghiệm thu bên dưới |

**Checklist nghiệm thu bộ 6** (Thắng xác nhận 31/10; Thắng giữ manifest + checklist giao nhận; v1 **02/11**, v2 thêm lóa + mask lóa **04/11**):

1. `gt.json` mỗi ảnh: mức độ **mọi** loại hỏng (kể cả 0); `H` ảnh sạch → ảnh hỏng (gộp phối cảnh + xoay); 4 góc giấy trên ảnh hỏng (TL, TR, BR, BL). *H khác chiều với H của tiền xử lý — không sao, chấm bằng 4 góc.*
2. Mẫu âm sinh **độc lập**: ~30% ảnh `blur = 0`, ~30% ảnh `glare = 0`; có cả ảnh không mờ nhưng lóa và ngược lại; **val và test đều có mẫu âm lẫn dương**.
3. Lưới D3 **nằm trong test**: 3 mức phối cảnh (5, 10, 20%) × 5 góc (0, 5, 10, 20, 30°) = 15 ô, mỗi ô ≥ 10 ảnh từ **10 ảnh gốc khác nhau** (≥ 150 ảnh); các ảnh gốc này không xuất hiện ở train / val. Tham số chọn trên tập val hỗn hợp riêng.
4. Tập D4 chỉ xoay: 0, ±2, ±5, ±10, ±15°; giữ kích thước; vùng trống tô màu trung vị viền ảnh sạch (không viền đen, không REPLICATE); `angle_deg` dương = ngược chiều kim đồng hồ.
5. `glare_mask.png` (lóa = 255) cùng hệ tọa độ ảnh hỏng — từ v2.
6. Chia train / val / test theo ảnh gốc.

**Nghiệm thu v1.1** (Hiếu kiểm tra độc lập 02/11 bằng manifest + toàn bộ `labels/*.json`; v1 ngày 01/11 chưa đạt điểm 3, 4):

| # | Kết quả |
| :---: | :--- |
| 1 | ✅ Chiếu 4 góc ảnh sạch qua `H_clean_to_degraded` khớp `corners` trên cả 1.000 mẫu (lệch tối đa 0,005 px); ảnh hỏng / ảnh sạch cùng kích thước; 0 file thiếu. Mức hỏng bằng 0 không ghi trong JSON nhưng đủ trong manifest |
| 2 | ✅ Train, val: 60% không mờ, 60% không bóng. Test: 85% / 85% (lưới D3, tập D4 không có mờ / bóng) → còn ~45 ảnh mờ trong test, đủ vẽ ROC |
| 3 | ✅ **Lưới D3 đủ 15/15 ô**: 3 mức phối cảnh (5, 10, 20%) × 5 góc (0, 5, 10, 20, 30°), mỗi ô 10–12 ảnh từ đúng 10 ảnh gốc khác nhau; 0 ảnh gốc trùng train / val. Thêm ô 2° và 15° (2 ảnh / ô) |
| 4 | ✅ Tập chỉ xoay trong test: **±2, ±5, ±10, ±15°**, mỗi góc 10 ảnh. Nền màu giấy (~245, 0 pixel đen), giữ kích thước, góc dương ngược chiều kim đồng hồ. ⚠️ **Không có 0°** → D4 đo mức 0° trên `clean/` |
| 5 | ✅ **v2.1 (03/11):** `glare_mask/` cùng kích thước ảnh hỏng, rỗng ⇔ `glare_level = 0`, diện tích khớp `glare_area_ratio`; ngoài mask ảnh không đổi. 13% mẫu lóa mỗi tập, 3 mức đều nhau (diện tích 0,3 / 3,3 / 11,5% ảnh); val chỉ 20 mẫu lóa. v2 (sáng 03/11) giấy scan = 255 nên lóa không làm giấy sáng thêm (bộ phát hiện IoU = 0) → Thắng sửa v2.1: giảm phơi sáng 0,80–0,92 trước khi thêm lóa, thêm `clean_exposed/` cho D5; IoU bộ phát hiện hiện tại 0,13 / 0,61 / 0,76 theo mức. v1 khớp checksum 6.000/6.000 |
| 6 | ✅ 60 / 10 / 30 ảnh gốc, không trùng giữa các tập |

**Lưu ý khi chạy D2, D3:** 290 mẫu chỉ xoay (không phối cảnh) có góc giấy **nằm ngoài khung** (giữ kích thước nên xoay làm mất góc) → loại khỏi D2, D3; chỉ dùng cho D4. Mẫu chỉ mờ / chỉ bóng có 4 góc trùng khung ảnh → kết quả đúng là `full_image`. Cấu trúc thư mục từ 02/11: `<split>/images|labels|clean|clean_binary|shadow_mask|page_mask/<sample_id>`.

**Thực tế dữ liệu đã tải** (kiểm tra 31/10, bộ 1–3 lấy từ Drive chung của nhóm — xem `code/README.md` mục 7):

| Bộ | File train / valid / test | **Ảnh gốc thật** | Kích thước | Nhãn |
| :--- | :--- | :---: | :--- | :--- |
| 1 | 1.173 / 0 / 172 | 1.345 | Giữ tỉ lệ gốc (760×1070, 1150×1600, …) | JSON từng dòng chữ (`text`, `label`, `box`); tập test không có nhãn |
| 2 | 8.426 / 560 / 375 | 3.744 (train = 2.809 ảnh × 3 bản tăng cường Roboflow) | **Toàn bộ bị ép về 640×640** | YOLO, chỉ 1 lớp `medicine-names` |
| 3 | 117 / 11 / 7 | **57** (train = 39 ảnh × 3 bản tăng cường) | **Toàn bộ bị ép về 640×640** | YOLO 5 lớp: Report Type, parameter, reference value, unit, value |

Hệ quả cho phần Hiếu:

* **Vai trò bộ 1 / bộ 2 đảo so với kế hoạch ban đầu:** bộ 1 là ảnh phẳng (nguồn sạch cho bộ 6), bộ 2 mới là ảnh chụp thật (thử nắn phẳng).
* **Bộ 2, bộ 3 đã mất tỉ lệ khung gốc** (ép 640×640) → đo được góc / tìm 4 góc trên ảnh thật, nhưng tỉ lệ khung ảnh sau nắn không có ý nghĩa; sai số nắn định lượng chỉ đo trên bộ 6.
* **Mất cân bằng nặng:** `lab_report` chỉ 57 ảnh gốc so với ~5.000 ảnh `prescription` → cần thêm phiếu XN từ bộ 5 (ưu tiên khi chụp) và có thể nguồn khác; báo cáo Balanced Accuracy / Macro-F1, undersample `prescription` khi train.
* **Nguy cơ học thuộc nguồn rất cao:** kích thước 640×640, ảnh xám hóa, nhiễu muối tiêu do tăng cường Roboflow đủ để đoán nguồn → bắt buộc chuẩn hóa ảnh (mục 4.5b) và chạy source probe.
* **Bản tăng cường Roboflow:** các file cùng phần tên trước `.rf.` là một ảnh gốc → luôn nằm cùng một group khi chia fold.

---

## 3. MÔN DIP — CHI TIẾT

### 3.1. Các bước và thuật toán

| Bước | Thuật toán chính | Phương án so sánh | Tham số khởi điểm (sẽ tinh chỉnh ở 3.3) |
| :--- | :--- | :--- | :--- |
| **Chuẩn hóa kích thước** | Resize cạnh dài về 1000 px cho các bước dò tìm; giữ ảnh gốc để nắn | — | `long_side = 1000` |
| **Đo mờ** | Phương sai toán tử Laplacian trên ảnh xám | Tenengrad (Sobel), năng lượng tần số cao FFT | ngưỡng mờ ≈ 100 (chọn bằng ROC) |
| **Đo lóa** | Tỉ lệ pixel có V > ngưỡng và S < ngưỡng (vùng trắng cháy) trong HSV; lọc thành phần liên thông nhỏ | Ngưỡng toàn cục trên ảnh xám | `V > 240`, `S < 30`; **< 1%**: không lóa · **1–8%**: lóa nhẹ → khử · **> 8%**: báo chụp lại |
| **Khử lóa** *(mới)* | Inpainting vùng lóa: Telea và Navier–Stokes | Không khử | bán kính 5 px; chỉ chạy khi lóa nhẹ (1–8%) |
| **Tìm 4 góc** | Canny → `findContours` → `approxPolyDP` lấy tứ giác lồi lớn nhất | **Dự phòng 1:** Hough lines, chọn 2 cạnh ngang + 2 cạnh dọc trội nhất, lấy giao điểm. **Dự phòng 2:** `minAreaRect` của contour lớn nhất | Canny 50/150, `epsilon = 0.02 × chu vi`, diện tích tứ giác ≥ 20% ảnh |
| **Nắn phối cảnh** | `getPerspectiveTransform` + `warpPerspective` với nội suy bicubic; xuất ma trận `H` | Nội suy bilinear | kích thước đích theo độ dài cạnh lớn nhất |
| **Chỉnh nghiêng** | Projection Profile: xoay thử, chọn góc cho phương sai profile ngang lớn nhất | Hough lines (góc trung vị), Radon transform | dải ±15°, bước 0,25° |
| **CLAHE** | CLAHE trên kênh L của Lab | Cân bằng histogram toàn cục | `clipLimit = 2.0`, `tileGridSize = 8×8` |
| **Phân loại** | HOG + $k$-NN, ngưỡng từ chối theo khoảng cách | Histogram màu + $k$-NN | xem mục 4.2 |

### 3.2. Thí nghiệm và độ đo

| # | Thí nghiệm | Dữ liệu | Độ đo | Hình / bảng đưa vào báo cáo |
| :---: | :--- | :--- | :--- | :--- |
| **D1** | IQA như bộ phát hiện: mờ / không mờ, lóa / không lóa | Bộ 6 (biết mức mờ, mask lóa) | **ROC, AUC**, F1 tại ngưỡng chọn; IoU mask lóa | Đường ROC của Laplacian vs Tenengrad vs FFT |
| **D2** | Tìm 4 góc: contour vs Hough vs minAreaRect vs **kết hợp có dự phòng** | Bộ 6 | **Tỉ lệ thành công** (sai số < 2% đường chéo), sai số góc trung bình (px và % đường chéo) | Bảng 4 phương án |
| **D3** | Sai số nắn theo mức độ méo | Bộ 6 | Sai số góc theo góc nghiêng 0°→30° và 3 mức phối cảnh | **Đường cong** — hình chính của phần Hiếu |
| **D4** | Chỉnh nghiêng: Hough vs Radon vs Projection | Bộ 6, **tập con chỉ xoay** (không phối cảnh, chạy chỉnh nghiêng trực tiếp — nếu nắn phối cảnh trước thì ảnh đã thẳng, không còn góc để đo) + ảnh chụp thật bộ 2, bộ 3 | Sai số góc (độ), thời gian (ms) | Bảng 3 phương án |
| **D5** | Khử lóa: không khử vs Telea vs Navier–Stokes | Bộ 6 (có ảnh sạch) | PSNR / SSIM **trong vùng lóa**, CER sau OCR | Ảnh trước–sau + bảng |
| **D6** | Hiệu quả toàn bước tiền xử lý lên OCR | Bộ 6 + bộ 2 + bộ 3 + bộ 5 | **CER** ảnh gốc vs sau tiền xử lý; PSNR / SSIM so với ảnh sạch (bộ 6) | Bảng + đường CER theo mức hỏng |
| **D7** | **So sánh với app scan thương mại** *(mới)* | 20 ảnh bộ 5 chụp thật | CER sau OCR, tỉ lệ nắn đúng (đánh giá bằng mắt, 2 người chấm), thời gian | Bảng: Pipeline nhóm vs Google Drive Scan / Adobe Scan / CamScanner *(kiểm tra app còn hoạt động trước khi làm; Microsoft Lens đã có thông báo ngừng hỗ trợ)* |
| **D8** | Ablation phần Hiếu | Bộ 6 | CER khi tắt lần lượt: khử lóa, nắn phối cảnh, chỉnh nghiêng, CLAHE | Bảng ablation |
| **D9** | Thời gian chạy | Bộ 6 | ms/ảnh từng bước trên CPU | Bảng nhỏ |
| **D10** | Phân loại 3 lớp (phần DIP) | Bộ 1–5 | Accuracy, Macro-F1, Confusion Matrix; tỉ lệ tài liệu khác lọt vào pipeline | Confusion Matrix |

> **D7 thực hiện thế nào:** dùng app trên điện thoại chụp lại đúng 20 tờ giấy của bộ 5 trong cùng điều kiện ánh sáng, xuất ảnh, cho qua **cùng một bộ OCR** với ảnh của pipeline nhóm. Ghi rõ trong báo cáo đây là so sánh tham khảo vì app thương mại dùng thuật toán và camera khác.

### 3.3. Bảng tham số & lý do chọn (yêu cầu của đề)

Mỗi tham số chọn trên **tập validation của bộ 6**, có hình "độ đo theo tham số" trong phụ lục:

| Tham số | Khoảng thử | Chọn bằng | Ghi vào báo cáo |
| :--- | :--- | :--- | :--- |
| Ngưỡng phương sai Laplacian | 20 → 400 | Điểm tối ưu F1 trên đường ROC (D1) | Giá trị + AUC |
| Ngưỡng V, S cho lóa | V 220–250, S 20–60 | IoU mask lóa lớn nhất | Giá trị + IoU |
| `epsilon` của `approxPolyDP` | 0,01–0,05 × chu vi | Tỉ lệ thành công tìm góc (D2) | Giá trị + tỉ lệ |
| Ngưỡng Canny | (30,90) → (100,200) | Tỉ lệ thành công tìm góc | Giá trị |
| Bước góc chỉnh nghiêng | 0,1° – 1° | Đánh đổi sai số góc và thời gian (D4) | Giá trị |
| `clipLimit`, `tileGridSize` của CLAHE | 1–4; 4×4–16×16 | CER thấp nhất (D6) | Giá trị |

### 3.4. Cấu trúc code

```
code/src/preprocessing/
├── quality.py        # blur_score(), glare_mask(), assess_quality()
├── glare.py          # remove_glare(method="telea" | "ns")
├── corners.py        # find_corners_contour(), find_corners_hough(), find_corners_minrect(), find_corners()  ← có dự phòng
├── rectify.py        # rectify(img, corners) -> (warped, H)
├── deskew.py         # deskew_projection(), deskew_hough(), deskew_radon()
├── enhance.py        # apply_clahe()
└── pipeline.py       # preprocess(img) -> dict theo hợp đồng dữ liệu ở mục 1
code/src/classification/
├── features.py       # hog_features(), lbp_hist(), layout_density(), line_ratio()
└── classify.py       # train_knn(), predict_with_reject()
code/experiments/
├── d1_iqa_roc.py … d9_runtime.py   # mỗi thí nghiệm một script, xuất CSV + hình vào results/
code/tests/
└── test_preprocessing.py           # kiểm tra trên 5 ảnh mẫu: tìm được góc, H khả nghịch, ảnh ra đúng kích thước
```

---

## 4. MÔN ML — BÀI TOÁN 1: PHÂN LOẠI BIỂU MẪU 3 LỚP

### 4.1. Bài toán

* **Đầu vào:** ảnh cả trang (đã qua tiền xử lý).
* **Đầu ra:** `prescription` / `lab_report` / `other` + xác suất; **từ chối** khi xác suất cao nhất < ngưỡng.
* **Câu hỏi nghiên cứu chính** (dùng làm "xương sống" khi trình bày):
  1. Đặc trưng thủ công + mô hình cổ điển có cạnh tranh được với CNN trên dữ liệu ít không?
  2. Mô hình đang học **loại tài liệu** hay học **kiểu ảnh của nguồn dữ liệu**? Khắc phục thế nào?
  3. Mô hình có biết **"không biết"** khi gặp loại giấy chưa từng thấy không?

### 4.2. Đặc trưng

| Nhóm | Chi tiết | Số chiều (ước lượng) |
| :--- | :--- | :---: |
| **HOG** | Ảnh xám 256×256; 9 hướng, ô 16×16, khối 2×2 | ~8.100 → PCA còn 100–200 |
| **LBP** | LBP P = 8, R = 1, `method="nri_uniform"` (scikit-image, 59 bin; nếu dùng `"uniform"` thì chỉ còn 10 bin); histogram toàn ảnh + lưới 2×2 | 59 × 5 |
| **Mật độ bố cục** | Tỉ lệ pixel mực trên lưới 3×3 và 6×6 sau nhị phân hóa | 9 + 36 |
| **Đường kẻ** | Tỉ lệ pixel đường kẻ ngang / dọc (opening với SE 1×40 và 40×1) | 2 |
| **Khối văn bản** | Số dòng (projection), độ rộng dòng trung bình, tỉ lệ vùng trắng | ~5 |

**Ablation đặc trưng:** chạy với từng nhóm riêng và bỏ từng nhóm → biết nhóm nào đóng góp nhiều nhất.

### 4.3. Mô hình

| Nhóm | Mô hình | Siêu tham số tìm bằng nested CV |
| :--- | :--- | :--- |
| Bài giảng | $k$-NN | $k$ ∈ {1, 3, 5, 7, 11}, khoảng cách Euclid / Manhattan / cosine, trọng số uniform / distance |
| Bài giảng | Naive Bayes | Gaussian NB (`var_smoothing`) |
| Bài giảng | Logistic / Softmax Regression | $C$ ∈ {0,01 … 100}, L1 / L2, `class_weight` |
| Bài giảng | Random Forest | `n_estimators`, `max_depth`, `max_features`, `class_weight` |
| Bài giảng | SVM | Linear và RBF; $C$, $\gamma$ |
| Điểm cộng | **ResNet-18** fine-tune | Learning rate, số epoch, augmentation; early stopping |

### 4.4. Đánh giá chặt chẽ

| Hạng mục | Cách làm |
| :--- | :--- |
| **Chia dữ liệu** | `StratifiedGroupKFold` 5 fold, **group = cụm ảnh gần trùng** (gom bằng perceptual hash: cùng tờ giấy chụp nhiều lần, cùng mẫu đơn của một bác sĩ; **3 bản tăng cường Roboflow của cùng ảnh gốc** — cùng phần tên trước `.rf.` — luôn chung group) để ảnh gần giống nhau không rơi vào cả train và test. **Không** dùng nguồn dữ liệu làm group: phiếu xét nghiệm chỉ có một nguồn (bộ 3), group theo nguồn sẽ làm có fold train không còn phiếu xét nghiệm nào. Ảnh hưởng của nguồn được đo riêng bằng thí nghiệm cross-source và bộ 5 (mục 4.5) |
| **ResNet-18 trong nested CV** | Không tìm siêu tham số trong vòng trong (quá tốn); dùng bộ siêu tham số cố định chọn trên 1 tập validation, chỉ chạy vòng ngoài 5 fold để so sánh công bằng với 5 mô hình bài giảng |
| **Chọn siêu tham số** | Nested CV: vòng trong 3 fold chọn tham số, vòng ngoài 5 fold ước lượng hiệu năng |
| **Mất cân bằng lớp** | So sánh: không xử lý · `class_weight="balanced"` · undersampling; báo cáo Balanced Accuracy, Macro-F1 |
| **Độ đo** | Accuracy, Balanced Accuracy, Macro-F1, ROC-AUC từng lớp, Confusion Matrix chuẩn hóa |
| **Learning curve** | Macro-F1 theo số ảnh train (10% → 100%) cho SVM, RF, ResNet-18 — trả lời câu hỏi nghiên cứu 1 |
| **Kiểm định** | Khoảng tin cậy bootstrap 95% cho Macro-F1; **McNemar** khi so mô hình tốt nhất bài giảng với ResNet-18 |
| **Thời gian** | Thời gian huấn luyện và dự đoán (ms/ảnh) |

### 4.5. Ba phần nâng cao — khắc phục điểm yếu "học thuộc nguồn"

**(a) Chứng minh có học thuộc nguồn hay không**
* **Source probe trong cùng một lớp:** chỉ lấy đơn thuốc, huấn luyện mô hình đoán **đơn đến từ bộ 1 hay bộ 2**. Nếu đoán nguồn dễ (Accuracy gần 100%) thì đặc trưng mang "dấu vân tay" của nguồn → nguy cơ shortcut cao. *(Không probe trên cả 4 bộ: bộ 3 chỉ có phiếu XN, bộ 4 chỉ có tài liệu khác, nên đoán nguồn khi đó trùng với đoán lớp, không chứng minh được gì.)*
* **Cross-source:** huấn luyện bộ phân loại 3 lớp với đơn thuốc **chỉ từ bộ 1** (phiếu XN từ bộ 3, khác từ bộ 4) → kiểm tra trên đơn thuốc **bộ 2**: báo cáo Recall lớp `prescription`. Làm ngược lại (train bộ 2 → test bộ 1).
* **Test trên bộ 5** (ảnh tự chụp, không thuộc nguồn nào lúc train): con số chính đưa vào kết luận.
* **Grad-CAM** trên ResNet-18 và **SHAP / feature importance** trên RF: mô hình nhìn vào bố cục, bảng, hay nhìn vào viền ảnh, màu nền?

**(b) Khắc phục**
* **Chuẩn hóa ảnh trước khi trích đặc trưng:** ảnh xám, CLAHE, nhị phân hóa, cắt bỏ viền, resize cố định, nén lại JPEG cùng chất lượng, lọc trung vị để xóa nhiễu muối tiêu do tăng cường Roboflow → xóa bớt "dấu vân tay" của từng nguồn (bộ 2, 3 đều 640×640, bộ 1 giữ tỉ lệ gốc).
* **Augmentation cho ResNet:** xoay nhẹ, mờ, thay đổi độ sáng, nhiễu JPEG.
* Báo cáo **trước và sau khắc phục** trên bộ 5 và cross-source → đây là phát hiện có giá trị nhất của Bài toán 1.

**(c) Tài liệu lạ (open-set) và calibration**
* Bộ "loại lạ" gồm 2 phần, **báo cáo riêng**:
  * **Loại lạ mang dáng y tế** — 10 ảnh tự chụp (giấy ra viện, kết quả siêu âm, biên lai viện phí): đây là **phép thử chính**, vì chúng dễ bị nhận nhầm thành đơn thuốc hoặc phiếu XN.
  * **Loại lạ không y tế** — 1–2 lớp RVL-CDIP **không dùng khi train**: dễ, vì cùng phong cách với lớp `other` đã học; chỉ làm đối chứng.
* Kết quả mong muốn với loại lạ: được xếp vào `other` **hoặc** bị từ chối — cả hai đều đúng; sai khi lọt vào `prescription` / `lab_report` với độ tin cậy cao.
* Điểm "lạ": 1 − xác suất cao nhất (softmax / `predict_proba`); khoảng cách tới láng giềng gần nhất ($k$-NN).
* Độ đo: **AUROC** phát hiện tài liệu lạ, **FPR@95%TPR**, tỉ lệ loại lạ mang dáng y tế lọt vào pipeline.
* **Calibration:** biểu đồ độ tin cậy, **ECE**, Brier score; hiệu chỉnh bằng Platt scaling (SVM) và temperature scaling (ResNet-18) → chọn ngưỡng từ chối trên xác suất đã hiệu chỉnh.

### 4.6. Phân tích lỗi

* Gom các ca sai thành nhóm, báo cáo tỉ lệ %: phiếu XN không kẻ bảng bị nhầm thành đơn · đơn có bảng thuốc bị nhầm thành phiếu XN · ảnh mờ / lóa · ảnh chụp màn hình · tài liệu "khác" có bảng (hóa đơn).
* Mỗi nhóm: 3 ảnh ví dụ + Grad-CAM + lý do; đề xuất sửa nhóm nào trước.
* Vị trí ca sai so với biên quyết định của SVM (khoảng cách tới siêu phẳng).

### 4.7. Kết quả cần có trong báo cáo ML (phần Hiếu)

1. Bảng 5 mô hình + ResNet-18: Macro-F1 ± khoảng tin cậy, Balanced Accuracy, thời gian.
2. Bảng ablation nhóm đặc trưng.
3. Learning curve.
4. Bảng cross-source và bộ 5, **trước / sau** khắc phục shortcut + kết quả source probe.
5. Grad-CAM (4–6 ảnh).
6. Biểu đồ độ tin cậy + AUROC open-set.
7. Bảng phân loại lỗi.

---

## 5. VIỆC NHÓM TRƯỞNG: KHUNG, GHÉP, ĐÓNG GÓI, NỘP

### 5.1. DIP (hạn 11/11)

| Mốc | Việc |
| :--- | :--- |
| **01–02/11** | Gửi khung báo cáo DIP (4–10 trang): Đặt vấn đề → Dữ liệu (gồm bộ 6) → Phương pháp (4 module) → Thực nghiệm → Thảo luận → Phân công & khai báo AI. Mỗi người ~1,5–2 trang, **1 bảng + 1 hình chính**, phần còn lại vào phụ lục |
| **08/11** | Nhận phần của Trường (kèm bảng ablation T8), Khang, Thắng; kiểm tra số liệu, hình, bảng tham số; nhận phần nào ghép phần đó |
| **09/11** | Ghép báo cáo, slide 10–15 trang; dựng video ≤ 3 phút từ đoạn demo 30 giây của mỗi người; gửi cả nhóm duyệt |
| **10/11** | Đóng gói `LopXX_NhomYY_TenDeTai.zip`: `code/`, `requirements.txt`, `samples/` (5–10 ảnh), `README.md`, báo cáo PDF, slide `.pptx` + `.pdf`, video; kiểm tra chạy được trên máy sạch |
| **11/11** | Nộp qua email `uyenvnb@ptit.edu.vn`, tiêu đề trùng tên file |

### 5.2. ML: ghép báo cáo & slide

| Mốc | Việc |
| :--- | :--- |
| **Tuần 1** | Gửi cả nhóm **khung báo cáo** (Word / LaTeX): cấu trúc mục, quy ước đặt tên hình `Hinh_<bài toán>_<số>`, bảng mẫu, kiểu trích dẫn IEEE, mẫu "khai báo sử dụng AI" |
| **Tuần 2** | Viết phần chung: Đặt vấn đề, tổng quan pipeline 4 bài toán, mô tả 3 bộ dữ liệu, quy trình đánh giá chung |
| **Cuối tuần 3** | Nhận phần của 3 bạn; kiểm tra đủ: bảng chỉ số, Confusion Matrix, learning curve, phân tích lỗi; trả lại nếu thiếu |
| **Tuần 4** | Bảng so sánh tổng 4 bài toán × 6 mô hình; Kết luận; ghép slide 12–15 trang (mỗi bạn 2–3 slide); tập trình bày 2 lần có bấm giờ |

**Checklist trước khi nộp:** đủ 4 phần · thuật ngữ thống nhất · hình / bảng đánh số liên tục, có nguồn · tài liệu tham khảo đúng IEEE · khai báo AI · phân công · số liệu khớp giữa báo cáo và slide.

---

## 6. MOBILE APP — MODULE `01-core`

> **Ưu tiên thấp đến sau 11/11.** App không tính điểm cho 2 môn; vòng 2 NEXUS kéo đến 07/12 nên làm sau vẫn kịp.

| Nhóm | Màn hình | Gắn với phần DIP / ML |
| :--- | :--- | :--- |
| Chụp & xử lý | `03` chụp tài liệu (khung căn giấy) · `04` xem trước · `04b` báo ảnh mờ · `05` đang xử lý · `06` kết quả phân loại · `06b` chọn lại loại hồ sơ · `16` AI không chắc | Dùng trực tiếp `quality`, `corners`, ngưỡng từ chối |
| Khung ứng dụng | `01`, `01b`, `01c` onboarding · `02` trang chủ · `02b` lịch trống · `08` danh sách hồ sơ · `09` chi tiết hồ sơ · `10` dòng thời gian · `11` biểu đồ tổng quát · `14` cá nhân · `15` trạng thái trống | — |
| Backend | `POST /documents` nhận ảnh → lưu → trả `job_id`; worker gọi `preprocess()` + phân loại; `GET /documents/{id}` trả kết quả theo hợp đồng dữ liệu | Chạy chính code của mục 3.4 |

**Điểm cộng cho NEXUS:** chạy **đo mờ và tìm góc ngay trên điện thoại** khi đang ngắm camera (khung xanh khi đủ nét và thấy đủ 4 góc) → giảm ảnh xấu gửi lên server.

---

## 7. LỊCH LÀM VIỆC

### 7.1. Đến hạn DIP (11/11)

| Ngày | Việc | Kết quả kiểm tra được |
| :--- | :--- | :--- |
| **30–31/10** | Tải bộ 1–3, lấy tập con RVL-CDIP; chụp 15 + 10 ảnh bộ 5; gửi Thắng yêu cầu cho bộ 6; chốt JSON với Trường | Thư mục `data/` có đủ, file JSON mẫu |
| **01/11** | `quality.py`, `corners.py` (contour), `rectify.py`, `enhance.py` chạy trên 10 ảnh | 10 ảnh trước/sau |
| **02/11** | Hough + minAreaRect dự phòng; `deskew.py` 3 cách; `glare.py` | Unit test xanh |
| **03/11** | Có bộ 6 v1 (Thắng, 02/11) → D1 **phần mờ**, D2, D4 | 3 bảng / hình đầu tiên |
| **04/11** | Phân loại HOG + k-NN (D10); nhận bộ 6 v2 (có mask lóa) | Confusion Matrix |
| **05/11** | **Mốc tích hợp:** `preprocess()` chạy đầu-cuối cùng module của Trường, Khang, Thắng | Pipeline chạy trên 5 ảnh |
| **06/11** | D1 **phần lóa**, D5 khử lóa (cần mask lóa của v2); D3 đường cong | Đường ROC lóa, bảng D5, đường cong D3 |
| **07/11** | D6 CER (dùng bộ OCR của Khang), D8 ablation, D9 thời gian; bảng tham số 3.3 | Đủ hình / bảng cho báo cáo |
| **08/11** | Viết phần Hiếu trong báo cáo DIP (1,5–2 trang), 2–3 slide, đoạn demo 30 giây, README module; nhận phần của 3 bạn | Đủ 4 phần trước tối 08/11 |
| **09/11** | Ghép báo cáo, slide, video (mục 5.1); D7 so sánh app thương mại nếu kịp → phụ lục | Bản ghép gửi nhóm duyệt |
| **10/11** | Đóng gói `.zip`, chạy thử trên máy sạch; tập trình bày, sửa theo góp ý | File `.zip` |
| **11/11** | **Nộp đồ án DIP** qua email | Đã nộp |

**Phụ thuộc vào bạn khác:** bộ 6 v1 (Thắng, 02/11) và v2 có mask lóa (04/11) · bộ OCR `src/ocr/` (Khang, 01/11) cho D6, D7 · ảnh VAIPE (Thắng kiểm tra 31/10).

### 7.2. Môn ML (sau 11/11 tập trung toàn lực)

> ⚠️ **Hạn nộp môn ML chưa rõ** (không có trong `ml/references/requirements/require.md`) — hỏi thầy Đức ngay. Nếu hạn trước cuối tháng 11, phải dồn tuần 1–2 vào song song với DIP.

| Tuần | Việc |
| :--- | :--- |
| **1** (song song DIP) | Gửi khung báo cáo ML; trích đặc trưng (dùng lại `features.py`); loại ảnh trùng; tạo fold `StratifiedGroupKFold` |
| **2** | 5 mô hình + nested CV; ablation đặc trưng; learning curve; xử lý mất cân bằng |
| **3** | ResNet-18; source probe, cross-source, khắc phục shortcut; Grad-CAM; open-set + calibration; McNemar / bootstrap |
| **4** | Phân tích lỗi; ghép báo cáo & slide; tập trình bày |

---

## 8. RỦI RO & PHƯƠNG ÁN DỰ PHÒNG

| Rủi ro | Dấu hiệu | Phương án |
| :--- | :--- | :--- |
| Bộ 6 của Thắng trễ | Đến 03/11 chưa có | Tự sinh bản đơn giản (chỉ phối cảnh + mờ) bằng `warpPerspective` + `GaussianBlur` trên 50 ảnh bộ 1 (ảnh phẳng, sạch) |
| Thiếu phiếu xét nghiệm | Bộ 3 chỉ 57 ảnh gốc | Ưu tiên phiếu XN khi chụp bộ 5 (nhờ cả nhóm); tìm thêm nguồn phiếu XN công khai; báo cáo theo Macro-F1, Balanced Accuracy |
| Không tìm được 4 góc trên ảnh chụp màn hình / giấy sát mép | Tỉ lệ thành công D2 < 70% | Dùng toàn ảnh làm vùng tài liệu, chỉ chỉnh nghiêng; ghi rõ là hạn chế |
| Phân loại đạt ~100% nhưng sập trên bộ 5 | Khoảng cách > 15 điểm | Đây chính là phát hiện shortcut — trình bày ở 4.5, không giấu |
| Thiếu ảnh "loại lạ" | < 10 ảnh | Dùng thêm 2 lớp RVL-CDIP không có trong train |
| Không kịp D7 | Đến 08/11 chưa làm | Bỏ D7 (là phần cộng thêm), giữ D1–D6 |

**Thứ tự cắt khi thiếu thời gian (cắt từ dưới lên):** D7 → D5 → D9 → Grad-CAM → calibration. **Không cắt:** D2, D3, D6, D10, cross-source, bộ 5.

---

## 9. CHUẨN BỊ HỎI ĐÁP

| Câu hỏi dễ gặp | Hướng trả lời |
| :--- | :--- |
| Vì sao dùng phương sai Laplacian để đo mờ? | Laplacian là toán tử đạo hàm bậc 2, nhạy với cạnh; ảnh mờ ít cạnh sắc → phương sai nhỏ. Đã so với Tenengrad và FFT bằng ROC (D1) |
| Ngưỡng mờ chọn thế nào? | Không đặt tay: chọn điểm tối ưu F1 trên đường ROC của tập validation bộ 6 |
| Khi không tìm được 4 góc thì sao? | Có 2 lớp dự phòng (Hough, minAreaRect); tỉ lệ thành công từng phương án ở D2 |
| Vì sao chỉnh nghiêng sau khi đã nắn phối cảnh? | Nắn phối cảnh dựa trên mép giấy; khi mép giấy không song song với dòng chữ (giấy in lệch) vẫn còn góc nghiêng nhỏ |
| Vì sao CLAHE thay vì cân bằng histogram toàn cục? | Ảnh chụp có ánh sáng không đều; CLAHE cân bằng theo từng ô và giới hạn khuếch đại nhiễu |
| Bộ 6 là ảnh tổng hợp, có thực tế không? | Bộ 6 cho ground truth chính xác để so sánh phương pháp; kết luận được kiểm chứng lại trên ảnh chụp thật (bộ 2, bộ 3, bộ 5) và so với app thương mại (D7) |
| Mô hình có học thuộc nguồn không? | Trình bày source probe, cross-source, bộ 5 và kết quả sau khắc phục (mục 4.5) |
| Vì sao không dùng luôn CNN? | Learning curve cho thấy với dữ liệu ít, đặc trưng thủ công + SVM cạnh tranh được; CNN cần nhiều dữ liệu và kém giải thích hơn; McNemar cho biết khác biệt có ý nghĩa hay không |
| Gặp giấy chưa từng thấy thì sao? | Ngưỡng từ chối trên xác suất đã hiệu chỉnh; AUROC open-set trên bộ "loại lạ" |

---

## 10. CHECKLIST CUỐI

**DIP**
- [ ] Module `preprocessing/`, `classification/` có unit test, chạy được từ một lệnh
- [ ] D1–D6, D8–D10 có số liệu; D7 nếu kịp
- [ ] Đường cong sai số nắn theo góc nghiêng (D3)
- [ ] Bảng tham số & lý do chọn
- [ ] Phần báo cáo 1,5–2 trang, 2–3 slide, đoạn demo 30 giây, README module

**ML**
- [ ] 5 mô hình + ResNet-18, nested CV, khoảng tin cậy
- [ ] Ablation đặc trưng, learning curve
- [ ] Source probe, cross-source, bộ 5 trước / sau khắc phục
- [ ] Grad-CAM, open-set AUROC, calibration
- [ ] Bảng phân loại lỗi
- [ ] Báo cáo & slide ML đã ghép, có khai báo AI

**Chung**
- [ ] 25 ảnh bộ 5 (15 thường + 10 loại lạ) đã che thông tin cá nhân
- [ ] JSON hợp đồng dữ liệu đã chốt với Trường
- [ ] Trả lời được câu hỏi về phần của 3 bạn còn lại
