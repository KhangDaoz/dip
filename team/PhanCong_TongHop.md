# TỔNG HỢP PIPELINE, DỮ LIỆU & PHÂN CÔNG — ĐỒ ÁN XỬ LÝ ẢNH

> **Nhóm:** Hiếu (nhóm trưởng), Trường, Khang, Thắng  
> **Áp dụng cho:** Đồ án Xử lý ảnh số (DIP, hạn 11/11/2026)  
> **Đề tài chi tiết (đã chốt):** [DeTai_XuLyAnh_ChiTiet.md](../docs/DeTai_XuLyAnh_ChiTiet.md)  
> **Kế hoạch chi tiết từng người:** [Hiếu](KeHoach_Hieu.md) · [Trường](KeHoach_Truong.md) · [Khang](KeHoach_Khang.md) · [Thắng](KeHoach_Thang.md)

Nguyên tắc chia việc: mỗi người sở hữu **trọn một mắt xích** của pipeline, từ code, thí nghiệm đến phần báo cáo tương ứng.

---

## 1. PIPELINE

```mermaid
flowchart TD
    IN["📷 Ảnh tài liệu bất kỳ người dùng tải lên<br>(đơn thuốc, phiếu XN, giấy khám, hóa đơn, ảnh không liên quan...)<br>Ảnh chụp hoặc ảnh xuất từ phần mềm; có thể nghiêng, mờ, lóa, bóng đổ"]

    subgraph S1 ["① THU NHẬN & PHÂN LOẠI — Hiếu"]
        A1["IQA: đo mờ (Laplacian), đo lóa (kênh V – HSV)"]
        A2["Nắn phối cảnh 4 điểm + Deskew + CLAHE<br>(bỏ qua nếu ảnh đã phẳng)"]
        A3{"Phân loại 3 lớp<br>Đơn thuốc / Phiếu xét nghiệm / Tài liệu khác"}
    end

    X["⛔ Tài liệu khác hoặc độ tin cậy thấp<br>Dừng pipeline · lưu làm tệp đính kèm<br>Báo người dùng: chưa hỗ trợ đọc tự động"]

    subgraph S2 ["② LÀM SẠCH & TÁCH LỚP MỰC — Trường"]
        B1["Khử bóng đổ (ước lượng nền)"]
        B2["Tách lớp mực HSV: chữ in / chữ đỏ / viết tay / con dấu"]
        B3["Xóa dấu & chữ ký (Inpainting) + Nhị phân hóa Sauvola"]
    end

    subgraph S3 ["③ PHIẾU XÉT NGHIỆM — Khang"]
        C1["Dựng bảng<br>• Bảng không kẻ: gom cụm từ thành dòng/cột bằng Projection + Dilation<br>• Bảng có kẻ: Closing ngang/dọc lấy lưới"]
        C2["Phân loại vai trò từng cụm từ<br>(ML, ngoài phạm vi DIP)<br>tên chỉ số · kết quả · đơn vị · khoảng tham chiếu<br>→ ghép thành từng dòng chỉ số"]
        C3["Gắn cờ bất thường bằng luật<br>So kết quả với khoảng tham chiếu; thêm chữ đỏ / ↑↓ / H/L nếu có"]
    end

    subgraph S4 ["④ ĐƠN THUỐC — Thắng"]
        D1["Khoanh vùng bảng thuốc, tách từng khoản thuốc<br>(Projection + Dilation)"]
        D2["Phân loại dòng: tên thuốc / hàm lượng / cách dùng / hành chính<br>(ML, ngoài phạm vi DIP)"]
    end

    OCR["OCR theo từng vùng<br>VietOCR (tiếng Việt) · bộ OCR tiếng Anh (dữ liệu Ấn Độ)"]

    OUT["📤 JSON đầu ra<br>LabReportResult · PrescriptionResult"]

    IN --> A1 --> A2 --> A3
    A3 -->|"Đơn thuốc / Phiếu XN<br>ảnh phẳng + loại tài liệu + ma trận H"| B1 --> B2 --> B3
    A3 -->|Khác| X
    B3 -->|Phiếu xét nghiệm| C1 --> C2 --> C3
    B3 -->|Đơn thuốc| D1 --> D2
    C3 --> OCR
    D2 --> OCR
    OCR --> OUT
```

### 1.1. Đầu vào và đầu ra của từng mắt xích

Đây là **hợp đồng dữ liệu** giữa các thành viên, cần chốt trong Đợt 1 (29/10–01/11). Sau khi chốt, mỗi người có thể làm song song bằng dữ liệu giả lập mà không phải chờ người trước.

| Mắt xích | Người | Nhận vào | Trả ra |
| :--- | :--- | :--- | :--- |
| **① Thu nhận & Phân loại** | Hiếu | Ảnh gốc, **mọi loại tài liệu** | Ảnh đã nắn phẳng; ma trận homography `H`; nhãn `prescription` / `lab_report` / `other` kèm độ tin cậy; điểm mờ, điểm lóa. **Chỉ `prescription` và `lab_report` có độ tin cậy vượt ngưỡng mới được chuyển sang ②** |
| **② Làm sạch & Tách lớp mực** | Trường | Ảnh phẳng + loại tài liệu | Ảnh nhị phân sạch; mask từng lớp mực; danh sách thành phần liên thông kèm nhãn |
| **③ Phiếu xét nghiệm** | Khang | Ảnh sạch + mask | JSON các dòng chỉ số: bbox, tên, giá trị, đơn vị, khoảng tham chiếu, cờ `HIGH` / `LOW` / `NORMAL` |
| **④ Đơn thuốc** | Thắng | Ảnh sạch + mask | JSON các khoản thuốc: bbox, tên thuốc, hàm lượng, liều sáng/trưa/chiều/tối, số ngày |

---

## 2. DỮ LIỆU

### 2.1. Các nguồn dữ liệu

| # | Bộ dữ liệu | Loại tài liệu | Nguồn gốc & dạng ảnh | Nhãn có sẵn | Quy mô |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **1** | [VAIPE Pill 2022](https://www.kaggle.com/datasets/tommyngx/vaipepill2022) (Kaggle) | Đơn thuốc | Việt Nam, tiếng Việt; ảnh chụp. **Chỉ dùng phần ảnh đơn thuốc**, bỏ ảnh viên thuốc | *Cần kiểm tra:* có file nhãn kèm ảnh đơn không; nhãn theo vùng hay theo từng dòng chữ (tên thuốc, cách dùng...) | *cần xác nhận* |
| **2** | [Prescription Image Labelling](https://universe.roboflow.com/prescription-image-vkedm/prescription-image-labelling-xh0np) (Roboflow) | Đơn thuốc | Ấn Độ, tiếng Anh (Practo); **ảnh xuất từ phần mềm**, sạch, không nghiêng, không dấu, không viết tay | Bounding box theo **vùng**: `medicine-names` (cả bảng thuốc, **chưa tách từng khoản**), `doctor_information`, `patient_information`, `Diagnosis` | *cần xác nhận* |
| **3** | [Report Keywords Detection](https://universe.roboflow.com/world-of-ai/report-keywords-detection) (Roboflow) | Phiếu xét nghiệm | Ấn Độ, tiếng Anh; **ảnh chụp** phiếu máu (CBC), có nghiêng, có ảnh chụp màn hình; bảng **không kẻ lưới**; không thấy chữ đỏ hay cờ H/L | Bounding box theo **từng cụm từ**: `Report Type`, `parameter`, kết quả, đơn vị, khoảng tham chiếu *(xác nhận tên nhãn chính xác)* | *cần xác nhận* |
| **4** | **Tài liệu khác** | Lớp `other` | Tập con bộ tài liệu công khai **RVL-CDIP** (form, invoice, letter, scientific report...) | Nhãn lớp tài liệu | *cần bổ sung* |
| **5** | **Ảnh nhóm tự chụp** | Cả 3 lớp | Đơn thuốc, phiếu xét nghiệm Việt Nam; giấy khám, hóa đơn, giấy tờ hành chính. Che thông tin cá nhân, có sự đồng ý | Nhóm tự gán | Mục tiêu ≥ 50 ảnh |
| **6** | **Benchmark tổng hợp** *(Thắng sinh)* | Đơn thuốc | Sinh từ ảnh sạch của bộ 2: nghiêng, phối cảnh, mờ, bóng đổ, lóa flash, dấu đỏ giả, mỗi loại nhiều mức | **Ground truth chính xác tuyệt đối:** ảnh sạch gốc, ma trận biến dạng, bounding box đã biến đổi, mức độ hỏng | Vài trăm ảnh, không phải vẽ tay |

> Bộ 6 là nguồn ground truth chính cho các độ đo DIP (PSNR, SSIM, F-measure, sai số nắn, IoU) và cho đường cong "độ đo theo mức độ hỏng". Bộ 5 **không dùng để huấn luyện**. Nó là **tập test độc lập** của cả nhóm, và là bằng chứng duy nhất cho thấy hệ thống chạy được trên tài liệu Việt Nam thật.

### 2.2. Ai dùng bộ nào

| | **Hiếu** | **Trường** | **Khang** | **Thắng** |
| :--- | :--- | :--- | :--- | :--- |
| **Bộ dùng để phát triển / chọn tham số** | 1 + 2 (đơn thuốc), 3 (phiếu XN), 4 (khác) | 1 (chính: có dấu, viết tay), 3 | 3 | 1 + 2 |
| **Bộ dùng để test** | Tập test của 1–4 (chia theo cụm ảnh gần trùng) + **bộ 5** + **bộ 6** (sai số nắn) | Tập test của 1, 3 + **bộ 5** + **bộ 6** (F-measure) | Tập test của 3 (chia theo phiếu) + **bộ 5** (phiếu VN) | Tập test của 1, 2 (chia theo mẫu đơn) + **bộ 5** + **bộ 6** (IoU) |
| **Ground truth có sẵn** | Nhãn lớp suy ra từ nguồn; ma trận biến dạng của bộ 6 | Ảnh sạch của bộ 6 làm ảnh nhị phân chuẩn | **Vai trò từng cụm từ** của bộ 3 | Vùng bảng thuốc (bộ 2); bounding box đã biến đổi của bộ 6 |
| **Phải tự gán thêm** | Nhãn lớp cho bộ 5 | Nhãn 4 lớp mực: luật HSV gán sơ bộ, sửa tay tập test | Transcript cụm từ để đo CER (`data/dataset3_50/`); box + vai trò cho ~10 phiếu bộ 5 | Bounding box **từng khoản thuốc** trên ~30 ảnh; viết bộ sinh bộ 6 |

### 2.3. Đặc điểm dữ liệu cần lưu ý

1. **Học thuộc nguồn (Hiếu).** Đơn thuốc bộ 2 là ảnh xuất từ phần mềm, phiếu xét nghiệm bộ 3 là ảnh chụp, tài liệu khác từ RVL-CDIP là ảnh scan đen trắng. Mô hình có thể học "kiểu ảnh của từng nguồn" thay vì loại tài liệu. Bắt buộc báo cáo kết quả trên **bộ 5** và thí nghiệm train trên bộ 1, test trên bộ 2.
2. **Bảng không kẻ lưới (Khang).** Cách tách lưới bằng closing ngang/dọc không chạy được trên bộ 3. Phương pháp chính là gom cụm từ thành dòng và cột bằng projection profile và dilation thích nghi; cách tách theo đường kẻ dùng cho phiếu Việt Nam có kẻ lưới trong bộ 5.
3. **Không có cờ màu (Khang).** Bộ 3 không có chữ đỏ hay H/L. Cờ bất thường là **luật so sánh số** (kết quả so với khoảng tham chiếu), không cần học; cờ chữ đỏ HSV chỉ là phần mở rộng trên phiếu Việt Nam của bộ 5.
4. **Thiếu dấu và chữ viết tay (Trường).** Bộ 2 và bộ 3 gần như không có con dấu, chữ ký hay chữ đỏ. Dữ liệu cho bài toán tách lớp mực chủ yếu đến từ **bộ 1** và bộ 5.
5. **Chưa có nhãn từng khoản thuốc (Thắng).** Bộ 2 chỉ khoanh cả bảng thuốc. Ground truth từng khoản phải tự vẽ trên một tập con (đủ để tính IoU), trừ khi bộ 1 có sẵn.
6. **Hai ngôn ngữ.** Bộ 1 và bộ 5 là tiếng Việt, bộ 2 và bộ 3 là tiếng Anh. OCR phải xử lý cả hai ngôn ngữ; báo cáo phải nói rõ phần xét nghiệm chủ yếu được kiểm chứng trên dữ liệu tiếng Anh.

### 2.4. Quy tắc dùng dữ liệu

- **Kho dữ liệu chung (Bắt buộc):** Toàn bộ dataset của nhóm được lưu trữ và chia sẻ chính thức tại [Google Drive Folder MedTrack Dataset](https://drive.google.com/drive/folders/1PtO0ret01AW0mF504ayRFE7GoE2Ez4_G?usp=drive_link). **Mọi thành viên bắt buộc tải dữ liệu từ link Drive này về và đưa vào thư mục `data/` ở gốc repo**, không tải riêng lẻ từ các nguồn khác để tránh lệch phiên bản dữ liệu.
- **Trường** giữ bộ dữ liệu gốc và một cách chia Train/Val/Test **70/15/15 cố định** (cùng seed, chia theo ảnh) cho cả nhóm. Không ai tự chia lại.
- Mọi bước chuẩn hóa, chọn đặc trưng, PCA chỉ fit trên tập Train (chống rò rỉ dữ liệu).
- Ảnh trùng hoặc gần trùng giữa các tập phải loại trước khi chia.
- Ghi lại license của từng bộ trong `data/README.md`; ảnh tự chụp phải che họ tên, ngày sinh, mã bệnh nhân, số điện thoại.

### 2.5. Việc cần làm ngay về dữ liệu

| Việc | Người | Hạn |
| :--- | :--- | :--- |
| Tải 3 bộ, ghi số ảnh, danh sách nhãn, license vào `data/README.md`; tạo split chung | Trường | 01/11 |
| Kiểm tra bộ 1 có ảnh đơn thuốc và file nhãn kèm theo không, nhãn theo vùng hay theo dòng | Thắng | 31/10 |
| Xác nhận tên nhãn chính xác của bộ 3; viết script OCR từng cụm từ và ghép cụm thành dòng | Khang | 01/11 |
| Viết bộ sinh **bộ 6** (benchmark tổng hợp) và sinh bản đầu tiên cho cả nhóm dùng | Thắng | 02/11 |
| Lấy tập con RVL-CDIP cho lớp `other` | Hiếu | 01/11 |
| Thu bộ 5: mỗi người chụp ≥ 15 ảnh (đơn thuốc, phiếu xét nghiệm, giấy tờ khác), che thông tin cá nhân | Cả nhóm | 03/11 |

---

## 3. PHÂN CÔNG CHI TIẾT

| | **Hiếu** | **Trường** | **Khang** | **Thắng** |
| :--- | :--- | :--- | :--- | :--- |
| **DIP**<br>*(Xử lý ảnh)* | **Tiền xử lý & Phân loại** *(Chương 2, 3, 8)*<br>• Đo mờ (phương sai Laplacian), đo lóa (kênh V – HSV)<br>• Tìm 4 góc giấy, nắn phối cảnh, xuất ma trận `H`; bỏ qua khi ảnh đã phẳng<br>• So sánh 3 cách chỉnh nghiêng: Hough, Radon, projection<br>• CLAHE; phân loại 3 lớp HOG + k-NN, chặn tài liệu khác<br>• Đường cong sai số nắn theo góc nghiêng 0°→30° trên bộ 6<br>• Bảng tham số & lý do chọn (ngưỡng mờ, ngưỡng lóa, tham số CLAHE)<br>📦 `src/preprocessing/`, `src/classification/`<br>📏 Sai số góc (px), PSNR, SSIM, Macro-F1 | **Làm sạch & Tách lớp mực** *(Chương 3, 4, 5, 7)*<br>• Khử bóng đổ bằng ước lượng nền (Median / Closing kích thước lớn)<br>• Tách mực HSV: dấu đỏ, bút xanh, chữ in đen; Inpainting, giữ chữ đỏ cảnh báo<br>• So sánh 4 cách nhị phân hóa: Otsu, Niblack, Sauvola, Wolf<br>• Đường cong F-measure theo mức bóng đổ / dấu trên bộ 6<br>• Bảng tham số & lý do chọn (cửa sổ và hệ số k của Sauvola, ngưỡng HSV)<br>📦 `src/cleaning/`<br>📏 F-measure, PSNR, CER trước/sau<br>📝 **Thêm:** ghép báo cáo & slide DIP; **ablation toàn pipeline** (tắt từng bước, đo CER cuối); gom video demo, đóng gói `.zip` | **Dựng bảng phiếu xét nghiệm** *(Chương 5, 7)*<br>• Nhị phân hóa, Dilation ngang gom cụm từ thành dòng; Projection Profile dọc tìm cột<br>• Ghép dòng × cột thành ô; chịu được ảnh nghiêng nhẹ và chụp màn hình<br>• So sánh 3 cách: gom theo khoảng cách cố định, Connected Components + XY-cut, Projection + Dilation thích nghi<br>• Bảng tham số & lý do chọn (kích thước SE theo chiều cao chữ)<br>• Mở rộng nếu kịp: bảng có kẻ (Closing ngang/dọc), cờ chữ đỏ HSV trên phiếu Việt Nam<br>📦 `src/table_extraction/`<br>📏 IoU dòng/ô so với nhãn cụm từ của bộ 3, CER | **Bóc tách đơn thuốc & Benchmark tổng hợp** *(Chương 2, 4, 7)*<br>• Khoanh vùng bảng thuốc; Projection ngang tách dòng, Dilation gom thành từng khoản<br>• Vẽ ground truth từng khoản trên ~30 ảnh<br>• Đối chứng: OCR cả trang vs OCR từng khoản<br>• ⭐ **Bộ sinh bộ 6:** từ ảnh sạch bộ 2 sinh nghiêng, phối cảnh, mờ, bóng, lóa, dấu đỏ theo nhiều mức; lưu ảnh sạch, ma trận biến dạng, box đã biến đổi cho cả nhóm<br>• Đường cong IoU / CER theo mức độ hỏng<br>📦 `src/prescription_segmentation/`, `src/synthetic/`<br>📏 IoU khoản thuốc, CER |

### 3.1. Kiểm tra cân bằng

Thang điểm ước lượng độ nặng: 1 = nhẹ, 2 = vừa, 3 = nặng.

| | Hiếu | Trường | Khang | Thắng |
| :--- | :---: | :---: | :---: | :---: |
| **DIP** | 2,5<br>*Tiền xử lý + so sánh chỉnh nghiêng + sai số nắn (2) + phân loại 3 lớp (0,5)*<br>*+ nhóm trưởng: khung, ghép báo cáo / slide / video, đóng gói, nộp* | 2,5<br>*Làm sạch + so sánh 4 cách nhị phân hóa (2) + ablation toàn pipeline (0,5)* | 2,5<br>*Dựng bảng không kẻ + so sánh 3 cách* | 2,5<br>*Bóc tách đơn + ground truth (1,5) + bộ sinh benchmark tổng hợp (1)* |
| **Thu bộ 5** | ≥ 15 ảnh | ≥ 15 ảnh | ≥ 15 ảnh | ≥ 15 ảnh |
| **Báo cáo** | Tự viết Phương pháp, Kết quả, Phân tích lỗi của phần mình | ← | ← | ← |

> Điểm là ước lượng ban đầu. Sau Đợt 1 (01/11), cả nhóm xem lại tiến độ thực tế; ai chậm do phần việc khó hơn dự kiến (dễ xảy ra nhất: dựng bảng không kẻ của Khang, tách lớp mực của Trường) thì chuyển bớt phần "mở rộng nếu kịp".
