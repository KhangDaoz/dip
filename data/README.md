# Dữ liệu MedTrack — kiểm kê, nhãn và nguồn

Cập nhật ngày **01/11/2026** từ bản Drive chung đã tải trên máy Trường. Phạm vi kiểm kê chi tiết: bộ ①–③. Số liệu là **file thực tế trong gói local**, gồm ảnh tăng cường; không mặc nhiên là số ảnh nguồn độc lập. Giữ nguyên phân tập đã tải.

**Các split đã chốt của bộ 1–4 và 6 được giữ nguyên.** Bộ 5 là ảnh tự chụp mới, hiện chưa có split; xem mục ⑤ bên dưới. Các bảng mô tả cấu trúc gói dữ liệu, không phải yêu cầu chia lại split của những bộ đã chốt.

Tài liệu này và [SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md) được cấu hình để chia sẻ qua Git. Ảnh, nhãn, cấu hình gốc trong các dataset và báo cáo `results/` tiếp tục được bỏ qua.

| Thư mục | Ảnh và nhãn hiện có | License và trạng thái đối chiếu |
|---|---|---|
| `1_vaipe/` | Train: 1.173 ảnh + 1.173 JSON; test: 172 ảnh, chưa có nhãn; valid chỉ có file giữ chỗ | Kaggle API khai báo `DbCL-1.0`; điều khoản nguồn VAIPE ban đầu chưa xác minh riêng |
| `2_roboflow_prescription/` | 9.361 ảnh + 9.361 nhãn TXT: train 8.426, valid 560, test 375; 1 lớp `medicine-names` | Trang công khai đúng version 1: `CC BY 4.0`; YAML local: `Private` — cần làm rõ mâu thuẫn |
| `3_roboflow_report_keywords/` | 135 ảnh + 135 nhãn TXT: train 117, valid 11, test 7; 5 lớp `Report Type`, `parameter`, `reference value`, `unit`, `value` | `CC BY 4.0`, khớp trang dự án và YAML local |
| `4_other_rvl_cdip/` | 1.000 ảnh PNG và `labels.csv` | Xem README đi kèm; chưa xác minh độc lập |
| `5_self_captured/` | 83 ảnh chưa gán nhãn chi tiết: 48 phiếu xét nghiệm, 18 đơn thuốc, 17 tài liệu khác; chia trong `lab_report/`, `prescription/`, `other/`; có `manifest.csv` | Bản hiện tại đã tải lên Drive nhóm; từng thành viên còn phải review ảnh và xác nhận ẩn danh/quyền sử dụng trước khi đánh giá |
| `6_synthetic/` | 1.000 mẫu: train 600, val 100, test 300; mỗi mẫu có 5 PNG và `gt.json` | Dữ liệu tổng hợp; cần đối chiếu điều kiện sử dụng ảnh nguồn |

## Kiểm kê chi tiết bộ ①–③

Đối chiếu với [phân công nhóm](../team/PhanCong_TongHop.md), [kế hoạch Hiếu](../team/KeHoach_Hieu.md), [kế hoạch Khang](../team/KeHoach_Khang.md) và metadata đi kèm. Dùng các gói từ [Drive chung của nhóm](https://drive.google.com/drive/folders/1PtO0ret01AW0mF504ayRFE7GoE2Ez4_G?usp=drive_link) theo [hướng dẫn repo](../code/README.md) mục 7. Căn cứ và giới hạn đối chiếu nguồn/license ở [SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md).

### ① VAIPE — `1_vaipe/`

- Nguồn được tài liệu nhóm ghi: [VAIPE Pill 2022 trên Kaggle](https://www.kaggle.com/datasets/tommyngx/vaipepill2022). Bản local chỉ giữ đơn thuốc, không phải toàn bộ kho ảnh viên thuốc. API Kaggle xác nhận dataset `tommyngx/vaipepill2022` hiện có version 1, phát hành 18/07/2022; gói local không kèm mã/checksum để chứng minh trùng toàn bộ gói Kaggle.
- Ảnh ở `<split>/images/`, nhãn ở `<split>/labels/`. Kích thước thực tế: 760×1070, 770×1420 hoặc 1150×1600; không ép thành 640×640.

| Split | Ảnh | JSON | Vùng nhãn | Ghi chú |
|---|---:|---:|---:|---|
| train | 1.173 | 1.173 | 41.208 | Ghép đủ theo tên file |
| valid | 0 | 0 | 0 | Chỉ có file giữ chỗ; chưa có tập validation riêng |
| test | 172 | 0 | 0 | Không có đáp án trong gói local |

JSON là danh sách đối tượng. Cả 41.208 đối tượng đều có `id`, `text`, `label`, `box`; `mapping` có ở 2.844 đối tượng. `box = [xmin, ymin, xmax, ymax]`, tọa độ pixel trên chính ảnh nguồn, chưa qua nắn phối cảnh. Toàn bộ text không rỗng; bbox có diện tích dương và nằm trong ảnh. `mapping` là số nguyên liên kết được nguồn cung cấp; chưa diễn giải thành mã thuốc chuẩn khi không có bảng tra kèm theo.

| Nhãn | Số vùng train | Ý nghĩa nội dung |
|---|---:|---|
| `diagnose` | 1.650 | Chẩn đoán |
| `drugname` | 2.844 | Tên thuốc |
| `quantity` | 2.844 | Số lượng thuốc |
| `usage` | 3.994 | Cách dùng/liều dùng |
| `date` | 1.173 | Ngày |
| `other` | 28.703 | Nội dung khác |

**Cảnh báo trùng trong split gốc của gói tải:** SHA-256 xác nhận `train/images/VAIPE_P_TRAIN_1172.png` trùng byte với `test/images/VAIPE_P_TEST_32.png`. Tổng 1.345 file ảnh tương ứng 1.344 nội dung file khác nhau theo SHA-256; chưa kiểm tra trùng gần giống bằng hình ảnh. Đây không phải yêu cầu chia lại: khi đánh giá dùng split chung đã chốt, không dùng nguyên split gốc thay thế.

**Sàng lọc dấu/viết tay ngày 01/11:** đã quét màu ở độ phân giải gốc và xem ảnh tổng quan đủ 1.173 ảnh train của gói tải, chưa tuyển được ca dương tính đáng tin cậy. Ứng viên xanh duy nhất `VAIPE_P_TRAIN_953.png` đã mở ảnh gốc: màu nằm ở quảng cáo, không phải chữ ký. Chưa duyệt mọi trang ở độ phân giải gốc, nên không kết luận tuyệt đối không có nét viết tay đen/dấu nhạt. Báo cáo: `results/vaipe_ink_screening/selection.json`; không dùng 172 ảnh test để tuyển mẫu phát triển hoặc chia lại split chung.

### ② Roboflow đơn thuốc — `2_roboflow_prescription/`

- `data.yaml`: workspace `prescription-image-vkedm`, project `prescription-image-labelling-xh0np`, **version 1**, `nc: 1`. [Trang phiên bản công khai](https://universe.roboflow.com/prescription-image-vkedm/prescription-image-labelling-xh0np/dataset/1); YAML local dẫn tới cùng dự án trong Roboflow App.
- `README.roboflow.txt` ghi thời điểm xuất 08/10/2025, tự xoay theo EXIF, ép 640×640, chuyển xám và tăng cường. Thực đếm: mọi ảnh đều 640×640; không coi những ảnh tăng cường là các ảnh nguồn độc lập.

| Split | Ảnh | TXT | Bbox | Nhãn rỗng |
|---|---:|---:|---:|---:|
| train | 8.426 | 8.426 | 8.779 | 1 |
| valid | 560 | 560 | 578 | 0 |
| test | 375 | 375 | 397 | 0 |
| **Tổng** | **9.361** | **9.361** | **9.754** | **1** |

Chỉ có class ID **0 → `medicine-names`**, gồm 9.754 bbox. Theo phân công nhóm, đây là vùng bảng/danh sách thuốc; nhãn hiện có không xác định riêng từng khoản thuốc, từng dòng hay transcript. Không có các lớp `doctor_information`, `patient_information`, `Diagnosis` trong bản này, dù tài liệu kế hoạch còn nhắc chúng.

**Chênh lệch cần giữ nguyên khi báo cáo:** README xuất và trang Roboflow version 1 ghi 9.362 ảnh (8.427 train); thực tế có 9.361 ảnh (8.426 train), đủ file nhãn tương ứng. Không xác định được ảnh nào thiếu chỉ bằng con số tổng. Phân nhóm theo phần tên trước `.rf.` cho 3.744 nhóm; đây là quy ước tên, chưa phải xác minh số ảnh gốc độc lập bằng thị giác. Không thấy nhóm tên hoặc ảnh trùng byte vượt split.

### ③ Report Keywords — `3_roboflow_report_keywords/`

- `data.yaml`: workspace `world-of-ai`, project `report-keywords-detection`, **version 1**, `nc: 5`. [Link dự án/phiên bản trong metadata](https://universe.roboflow.com/world-of-ai/report-keywords-detection/dataset/1).
- README xuất ghi 07/06/2023; mô tả ép 640×640 và tăng cường. Thực tế mọi ảnh đều 640×640. Có 57 nhóm theo phần tên trước `.rf.`, tương ứng quy mô nguồn được Hiếu ghi; không coi 135 ảnh xuất là 135 ảnh gốc độc lập đã xác minh.

| Split | Ảnh | TXT | Bbox | Nhãn rỗng |
|---|---:|---:|---:|---:|
| train | 117 | 117 | 7.350 | 0 |
| valid | 11 | 11 | 490 | 1 |
| test | 7 | 7 | 358 | 0 |
| **Tổng** | **135** | **135** | **8.198** | **1** |

| Class ID | Tên chính xác | Số bbox | Ý nghĩa / role của Khang |
|---|---|---:|---|
| 0 | `Report Type` | 113 | Loại/tên báo cáo → `report_type` |
| 1 | `parameter` | 2.170 | Tên chỉ số → `parameter` |
| 2 | `reference value` | 1.751 | Khoảng tham chiếu → `range` |
| 3 | `unit` | 2.011 | Đơn vị → `unit` |
| 4 | `value` | 2.153 | Giá trị/kết quả → `result` |

Số liệu lớp khớp thống kê của Khang. Các bbox là vùng từ/cụm từ theo vai trò, chưa phải ground truth hàng/cột hoàn chỉnh hoặc bản chép chữ OCR. Không thấy ảnh trùng byte hoặc nhóm tên vượt split; chưa kiểm tra trùng gần giống.

### Bộ benchmark OCR cụm từ — `dataset3_50/`

- Gồm **50 crop** lấy từ split `valid` và `test` của bộ ③, bao phủ 15 phiếu nguồn và 4 role: `parameter` (13), `value` (13), `unit` (12), `reference value` (12).
- Manifest chứa transcript `text_gt` đã được người dùng xác nhận và thông tin truy vết đến nhãn/crop nguồn. Crop không resize hay tăng cường.
- Đây là tập đánh giá OCR cố định, không dùng để huấn luyện; không sửa lại split nguồn.
- Chạy benchmark bằng `python experiments/k0_ocr_benchmark.py --dataset ../data/dataset3_50` từ thư mục `code/`. Script và định dạng kết quả xem [code/experiments/README.md](../code/experiments/README.md). Output nằm dưới `results/ocr_benchmark_dataset3_50/` và không commit.

### Quy ước nhãn và giới hạn kiểm tra

- **Hai bộ Roboflow:** mỗi dòng TXT có đúng 5 trường `class_id x_center y_center width height`; bốn tọa độ được chuẩn hóa theo chiều rộng/cao ảnh. Mọi bbox đọc được đều có class hợp lệ, tọa độ hữu hạn, kích thước dương và nằm trong ảnh. Đây không phải bbox `[x, y, w, h]` dạng pixel trong hợp đồng pipeline; phải chuyển đổi và ánh xạ hình học khi dùng cùng ảnh đã nắn.
- Không có ảnh thiếu nhãn hoặc nhãn không có ảnh trong các split có gán nhãn của ba bộ. Ngoại lệ có chủ ý trong mô tả dữ liệu: VAIPE test không có nhãn, valid trống. File nhãn rỗng vẫn được tính là có file, chưa khẳng định đó là mẫu âm hay thiếu annotation.
- Hai file cần xác nhận: bộ ② `train/labels/1ZURu9NF4V_jpg.rf.031621143d4d4ff589a6bda90724ee16.txt`; bộ ③ `valid/labels/WhatsApp-Image-2022-09-25-at-10-56-11_jpeg.rf.37d622f14f9cfd76663b353b0b8b8732.txt`.
- Đã dùng Pillow `verify()` trên 10.841 file ảnh, không phát hiện lỗi ở mức kiểm tra này. Chưa duyệt từng ảnh/bbox/text bằng mắt, chưa chứng minh không rò rỉ ảnh gần giống và chưa đánh giá chất lượng thuật toán làm sạch.
- Nhãn hiện có không cung cấp mask lớp mực/dấu/chữ ký hoặc ảnh nhị phân chuẩn. Text VAIPE có thể làm tham chiếu OCR sau khi kiểm tra; TXT YOLO của bộ ②–③ chỉ có class/bbox.
- Các đường dẫn `../train/images` trong YAML là đường dẫn tương đối của bản xuất; khi chạy công cụ cần cấu hình đúng thư mục dataset thực tế, không mặc nhiên suy thành `data/train/images`.

Chi tiết từng file và chứng cứ kiểm tra nằm ở `results/primary_dataset_audit/inventory.json`, báo cáo `results/primary_dataset_audit/README.md`; script đọc dữ liệu là `results/audit_primary_datasets.py`. Đây là tài liệu hỗ trợ **chỉ có trên máy Trường, không đi kèm repo**. Các số liệu, đường dẫn bất thường và nguồn tham khảo cần cho nhóm đã được ghi ngay trong hai tài liệu Data để đọc độc lập.

### Các điểm còn cần nhóm xác nhận

1. Cặp trùng VAIPE ghi bên trên thuộc train/test gốc trong gói tải; khi đánh giá dùng split chung đã chốt, không tự chia lại hoặc lấy nguyên split gốc thay thế.
2. Hai TXT rỗng là mẫu âm có chủ ý hay nhãn còn thiếu.
3. Vì sao gói bộ ② ghi `Private` trong khi trang công khai version 1 ghi CC BY 4.0; đồng thời xác nhận gói nhóm có chủ ý ít hơn nguồn một ảnh train hay không.
4. Điều khoản riêng của nguồn VAIPE ban đầu cho gói 2022, nếu cần xác minh ngoài khai báo của người đăng Kaggle.

Các điểm này được ghi nhận để nhóm quyết định; không tự sửa ảnh, nhãn, YAML hoặc phân tập. Việc hoàn thiện tài liệu không đồng nghĩa mọi nhãn đã được duyệt bằng mắt hoặc mọi vấn đề license đã được giải quyết. Xem căn cứ tại [SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md).

## Các bộ khác — thông tin bổ sung

Bộ ⑥ mới khớp toàn bộ checksum trong `checksums_v1.sha256`; đạt hai test dữ liệu của nhóm về ánh xạ khoản thuốc và độ phủ D3/D4. Manifest vẫn ghi `v1`; không suy phiên bản chỉ từ tên thư mục. Kiểm tra này không thay thế duyệt nhãn bằng mắt.

### ⑤ Ảnh tự chụp — `5_self_captured/`

Cập nhật cấu trúc ngày **03/11/2026**. Bản hiện tại có 83 ảnh: 48 phiếu xét nghiệm, 18 đơn thuốc và 17 tài liệu khác; đã tải lên Drive nhóm cùng `manifest.csv`. Ảnh dùng ID ổn định liên tục theo loại `sc_<category>_<number>.<ext>` và sắp theo `lab_report/`, `prescription/`, `other/`.

Đây là bộ ảnh **chưa gán nhãn chi tiết**: chưa có box, transcript OCR, dòng/cột, khoản thuốc hoặc nhãn thành phần. Thư mục chỉ nhóm ảnh theo loại tài liệu đã biết. `manifest.csv` để trống thông tin người đóng góp, nhóm tài liệu trùng, split; `privacy_review` đang là `required` và `annotation_status` là `not_started` cho mọi ảnh. Bản hiện tại đã được tải lên Drive nhóm, nhưng chưa xác nhận bằng duyệt nội dung rằng toàn bộ ảnh đã được ẩn danh. Người phụ trách từng ảnh cần hoàn tất review trước khi dùng ảnh cho benchmark.

Quy ước dùng bộ 5:
- Giữ nguyên ảnh nguồn sau khi đã ẩn danh; không resize hay nén lại khi chưa có lý do và ghi nhận rõ.
- Mỗi người điền `contributor_id`; cùng một tài liệu chụp nhiều lần phải dùng chung `document_group_id`.
- Chưa gán `split` tại đây. Theo phân công ban đầu, bộ 5 là dữ liệu kiểm thử độc lập, không đưa vào train/tinh chỉnh. Sau khi review và gán nhãn, nhóm cần chốt danh sách test theo từng task, đồng thời tránh tách ảnh cùng tài liệu sang nhiều tập. Nếu muốn dùng bộ 5 cho train/valid, cần thống nhất thay đổi quy định trước.
- Mỗi người tạo file nhãn riêng theo task, liên kết bằng `image_id`; không ghi đè manifest hoặc ảnh nguồn. Khang phụ trách nhãn dòng/cụm từ cho khoảng 10 phiếu xét nghiệm; các phần khác theo phân công tại `team/`.
- `data/` bị Git ignore. Chia sẻ ảnh, manifest và nhãn qua Drive nhóm; tài liệu này là hướng dẫn được theo dõi bằng Git.

## Nguồn phát triển đang giữ riêng trên máy Trường

- `6_synthetic_v1_old/6_synthetic/`: đúng dữ liệu đo T1–T2 và nguồn của 5 ảnh trong `img_test`. Không thay bằng bộ ⑥ mới khi chạy lại kết quả cũ.
- `PhieuXetNghiem/report keywords detection.v1i.yolov8/`: bản 57 ảnh, còn là nguồn của 5 ảnh xét nghiệm trong `img_test`. Bản mới có quy mô/biến thể khác nên giữ nguồn gốc này.

Các nguồn cũ trên chỉ phục vụ tái lập thử nghiệm đã có, không thay bộ chuẩn của nhóm. Nhật ký dọn dữ liệu và bảng ánh xạ VAIPE cũ → mới nằm ở `results/data_sync_cleanup/` trên máy Trường, không đi kèm repo.
