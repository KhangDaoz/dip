# Dữ liệu MedTrack — kiểm kê, nhãn và nguồn

Cập nhật ngày **01/11/2026** từ bản Drive chung đã tải trên máy Trường. Phạm vi kiểm kê chi tiết: bộ ①–②. Số liệu là **file thực tế trong gói local**, gồm ảnh tăng cường; không mặc nhiên là số ảnh nguồn độc lập. Giữ nguyên phân tập đã tải.

**Cập nhật 08/11/2026:** bỏ bộ Roboflow đơn thuốc (bộ 2 cũ) khỏi dữ liệu nhóm và đánh số lại 1–5: VAIPE ① · Report Keywords ② · RVL-CDIP ③ · tự chụp ④ · tổng hợp ⑤. Bộ ② chỉ giữ 57 ảnh gốc; 78 bản tăng cường của train chuyển sang `2_roboflow_report_keywords_augmented/` (vai trò từng file trong `split_manifest.csv`). Số liệu kiểm kê 01/11 bên dưới giữ nguyên làm lịch sử.

**Các split đã chốt của bộ 1, 2 và 5 được giữ nguyên.** Bộ 4 là ảnh tự chụp mới, hiện chưa có split; xem mục ④ bên dưới. Các bảng mô tả cấu trúc gói dữ liệu, không phải yêu cầu chia lại split của những bộ đã chốt.

Tài liệu này và [SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md) được cấu hình để chia sẻ qua Git. Ảnh, nhãn, cấu hình gốc trong các dataset và báo cáo `results/` tiếp tục được bỏ qua.

| Thư mục | Ảnh và nhãn hiện có | License và trạng thái đối chiếu |
|---|---|---|
| `1_vaipe/` | Train: 1.173 ảnh + 1.173 JSON; test: 172 ảnh, chưa có nhãn; valid chỉ có file giữ chỗ | Kaggle API khai báo `DbCL-1.0`; điều khoản nguồn VAIPE ban đầu chưa xác minh riêng |
| `2_roboflow_report_keywords/` | 57 ảnh gốc + nhãn TXT: train 39, valid 11, test 7 (phần ③ không dùng split này mà dùng kiểm định chéo 5-fold trên cả 57 ảnh) (78 bản tăng cường ở `2_roboflow_report_keywords_augmented/`); 5 lớp `Report Type`, `parameter`, `reference value`, `unit`, `value` | `CC BY 4.0`, khớp trang dự án và YAML local |
| `3_other_rvl_cdip/` | 1.000 ảnh PNG và `labels.csv` | Xem README đi kèm; chưa xác minh độc lập |
| `4_self_captured/` | 83 ảnh chưa gán nhãn chi tiết: 48 phiếu xét nghiệm, 18 đơn thuốc, 17 tài liệu khác; chia trong `lab_report/`, `prescription/`, `other/`; có `manifest.csv` | Bản hiện tại đã tải lên Drive nhóm; từng thành viên còn phải review ảnh và xác nhận ẩn danh/quyền sử dụng trước khi đánh giá |
| `5_synthetic/` | 1.500 mẫu (v1: 1.000 mẫu v00–v09; v2 thêm 500 mẫu v10–v14: lóa, dấu, bút, JPEG + nhiễu, tổ hợp), chia theo ảnh gốc 60 / 10 / 30; mỗi mẫu có ảnh, mask và `gt.json` | Dữ liệu tổng hợp; cần đối chiếu điều kiện sử dụng ảnh nguồn |

## Kiểm kê chi tiết bộ ①–②

Đối chiếu với [kế hoạch toàn nhóm](../team/KeHoach.md) mục 3 và metadata đi kèm. Dùng các gói từ [Drive chung của nhóm](https://drive.google.com/drive/folders/1PtO0ret01AW0mF504ayRFE7GoE2Ez4_G?usp=drive_link) theo [hướng dẫn repo](../code/README.md) mục 7. Căn cứ và giới hạn đối chiếu nguồn/license ở [SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md).

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


### ② Report Keywords — `2_roboflow_report_keywords/`

- `data.yaml`: workspace `world-of-ai`, project `report-keywords-detection`, **version 1**, `nc: 5`. [Link dự án/phiên bản trong metadata](https://universe.roboflow.com/world-of-ai/report-keywords-detection/dataset/1).
- README xuất ghi 07/06/2023; mô tả ép 640×640 và tăng cường. Thực tế mọi ảnh đều 640×640. Có 57 nhóm theo phần tên trước `.rf.`, tương ứng quy mô nguồn được Hiếu ghi; không coi 135 ảnh xuất là 135 ảnh gốc độc lập đã xác minh. Bảng dưới là số liệu bản xuất (01/11); từ 08/11 thư mục chính chỉ giữ 57 ảnh: valid/test là bản đơn chưa tăng cường, train giữ 1 bản đại diện/nhóm (đủ box nhất, ít nhiễu muối tiêu nhất) — có thể vẫn bị crop/shear nhẹ vì Roboflow không giữ bản gốc của train.

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

### Bộ benchmark OCR cụm từ — `dataset2_50/`

- Gồm **50 crop** lấy từ split `valid` và `test` của bộ ②, bao phủ 15 phiếu nguồn và 4 role: `parameter` (13), `value` (13), `unit` (12), `reference value` (12).
- Manifest chứa transcript `text_gt` đã được người dùng xác nhận và thông tin truy vết đến nhãn/crop nguồn. Crop không resize hay tăng cường.
- Đây là tập đánh giá OCR cố định, không dùng để huấn luyện; không sửa lại split nguồn.
- Dùng để chọn engine OCR cho bộ 2 (công cụ đo CER, xem `code/src/ocr/`). Script benchmark viết lại theo mẫu [code/experiments/_template.py](../code/experiments/_template.py); output đặt dưới `results/` và không commit.

### Quy ước nhãn và giới hạn kiểm tra

- **Bộ Roboflow:** mỗi dòng TXT có đúng 5 trường `class_id x_center y_center width height`; bốn tọa độ được chuẩn hóa theo chiều rộng/cao ảnh. Mọi bbox đọc được đều có class hợp lệ, tọa độ hữu hạn, kích thước dương và nằm trong ảnh. Đây không phải bbox `[x, y, w, h]` dạng pixel trong hợp đồng pipeline; phải chuyển đổi và ánh xạ hình học khi dùng cùng ảnh đã nắn.
- Không có ảnh thiếu nhãn hoặc nhãn không có ảnh trong các split có gán nhãn của hai bộ. Ngoại lệ có chủ ý trong mô tả dữ liệu: VAIPE test không có nhãn, valid trống. File nhãn rỗng vẫn được tính là có file, chưa khẳng định đó là mẫu âm hay thiếu annotation.
- File cần xác nhận: bộ ② `valid/labels/WhatsApp-Image-2022-09-25-at-10-56-11_jpeg.rf.37d622f14f9cfd76663b353b0b8b8732.txt`.
- Đã dùng Pillow `verify()` trên 10.841 file ảnh, không phát hiện lỗi ở mức kiểm tra này. Chưa duyệt từng ảnh/bbox/text bằng mắt, chưa chứng minh không rò rỉ ảnh gần giống và chưa đánh giá chất lượng thuật toán làm sạch.
- Nhãn hiện có không cung cấp mask lớp mực/dấu/chữ ký hoặc ảnh nhị phân chuẩn. Text VAIPE có thể làm tham chiếu OCR sau khi kiểm tra; TXT YOLO của bộ ② chỉ có class/bbox.
- Các đường dẫn `../train/images` trong YAML là đường dẫn tương đối của bản xuất; khi chạy công cụ cần cấu hình đúng thư mục dataset thực tế, không mặc nhiên suy thành `data/train/images`.

Chi tiết từng file và chứng cứ kiểm tra nằm ở `results/primary_dataset_audit/inventory.json`, báo cáo `results/primary_dataset_audit/README.md`; script đọc dữ liệu là `results/audit_primary_datasets.py`. Đây là tài liệu hỗ trợ **chỉ có trên máy Trường, không đi kèm repo**. Các số liệu, đường dẫn bất thường và nguồn tham khảo cần cho nhóm đã được ghi ngay trong hai tài liệu Data để đọc độc lập.

### Các điểm còn cần nhóm xác nhận

1. Cặp trùng VAIPE ghi bên trên thuộc train/test gốc trong gói tải; khi đánh giá dùng split chung đã chốt, không tự chia lại hoặc lấy nguyên split gốc thay thế.
2. TXT rỗng ở bộ ② là mẫu âm có chủ ý hay nhãn còn thiếu.
3. Điều khoản riêng của nguồn VAIPE ban đầu cho gói 2022, nếu cần xác minh ngoài khai báo của người đăng Kaggle.

Các điểm này được ghi nhận để nhóm quyết định; không tự sửa ảnh, nhãn, YAML hoặc phân tập. Việc hoàn thiện tài liệu không đồng nghĩa mọi nhãn đã được duyệt bằng mắt hoặc mọi vấn đề license đã được giải quyết. Xem căn cứ tại [SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md).

## Các bộ khác — thông tin bổ sung

Bộ ⑤ mới khớp toàn bộ checksum trong `checksums_v1.sha256`; đạt hai test dữ liệu của nhóm về ánh xạ khoản thuốc và độ phủ D3/D4. Manifest vẫn ghi `v1`; không suy phiên bản chỉ từ tên thư mục. Kiểm tra này không thay thế duyệt nhãn bằng mắt.

### ④ Ảnh tự chụp — `4_self_captured/`

Cập nhật cấu trúc ngày **03/11/2026**. Bản hiện tại có 83 ảnh: 48 phiếu xét nghiệm, 18 đơn thuốc và 17 tài liệu khác; đã tải lên Drive nhóm cùng `manifest.csv`. Ảnh dùng ID ổn định liên tục theo loại `sc_<category>_<number>.<ext>` và sắp theo `lab_report/`, `prescription/`, `other/`.

Đây là bộ ảnh **chưa gán nhãn chi tiết**: chưa có box, transcript OCR, dòng/cột, khoản thuốc hoặc nhãn thành phần. Thư mục chỉ nhóm ảnh theo loại tài liệu đã biết. `manifest.csv` để trống thông tin người đóng góp, nhóm tài liệu trùng, split; `privacy_review` đang là `required` và `annotation_status` là `not_started` cho mọi ảnh. Bản hiện tại đã được tải lên Drive nhóm, nhưng chưa xác nhận bằng duyệt nội dung rằng toàn bộ ảnh đã được ẩn danh. Người phụ trách từng ảnh cần hoàn tất review trước khi dùng ảnh cho benchmark.

Quy ước dùng bộ 4:
- Giữ nguyên ảnh nguồn sau khi đã ẩn danh; không resize hay nén lại khi chưa có lý do và ghi nhận rõ.
- Mỗi người điền `contributor_id`; cùng một tài liệu chụp nhiều lần phải dùng chung `document_group_id`.
- **Quyết định 09/11:** bộ 4 không gán nhãn chi tiết. Nhãn duy nhất là loại tài liệu theo thư mục (dùng cho D10); các bước khác dùng ảnh bộ 4 làm ảnh trước / sau, kiểm tra định tính ([kế hoạch](../team/KeHoach.md) mục 3.2). Bộ 4 là dữ liệu kiểm thử độc lập, không đưa vào train/tinh chỉnh.
- Không ghi đè manifest hoặc ảnh nguồn. Việc từng người làm trên bộ 4: [team/TienDo.md](../team/TienDo.md) mục "Bộ 4".
- `data/` bị Git ignore. Chia sẻ ảnh, manifest và nhãn qua Drive nhóm; tài liệu này là hướng dẫn được theo dõi bằng Git.

## Nguồn phát triển đang giữ riêng trên máy Trường

- `6_synthetic_v1_old/6_synthetic/`: đúng dữ liệu đo T1–T2 và nguồn của 5 ảnh trong `img_test`. Không thay bằng bộ ⑤ mới khi chạy lại kết quả cũ.
- `PhieuXetNghiem/report keywords detection.v1i.yolov8/`: bản 57 ảnh, còn là nguồn của 5 ảnh xét nghiệm trong `img_test`. Bản mới có quy mô/biến thể khác nên giữ nguồn gốc này.

Các nguồn cũ trên chỉ phục vụ tái lập thử nghiệm đã có, không thay bộ chuẩn của nhóm. Nhật ký dọn dữ liệu và bảng ánh xạ VAIPE cũ → mới nằm ở `results/data_sync_cleanup/` trên máy Trường, không đi kèm repo.
