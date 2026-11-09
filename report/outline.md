# Pipeline tiền xử lý, phân loại và bóc tách cấu trúc tài liệu y tế từ ảnh chụp điện thoại

## 1. Đặt vấn đề
Bối cảnh số hóa hồ sơ y tế giấy; khó khăn của ảnh chụp điện thoại (méo, nghiêng, mờ, lóa, bóng, dấu đè chữ, bảng không kẻ); đầu vào, đầu ra của bài toán; đóng góp của nhóm.

## 2. Dữ liệu
### 2.1. Các bộ dữ liệu
Bộ 1–5: nguồn, số ảnh gốc, loại nhãn, mục đích sử dụng; bảng tập đánh giá ≥ 50 ảnh của từng người.
### 2.2. Chia tập và chống rò rỉ
Chia theo ảnh gốc; bộ 2 chỉ dùng 57 ảnh gốc, không dùng bản tăng cường của Roboflow; tham số chọn trên val, báo cáo trên test; riêng phần dựng bảng dùng kiểm định chéo 5-fold trên 57 ảnh bộ 2.
### 2.3. Bộ 5 — benchmark tổng hợp
Cách sinh từ ảnh sạch, các loại và mức hỏng, ground truth đi kèm, mức độ giống ảnh thật.

## 3. Phương pháp
### 3.0. Tổng quan pipeline
Sơ đồ 4 bước, dữ liệu chuyển giữa các bước, ma trận chương học phần (8/8 chương, trục chính C3, C5, C7).
*Ghi chú:* phải có bước **Hậu xử lý** theo khung pipeline Slide 7 (Dữ liệu → Tiền xử lý → Xử lý chính → Hậu xử lý → Đánh giá); bảng chương học phần ↔ kỹ thuật thêm cột Hậu xử lý.
### 3.1. Tiền xử lý và phân loại tài liệu
Khử nhiễu, đo chất lượng (mờ, lóa), tìm 4 góc, nắn phối cảnh, chỉnh nghiêng, tăng tương phản, khử lóa; phân loại 3 lớp bằng đặc trưng + kNN.
*Ghi chú:* phải viết 1–2 câu về bước hậu xử lý.
### 3.2. Làm sạch ảnh và tách lớp mực
Ước lượng nền và khử bóng, tách mực màu, nhận dạng dấu theo độ tròn, xóa dấu / chữ ký (đơn thuốc), giữ chữ đỏ cảnh báo (phiếu xét nghiệm), nhị phân hóa.
*Ghi chú:* phải viết 1–2 câu về bước hậu xử lý.
### 3.3. Dựng bảng phiếu xét nghiệm
Bảng không kẻ: chiều cao chữ, tách cụm từ, gom dòng, tìm cột, ghép ô; vai trò cột và cờ bất thường bằng luật (hậu xử lý văn bản).
*Ghi chú:* phải viết 1–2 câu về bước hậu xử lý.
### 3.4. Bóc tách đơn thuốc
Khoanh vùng thuốc, tách dòng, gom thành từng khoản thuốc; tách trường bằng luật (hậu xử lý văn bản).
*Ghi chú:* phải viết 1–2 câu về bước hậu xử lý.

## 4. Thực nghiệm
### 4.1. Thiết lập và độ đo
Môi trường chạy, độ đo dùng chung (PSNR, SSIM, F-measure, F1, IoU, CER).
### 4.2. Tiền xử lý và phân loại
So sánh các phương án tìm góc, chỉnh nghiêng, khử lóa; sai số nắn theo mức méo; kết quả phân loại.
### 4.3. Làm sạch và tách lớp mực
So sánh các cách nhị phân hóa; độ bền theo mức hỏng; ràng buộc giữ chữ đỏ.
### 4.4. Dựng bảng
Độ chính xác gán cụm từ vào dòng, cột, F1 quan hệ kề; độ bền theo góc nghiêng; tham số theo chiều cao chữ vs cố định.
### 4.5. Bóc tách đơn thuốc
Độ chính xác tách khoản thuốc; độ bền theo mức hỏng; lợi ích cho OCR.
### 4.6. Ablation toàn pipeline
Tắt lần lượt từng bước, đo ảnh hưởng lên kết quả cuối.
### 4.7. Thời gian chạy
Thời gian mỗi bước và toàn pipeline.

## 5. Thảo luận
Điểm mạnh, hạn chế, ca thất bại điển hình, hướng phát triển.

## 6. Phân công
Phần việc và mức đóng góp của từng thành viên.

## 7. Khai báo sử dụng AI
Công cụ đã dùng, dùng cho việc gì, cách kiểm tra kết quả.

## Tài liệu tham khảo

## Phụ lục
Bảng tham số đầy đủ, đường cong theo tham số, ảnh ví dụ bổ sung.
