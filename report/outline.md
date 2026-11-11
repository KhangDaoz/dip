# Trích xuất thông tin đơn thuốc từ ảnh: học sâu kết hợp xử lý ảnh cổ điển

Dàn ý báo cáo v2 (theo [plan.md](../plan.md) §10). Bản chính ≤ 10 trang.
Số liệu chờ điền đánh dấu `\KQ{...}` (chữ xanh), đoạn văn chờ viết đánh dấu `\TODO{...}` (chữ đỏ).
Tìm chỗ còn trống: `grep -n 'KQ{\|TODO{' report/latex/sections/*.tex report/latex/appendix/*.tex`.

| Mục | File | Nội dung | Số liệu cần | Người |
| :-- | :-- | :-- | :-- | :-- |
| 1. Đặt vấn đề | `1_introduction.tex` | Bối cảnh MedTrack AI, phát biểu bài toán (ảnh → JSON), hướng tiếp cận và đóng góp | F1 khoản (E-D2), nguồn trích bối cảnh | Hiếu |
| 2. Dữ liệu | `2_data.tex` | VAIPE 6 lớp, chia 820/176/176 theo bệnh nhân, đáp án khoản thuốc, ảnh suy giảm L0/L2/L4 | số khoản test, % OOV, số ảnh soát | Hiếu |
| 3. Phương pháp | `3_0_overview.tex` | Sơ đồ A→D (TikZ), hợp đồng dữ liệu, bảng chương học phần | | Hiếu |
| 3.1 Tiền xử lý ảnh (khối A) | `3_1_enhance.tex` | 5 bước, công thức chỉnh nghiêng và chia nền | lý do chọn tham số | Hiếu |
| 3.2 Phát hiện vùng chữ (khối B) | `3_2_detect.tex` | YOLOv8s, cấu hình, ngưỡng 0,443, đối chứng cổ điển | đã điền | Trường |
| 3.3 Nhận dạng chữ (khối C) | `3_3_recognize.tex` | VietOCR, tinh chỉnh 20k vòng, độ rộng 1024, đối chứng C2, C0, C1 | đã điền | Khang |
| 3.4 Ghép khoản thuốc (khối D) | `3_4_assemble.tex` | Chọn lớp, dải ngang, chuẩn hóa, từ điển, đối chứng VLM | số tên từ điển, tên mô hình VLM | Thắng |
| 4.1 Thiết lập và độ đo | `4_1_setup.tex` | Phần cứng, IoU/P/R/F1, CER/WER, F1 khoản, bootstrap | đã điền | Hiếu |
| 4.2 Phát hiện vùng chữ | `4_2_detect.tex` | Bảng E-B1 (bỏ lớp) và E-B2 (6 lớp) | đã điền (E-B1, E-B2) | Trường |
| 4.3 Nhận dạng chữ | `4_3_recognize.tex` | Bảng E-C1. C2 và C3 toàn tập đã điền | đã điền (E-C1) | Khang |
| 4.4 So sánh đầu-cuối | `4_4_endtoend.tex` | E-D1 và bảng chính E-D2 | toàn bảng | Thắng |
| 4.5 Phân tích lỗi và độ bền | `4_5_ablation.tex` | Bảng E-X1, hình E-A1 theo mức suy giảm | E-A2, E-A3 đã điền. Còn bảng E-X1 và hình E-A1 | Hiếu |
| 4.6 Thời gian chạy | `4_6_runtime.tex` | Giây/ảnh từng khối | đã điền (khối A, B, C đo, khối D ước lượng) | Hiếu |
| 5. Thảo luận | `5_discussion.tex` | Cổ điển và học sâu, VLM, hạn chế, hướng phát triển (không chia mục con) | dẫn số từ mục 4 | Hiếu, Thắng |
| 6. Kết luận | `5_1_conclusion.tex` | 4 đến 5 câu | F1 các dòng chính | Hiếu |
| 7. Phân công | `6_contribution.tex` | Bảng phân công | | Hiếu |
| 8. Khai báo AI | `7_ai_declaration.tex` | Công cụ, phần tự làm của từng người, cách kiểm tra | số bài pytest | Cả nhóm |
| Phụ lục A | `appendix/A_params.tex` | Bảng tham số đầy đủ | các ô còn trống | Mỗi người dòng của mình |
| Phụ lục B | `appendix/B_examples.tex` | Ảnh kết quả mẫu (đã có), ma trận nhầm lẫn, ca lỗi | hình | Trường, Khang, Thắng |

Bản chính hiện khoảng 8,5 trang khi còn chỗ trống, giới hạn là 10 trang. Không dùng tiểu mục cấp 3.

Hình đặt trong `report/latex/figure/{common,enhance,detect,recognize,assemble}/`.
