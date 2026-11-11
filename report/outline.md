# Trích xuất thông tin đơn thuốc từ ảnh: học sâu kết hợp xử lý ảnh cổ điển

Dàn ý báo cáo v2 (theo [plan.md](../plan.md) §10). Bản chính ≤ 10 trang, hiện khung ≈ 9,5 trang khi còn chỗ trống.
Số liệu chờ điền đánh dấu `\KQ{...}` (chữ xanh), đoạn văn chờ viết đánh dấu `\TODO{...}` (chữ đỏ).
Tìm chỗ còn trống: `grep -n 'KQ{\|TODO{' report/latex/sections/*.tex report/latex/appendix/*.tex`.

| Mục | File | Nội dung | Số liệu cần | Người |
| :-- | :-- | :-- | :-- | :-- |
| 1. Đặt vấn đề | `1_introduction.tex` | Bài toán ảnh → JSON, lý do đổi hướng từ v1, đóng góp | F1 khoản (E-D2) | Hiếu |
| 2. Dữ liệu | `2_data.tex` | VAIPE 6 lớp, chia 820/176/176 theo bệnh nhân, đáp án khoản thuốc, bộ suy giảm L0/L2/L4 | số khoản test, % OOV, số ảnh soát | Hiếu |
| 3.0 Tổng quan | `3_0_overview.tex` | Sơ đồ A→D (TikZ), hợp đồng dữ liệu, bảng chương học phần | — | Hiếu |
| 3.1 Khối A | `3_1_enhance.tex` | 5 bước + công thức chỉnh nghiêng, khử bóng | — | Hiếu |
| 3.2 Khối B | `3_2_detect.tex` | YOLOv8s, cấu hình + lý do, ngưỡng 0,443; đối chứng cổ điển | — | Trường |
| 3.3 Khối C | `3_3_recognize.tex` | VietOCR `vgg_transformer`, tinh chỉnh 20k vòng, max width 1024; đối chứng C2 (C0/C1 nếu có) | — | Khang |
| 3.4 Khối D | `3_4_assemble.tex` | Chọn lớp, dải ngang, chuẩn hóa, từ điển; đối chứng VLM | số tên trong từ điển | Thắng |
| 4.1 Thiết lập | `4_1_setup.tex` | Phần cứng, IoU/P/R/F1, CER/WER, F1 khoản, bootstrap | phiên bản thư viện | Hiếu |
| 4.2 E-B1, E-B2 | `4_2_detect.tex` | Bảng P/R/F1/mAP50 (bỏ lớp + 6 lớp) | toàn bảng | Trường |
| 4.3 E-C1 | `4_3_recognize.tex` | Bảng CER/WER/khớp, CER theo lớp, ms/vùng | C2, C3 toàn tập **đã có**; cột theo lớp, thời gian | Khang |
| 4.4 E-D1, E-D2 | `4_4_endtoend.tex` | **Bảng chính** 5–7 dòng | toàn bảng | Thắng |
| 4.5 E-X1, E-A1 | `4_5_ablation.tex` | Bảng thay khối bằng đáp án; hình F1 theo mức suy giảm | toàn bảng + hình | Hiếu |
| 4.6 Thời gian | `4_6_runtime.tex` | Giây/ảnh từng khối | toàn bảng | Hiếu |
| 5. Thảo luận | `5_discussion.tex` | Cổ điển vs DL, DL vs VLM, hạn chế, hướng phát triển | dẫn số từ mục 4 | Hiếu + Thắng |
| 6. Kết luận | `5_1_conclusion.tex` | 4–5 câu | F1 các dòng chính | Hiếu |
| 7. Phân công | `6_contribution.tex` | Bảng | — | Hiếu |
| 8. Khai báo AI | `7_ai_declaration.tex` | Công cụ, phạm vi, mỗi người 1–2 câu | số bài pytest | Cả nhóm |
| Phụ lục A | `appendix/A_params.tex` | Bảng tham số đầy đủ | — | Mỗi người dòng của mình |
| Phụ lục B | `appendix/B_examples.tex` | Ảnh overlay (đã có), ma trận nhầm lẫn, ca lỗi | hình | Trường, Khang, Thắng |

Hình đặt trong `report/latex/figure/{common,enhance,detect,recognize,assemble}/`.
