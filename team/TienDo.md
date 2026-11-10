# BẢNG TIẾN ĐỘ — NHÓM MEDTRACK

> **Kế hoạch chính:** [KeHoach.md](KeHoach.md). Bảng này chỉ theo dõi trạng thái, nội dung từng việc xem mục được ghi ở cột "Mục".  
> **Cách cập nhật:** sửa cột **Trạng thái** (⬜ Chưa làm · 🔄 Đang làm · ⛔ Bị chặn · ✅ Xong), ghi đường dẫn kết quả hoặc lý do bị chặn vào **Ghi chú**, rồi commit lên `main`.  
> **Code:** [code/README.md](../code/README.md). Ngày 10/11 đã viết đủ các module ①–④ (25/25 test qua), chạy các thí nghiệm không cần OCR; kết quả ở `results/<mã>/`, đã điền vào báo cáo `report/latex/`.

## Hạn chót

| Ngày | Hạn |
| :---: | :--- |
| **09/11 tối** | 🎯 Chạy cả pipeline trên 10 ảnh mẫu |
| **10/11 sáng** | Mỗi người gửi phần báo cáo + 2 slide |
| **10/11 tối** | Đóng gói `.zip`, thử trên máy sạch |
| **11/11** | 📮 Nộp đồ án qua email `uyenvnb@ptit.edu.vn` |

## 09/11 (Thứ hai)

| Người | Việc | Mục | Trạng thái | Ghi chú |
| :--- | :--- | :---: | :---: | :--- |
| Hiếu | Đặc trưng + kNN, D10; D3; chọn 10 ảnh mẫu `code/samples/` | 5, 12.4 | ✅ | D1 (AUC Laplacian 0,979), D2 (tìm góc 99 %), D3 (đường cong), D4 (projection sai số 0°), D9 (< 0,7 s/ảnh), D10 (test 0,94; bộ 4 chỉ 0,39 — học thuộc nguồn); 10 ảnh mẫu; mô hình `code/models/doc_knn.pkl`. **Còn:** D5, D6 (cần OCR) |
| Trường | Nối các bước ② đầu-cuối; T5; thêm ISODATA, Tam giác vào T2 | 6, 12.4 | 🔄 | `clean()` chạy đầu-cuối; T1 (Closing 81: 47,9 dB), T2 7 cách (Sauvola 0,922 bền với bóng, Otsu tụt 0,849 khi không khử bóng), T3 đường cong, T4 (tìm 95 % dấu). **Còn:** T5 (soát chữ đỏ bộ 4), T2-CER, T8 (cần OCR) |
| Khang | GT 57 ảnh bộ 2 (soát bằng mắt); K1, K2 bằng 5-fold trên 57 ảnh | 7, 12.4 | ✅ | 10/11: soát 55 ảnh, giữ **32 ảnh** GT (`code/experiments/k0_review.json`, 9 ảnh sửa dòng); chữ GT do LLM phiên âm + soát (`k0_text_gt.json`). Chạy lại trên 32 ảnh: cột projection 0,963; dòng projection 0,705 → **0,937 sau xóa đường kẻ** (tâm y 0,990, nhưng ít ảnh đúng hết hơn); tách cụm từ 0,748; K4 không khác biệt |
| Thắng | Chia VAIPE 70 / 15 / 15; GT tự gom cho valid + test; GT ≥ 50 ảnh test (soát bằng mắt); S1, S3 | 8, 12.4 | 🔄 | Đã chia VAIPE; S0 GT tự gom (`results/s0/review/test/` — **cần soát ≥ 50 ảnh**); S1 F1 0,973; S4 khoanh vùng 0,820; S2 đầu-cuối bộ 5 (0,725, bóng là ca lỗi chính); S6 mask dải viền + 99,4 % bất đồng Otsu/Sauvola nằm trong dải viền. **Còn:** soát GT, S3 (cần OCR) |
| Cả nhóm | 🎯 **Tối: chạy cả pipeline trên 10 ảnh mẫu** (`python run_pipeline.py samples/`) | 12.4 | ✅ | 10/11: chạy hết 10 ảnh không lỗi; phân loại đúng 7/10; ảnh `out/*_overlay.png` dùng cho demo |

## 10/11 (Thứ ba)

| Người | Sáng | Chiều | Tối | Trạng thái | Ghi chú |
| :--- | :--- | :--- | :--- | :---: | :--- |
| Hiếu | Ghép báo cáo, slide | Quay đoạn demo; ghép video | Đóng gói; một bạn cài lại trên máy sạch | ⬜ | |
| Trường | T8 ablation; phần báo cáo + 2 slide | Quay đoạn demo | Đọc chéo báo cáo | ⬜ | |
| Khang | K3, K4, K5; phần báo cáo + 2 slide | Quay đoạn demo | Đọc chéo báo cáo | 🔄 | ✅ K3 (không chỉnh nghiêng sụp từ 1°; có ① giữ 0,87–0,92 đến 5°), K4, K5 (CER cả trang 0,765 → theo ô 0,470; khung GT 0,216); báo cáo 3.3, 4.4, phụ lục C, 2.1, 2.2, 5, 7 đã cập nhật. **Còn:** 2 slide, demo |
| Thắng | S2; phần báo cáo + 2 slide | Quay đoạn demo | Đọc chéo + thử máy sạch | ⬜ | |

## 11/11 (Thứ tư) — HẠN NỘP

| Người | Việc | Trạng thái |
| :--- | :--- | :---: |
| Hiếu | 📮 Nộp `.zip` qua email | ⬜ |
| Trường, Khang, Thắng | Tập thuyết trình, hỏi đáp thử (mục "Hỏi đáp" của phần mình) | ⬜ |

## Bộ 4 — chỉ để test, không gán nhãn chi tiết

Quyết định 09/11 (kế hoạch mục 3.2): bộ 4 **không gán nhãn chi tiết**. Chỉ D10 đo định lượng, nhãn lấy theo thư mục; mọi phần khác dùng **ảnh trước / sau, kiểm tra định tính**. Ảnh nào đưa vào báo cáo, video, `samples/` vẫn phải che họ tên, ngày sinh, mã bệnh nhân, số điện thoại.

| Người | Việc trên bộ 4 | Dùng cho | Trạng thái |
| :--- | :--- | :--- | :---: |
| Hiếu | Soát nhanh 83 ảnh nằm đúng thư mục loại tài liệu | D10 (định lượng) | ⬜ |
| Trường | Chọn ≥ 5 phiếu có chữ đỏ, xuất lưới ảnh trước / sau | T5 (định tính) | ⬜ |
| Khang | Chọn 3–5 phiếu, xuất ảnh overlay dòng / cột | K6 (định tính) | ✅ 5 phiếu (`results/k6/`, có thông tin BN — không commit): cột gần đúng, dòng sụp cả 5 (kẻ cong, nét tay) |
| Thắng | Chọn 3–5 đơn, xuất ảnh khung từng khoản | S5 (định tính) | ⬜ |

## Thứ tự cắt khi thiếu thời gian (mục 12.5)

| Người | Cắt trước (từ trái sang) | Không cắt |
| :--- | :--- | :--- |
| Hiếu | D9 → D5 → GLCM | D2, D3, D6, D10, ghép và nộp |
| Trường | T6 → Đối xứng nền → ISODATA / Tam giác | T1, T2, T3, T5, T8 |
| Khang | K8, K9 (mở rộng) → K6 → K3 | GT 57 ảnh, K1, K2, K4, K5 |
| Thắng | S6 → S5 → S4 → Closing dọc | GT ≥ 50 ảnh, S1, S2, S3 |
