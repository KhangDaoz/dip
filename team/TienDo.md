# BẢNG TIẾN ĐỘ — NHÓM MEDTRACK

> **Kế hoạch chính:** [KeHoach.md](KeHoach.md). Bảng này chỉ theo dõi trạng thái, nội dung từng việc xem mục được ghi ở cột "Mục".  
> **Cách cập nhật:** sửa cột **Trạng thái** (⬜ Chưa làm · 🔄 Đang làm · ⛔ Bị chặn · ✅ Xong), ghi đường dẫn kết quả hoặc lý do bị chặn vào **Ghi chú**, rồi commit lên `main`.  
> **Code:** khung skeleton ở [code/README.md](../code/README.md); mọi hàm còn `NotImplementedError`.

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
| Hiếu | Đặc trưng + kNN, D10; D3; chọn 10 ảnh mẫu `code/samples/` | 5, 12.4 | ⬜ | |
| Trường | Nối các bước ② đầu-cuối; T5; thêm ISODATA, Tam giác vào T2 | 6, 12.4 | ⬜ | |
| Khang | GT 57 ảnh bộ 2 (soát bằng mắt); K1, K2 bằng 5-fold trên 57 ảnh | 7, 12.4 | ⬜ | |
| Thắng | Chia VAIPE theo cụm pHash; GT tự gom cho ~150–200 ảnh chọn tham số; GT ≥ 50 ảnh đánh giá (soát bằng mắt); S1, S3 | 8, 12.4 | ⬜ | |
| Cả nhóm | 🎯 **Tối: chạy cả pipeline trên 10 ảnh mẫu** (`python run_pipeline.py samples/`) | 12.4 | ⬜ | |

## 10/11 (Thứ ba)

| Người | Sáng | Chiều | Tối | Trạng thái | Ghi chú |
| :--- | :--- | :--- | :--- | :---: | :--- |
| Hiếu | Ghép báo cáo, slide | Quay đoạn demo; ghép video | Đóng gói; một bạn cài lại trên máy sạch | ⬜ | |
| Trường | T8 ablation; phần báo cáo + 2 slide | Quay đoạn demo | Đọc chéo báo cáo | ⬜ | |
| Khang | K3, K4, K5; phần báo cáo + 2 slide | Quay đoạn demo | Đọc chéo báo cáo | ⬜ | |
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
| Khang | Chọn 3–5 phiếu, xuất ảnh overlay dòng / cột | K6 (định tính) | ⬜ |
| Thắng | Chọn 3–5 đơn, xuất ảnh khung từng khoản | S5 (định tính) | ⬜ |

## Thứ tự cắt khi thiếu thời gian (mục 12.5)

| Người | Cắt trước (từ trái sang) | Không cắt |
| :--- | :--- | :--- |
| Hiếu | D9 → D5 → GLCM | D2, D3, D6, D10, ghép và nộp |
| Trường | T6 → Đối xứng nền → ISODATA / Tam giác | T1, T2, T3, T5, T8 |
| Khang | K8, K9 (mở rộng) → K6 → K3 | GT 57 ảnh, K1, K2, K4, K5 |
| Thắng | S6 → S5 → S4 → Closing dọc | GT ≥ 50 ảnh, S1, S2, S3 |
