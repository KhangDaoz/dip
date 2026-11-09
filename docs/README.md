# TRA CỨU LÝ THUYẾT — MÔN XỬ LÝ ẢNH SỐ (DIP)

> [!NOTE]
> Trang này dùng để **tra cứu lý thuyết**: đối chiếu chương bài giảng với bước của pipeline, và liệt kê notebook thực hành. Tổng quan đề tài ở [DeTai_XuLyAnh_ChiTiet.md](DeTai_XuLyAnh_ChiTiet.md). **Bảng tra đầy đủ kỹ thuật → slide → công thức** và phân công nằm ở [kế hoạch toàn nhóm](../team/KeHoach.md) mục 4 và 9; tiến độ ở [team/TienDo.md](../team/TienDo.md).

> **Môn học:** Xử lý ảnh số (Digital Image Processing), Khoa Trí tuệ Nhân tạo, Học viện Công nghệ Bưu chính Viễn thông  
> **Giảng viên hướng dẫn:** PGS.TS. Phạm Văn Cường  
> **Hạn nộp đồ án:** **11/11/2026** (gửi email `uyenvnb@ptit.edu.vn`)

> [!IMPORTANT]
> Các bài giảng đã được chuyển sang Markdown để dễ tra cứu, nhưng **ảnh minh họa chưa được nhúng** (các link `images/...` trong file `.md` không có ảnh đi kèm). Khi cần xem hình, sơ đồ hoặc ví dụ trực quan, hãy mở **file PDF gốc** tương ứng trong bảng ở mục 2.

---

## 1. YÊU CẦU ĐỒ ÁN VÀ ĐỀ TÀI

| Tài liệu | File |
| :--- | :--- |
| Tổng quan đề tài của nhóm | [DeTai_XuLyAnh_ChiTiet.md](DeTai_XuLyAnh_ChiTiet.md) |
| Kế hoạch chính (phương pháp, thí nghiệm, lịch) | [KeHoach.md](../team/KeHoach.md) |
| Quy định đồ án của thầy (Markdown) | [DoAn_XuLyAnh.md](references/requirements/DoAn_XuLyAnh.md) |
| Quy định đồ án của thầy (PDF gốc) | [DoAn_XuLyAnh.pdf](references/requirements/DoAn_XuLyAnh.pdf) |

Thông tin chính:

* **Tên đề tài:** *Nghiên cứu & Xây dựng Pipeline Tiền xử lý, Phân loại biểu mẫu và Bóc tách cấu trúc tài liệu y tế từ ảnh chụp thực tế*.
* **Hình thức:** **Lựa chọn 2, Dự án xử lý ảnh thực tế** (pipeline nhiều bước, có đối chứng và đánh giá định lượng). Nhóm 4 thành viên: Hiếu, Trường, Khang, Thắng.
* **Gói nộp** (một file `.zip` gửi `uyenvnb@ptit.edu.vn`, chi tiết ở mục 7 của đề tài):
  1. Mã nguồn `code/`, `requirements.txt`, ảnh mẫu trong `samples/`.
  2. `README.md` hướng dẫn chạy.
  3. Báo cáo PDF 4–10 trang.
  4. Slide 10–15 trang (`.pptx` và `.pdf`).
  5. Video demo tối đa 3 phút.

---

## 2. BÀI GIẢNG VÀ ÁNH XẠ VÀO PIPELINE

Cột "Áp dụng trong đề tài" tóm tắt từ ma trận chương ở mục 9 của [kế hoạch toàn nhóm](../team/KeHoach.md). Đề tài áp dụng **8/8 chương**, trục chính là Chương 3, 5, 7.

| Chương | Nội dung cốt lõi | Markdown | PDF gốc | Áp dụng trong đề tài |
| :---: | :--- | :---: | :---: | :--- |
| **01** | **Nhập môn Xử lý ảnh số**: khái niệm, các bước cơ bản của một hệ thống xử lý ảnh. | [md](references/lectures/chuong1_gioi_thieu.md) | [pdf](references/lectures/chuong1_gioi_thieu.pdf) | Các giai đoạn xử lý ảnh (C1 · S27–34) làm khung ánh xạ pipeline. |
| **02** | **Cơ bản về Xử lý ảnh**: lấy mẫu, lượng tử hóa, biểu diễn ảnh, biến đổi hình học (Affine, xoay, nội suy). | [md](references/lectures/chuong2_co_ban_ve_xla.md) | [pdf](references/lectures/chuong2_co_ban_ve_xla.pdf) | ①: nội suy khi nắn / xoay; nắn phối cảnh (theo gợi ý đề tài); độ phân giải → tham số theo chiều cao chữ. |
| **03** | **Nâng cao chất lượng ảnh**: biến đổi độ sáng, cân bằng Histogram, CLAHE, lọc không gian (Gauss, Median, Laplace). | [md](references/lectures/chuong3_nang_cao_chat_luong_anh.md) | [pdf](references/lectures/chuong3_nang_cao_chat_luong_anh.pdf) | ①: lọc trung vị, cân bằng lược đồ (đối chứng CLAHE). ②: lọc Max/Min ước lượng nền, toán tử điểm khử bóng. ③④: lược đồ xám là cơ sở của projection profile, chiều cao chữ = trung vị. |
| **04** | **Các hệ màu & Kỹ thuật in ấn**: RGB, CMYK, HSI/HSV, Lab, YCbCr; đặc tính mực in. | [md](references/lectures/chuong4_cac_he_mau_va_ky_thuat_in_anh.md) | [pdf](references/lectures/chuong4_cac_he_mau_va_ky_thuat_in_anh.pdf) | ①: lóa = V cao, S thấp. ②: tách lớp mực HSV (dấu đỏ, bút xanh, chữ in đen), giữ chữ đỏ cảnh báo. Bộ 5: suy biến JPEG. |
| **05** | **Phát hiện biên & Phân vùng ảnh**: Sobel, Prewitt, Canny; ngưỡng toàn cục / cục bộ. | [md](references/lectures/chuong5_phat_hien_bien.md) | [pdf](references/lectures/chuong5_phat_hien_bien.pdf) | ①: Laplacian đo mờ, Canny tìm mép giấy. ②: Otsu (chính), ISODATA, Tam giác, Đối xứng nền (đối chứng). ③④: làm trơn lược đồ, tách ngưỡng tại thung lũng, vùng liên thông, tách / hợp vùng (XY-cut). |
| **06** | **Nén ảnh**: Huffman, RLE, LZW; DCT, JPEG. | [md](references/lectures/chuong6_nen_anh.md) | [pdf](references/lectures/chuong6_nen_anh.pdf) | Đánh giá chất lượng: RMS → PSNR (①②), thang chủ quan (kiểm chứng bộ 5); giải thích suy biến JPEG trong bộ 5. |
| **07** | **Xử lý hình thái học**: co, giãn, mở, đóng, Hit-or-Miss, thành phần liên thông. | [md](references/lectures/chuong7_xu_ly_hinh_thai.md) | [pdf](references/lectures/chuong7_xu_ly_hinh_thai.pdf) | **Trọng tâm.** ②: Closing ước lượng nền, Opening hậu xử lý, tách biên + làm đầy vùng lấy mask dấu. ③: Dilation ngang gom cụm từ; mở rộng: Opening lấy đường kẻ bảng. ④: Dilation / Closing dọc gom khoản thuốc. |
| **08** | **Nhận dạng ảnh**: trích chọn đặc trưng, phân cụm và phân loại (K-means, kNN, BoVW), mạng nơ-ron. | [md](references/lectures/chuong8_nhan_dang_anh.md) | [pdf](references/lectures/chuong8_nhan_dang_anh.pdf) | ①: phân loại 3 lớp bằng HOG / GLCM / mật độ mực + $k$-NN, từ chối tài liệu lạ. Cả nhóm: Precision / Recall / F1, ma trận nhầm lẫn; độ tròn (②). |

---

## 3. BÀI TẬP THỰC HÀNH

Tổng quan: [references/practices/README.md](references/practices/README.md). Các notebook viết bằng Python (`OpenCV`, `NumPy`, `Matplotlib`).

| Chương | Notebook | Thuật toán đã cài đặt | Dùng lại trong pipeline |
| :--- | :--- | :--- | :--- |
| **1** | [`B23DCVT218_DaoVanKhang_Chuong1.ipynb`](references/practices/Chuong_1/B23DCVT218_DaoVanKhang_Chuong1.ipynb) | Ma trận ảnh, chuyển BGR $\leftrightarrow$ RGB, lọc nhiễu và phân ngưỡng sơ bộ. | Nạp ảnh, hiển thị lưới so sánh trước/sau xử lý. |
| **2** | [`B23DCVT218_DaoVanKhang_Chuong2.ipynb`](references/practices/Chuong_2/B23DCVT218_DaoVanKhang_Chuong2.ipynb) | Lấy mẫu không gian, lượng tử hóa mức xám, tự cài đặt Histogram. | Nền tảng cho nắn phối cảnh và chỉnh nghiêng. |
| **3** | [`B23DCVT218_DaoVanKhang_Chuong3.ipynb`](references/practices/Chuong_3/B23DCVT218_DaoVanKhang_Chuong3.ipynb) | **CLAHE**, Gamma/Log, lọc sắc nét Laplacian, lọc trung vị. | ①: đo mờ bằng phương sai Laplacian, CLAHE. |
| **4** | [`B23DCVT218_DaoVanKhang_Chuong4.ipynb`](references/practices/Chuong_4/B23DCVT218_DaoVanKhang_Chuong4.ipynb) | Không gian màu **HSV**, độ bền kênh V với ánh sáng, phân ngưỡng màu, dithering. | ①: phát hiện lóa trên kênh V. ②: tách lớp mực theo màu. |
| **7** | [`B23DCVT218_DaoVanKhang_Chuong7.ipynb`](references/practices/Chuong_7/B23DCVT218_DaoVanKhang_Chuong7.ipynb) | Co, giãn, **Closing**, tách biên, **thành phần liên thông**. | ③④: gom ký tự bằng SE ngang / dọc; ②: Closing ước lượng nền, thành phần liên thông. |
