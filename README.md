# Đồ án Xử lý ảnh số — Pipeline xử lý ảnh tài liệu y tế (MedTrack)

**Đề tài:** Nghiên cứu & xây dựng pipeline tiền xử lý, phân loại biểu mẫu và bóc tách cấu trúc tài liệu y tế (đơn thuốc, phiếu xét nghiệm) từ ảnh chụp thực tế.

Môn Xử lý ảnh số (DIP), PGS.TS. Phạm Văn Cường, Học viện Công nghệ Bưu chính Viễn thông. Hình thức: **Lựa chọn 2, Dự án xử lý ảnh thực tế**.
Nhóm: Hiếu (nhóm trưởng) · Trường · Khang · Thắng.

**Kế hoạch chính** (phương pháp từng bước, thí nghiệm, tham số, lịch, checklist): [team/KeHoach.md](team/KeHoach.md). Tổng quan đề tài: [docs/DeTai_XuLyAnh_ChiTiet.md](docs/DeTai_XuLyAnh_ChiTiet.md). Hai file khác nhau thì theo kế hoạch toàn nhóm; sửa kế hoạch phải thống nhất cả nhóm.

## Pipeline

| Giai đoạn | Người | Nội dung | Module |
| :--- | :--- | :--- | :--- |
| ① Tiền xử lý & phân loại | Hiếu | Đo mờ (Laplacian) / lóa (HSV), Canny → 4 góc, nắn phối cảnh, chỉnh nghiêng (projection), CLAHE; đặc trưng + $k$-NN 3 lớp | `code/src/preprocessing/` · `code/src/classification/` |
| ② Làm sạch & tách lớp mực | Trường | Khử bóng (Closing ước lượng nền), tách mực HSV, nhận dấu theo độ tròn, xóa dấu (đơn thuốc), nhị phân Sauvola (đối chứng Otsu) + Opening | `code/src/cleaning/` |
| ③ Dựng bảng phiếu xét nghiệm | Khang | Bảng không kẻ: Dilation ngang → cụm từ, projection → dòng, projection dọc / XY-cut → cột, ghép ô | `code/src/table_extraction/` |
| ④ Bóc tách đơn thuốc | Thắng | Dilation khoanh vùng thuốc, projection tách dòng, Dilation dọc gom khoản | `code/src/prescription_segmentation/` |
| Dùng chung | Khang · Thắng | Bộ OCR (công cụ đo CER) · bộ sinh benchmark tổng hợp (bộ 5) | `code/src/ocr/` · `code/src/synthetic/` |

Chi tiết từng bước, chương · slide bài giảng và đối chứng: kế hoạch toàn nhóm mục 5–8.

## Cấu trúc repo

```
.
├── code/          Khung code (skeleton): src/ (module từng người), tests/, experiments/, samples/
├── data/          Dữ liệu bộ 1–5 (KHÔNG commit, tải từ Drive nhóm); chỉ theo dõi README và nguồn/license
├── docs/          Tổng quan đề tài, trang tra cứu lý thuyết theo chương
│   └── references/  Tài liệu của thầy: requirements/ (quy định đồ án), lectures/ (bài giảng chương 1–8), practices/ (notebook thực hành)
├── report/        Báo cáo: dàn ý, hướng dẫn viết, khung LaTeX
├── slides/        Slide thuyết trình
└── team/          Kế hoạch toàn nhóm (kế hoạch chính), bảng tiến độ
```

`results/` và `out/` (kết quả thí nghiệm, ảnh trung gian) nằm ở gốc repo và không commit.

## Bắt đầu từ đâu

| Cần gì | Mở file |
| :--- | :--- |
| **Kế hoạch chính:** phương pháp, thí nghiệm, tham số, phân công, lịch, checklist | [team/KeHoach.md](team/KeHoach.md) |
| Tổng quan đề tài, gói nộp | [docs/DeTai_XuLyAnh_ChiTiet.md](docs/DeTai_XuLyAnh_ChiTiet.md) |
| Quy định đồ án của thầy | [docs/references/requirements/DoAn_XuLyAnh.md](docs/references/requirements/DoAn_XuLyAnh.md) |
| Tra cứu lý thuyết theo chương, notebook thực hành | [docs/README.md](docs/README.md) |
| Hôm nay ai phải làm gì | [team/TienDo.md](team/TienDo.md) |
| Cài đặt, chạy code, hợp đồng dữ liệu | [code/README.md](code/README.md) |
| Kiểm kê dữ liệu, nguồn và license | [data/README.md](data/README.md) · [data/SOURCES_AND_LICENSES.md](data/SOURCES_AND_LICENSES.md) |
| Dàn ý và hướng dẫn viết báo cáo | [report/outline.md](report/outline.md) · [report/guidelines.md](report/guidelines.md) |

## Chạy nhanh

```bash
cd code
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest                                              # test của hàm chưa viết hiện SKIPPED
python run_pipeline.py samples/                     # chạy đầu-cuối trên ảnh mẫu
```

OCR cần thêm `requirements-ocr.txt` và Tesseract 5; chi tiết ở [code/README.md](code/README.md).

## Hạn chót

| Ngày | Việc |
| :---: | :--- |
| 09/11/2026 (tối) | Chạy cả pipeline trên 10 ảnh mẫu |
| 10/11/2026 | Sáng: mỗi người gửi phần báo cáo + 2 slide, Hiếu ghép · Chiều: quay demo, ghép video · Tối: đóng gói `.zip`, thử trên máy sạch |
| **11/11/2026** | **Nộp đồ án** qua email `uyenvnb@ptit.edu.vn` |

## Quy ước

- Tài liệu tự viết đặt vào `docs/`; tài liệu của thầy vào `docs/references/`; sản phẩm nộp vào `report/` và `slides/`.
- Tên file mới: chữ thường, không dấu, không khoảng trắng (ví dụ `chuong7_xu_ly_hinh_thai.md`). Tài liệu nhóm hiện có giữ tên cũ.
- Link giữa các tài liệu luôn dùng đường dẫn tương đối.
- Quy ước code (ảnh BGR, mask mực = 255, bbox `[x, y, w, h]`, đọc ảnh qua `common.io`) ở [code/README.md](code/README.md) mục 3.
