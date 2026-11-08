# Đồ án Xử lý ảnh số — Pipeline xử lý ảnh tài liệu y tế (MedTrack)

**Đề tài:** Nghiên cứu & xây dựng pipeline tiền xử lý, phân loại biểu mẫu và bóc tách cấu trúc tài liệu y tế (đơn thuốc, phiếu xét nghiệm) từ ảnh chụp thực tế.

Môn Xử lý ảnh số (DIP), PGS.TS. Phạm Văn Cường, Học viện Công nghệ Bưu chính Viễn thông. Hình thức: **Lựa chọn 2, Dự án xử lý ảnh thực tế**.
Nhóm: Hiếu (nhóm trưởng) · Trường · Khang · Thắng.

Phạm vi, thí nghiệm, độ đo và phân công **đã chốt** ở [docs/DeTai_XuLyAnh_ChiTiet.md](docs/DeTai_XuLyAnh_ChiTiet.md). Không sửa file này khi chưa thống nhất cả nhóm.

## Pipeline

| Giai đoạn | Người | Nội dung | Module |
| :--- | :--- | :--- | :--- |
| 1. Tiền xử lý & nắn chỉnh | Hiếu | Đo mờ / lóa, tìm 4 góc, nắn phối cảnh, chỉnh nghiêng, CLAHE | `code/src/preprocessing/` |
| 2. Phân loại biểu mẫu | Hiếu | HOG + $k$-NN: đơn thuốc / phiếu xét nghiệm / tài liệu khác | `code/src/classification/` |
| 3. Làm sạch & tách lớp mực | Trường | Khử bóng, tách mực HSV, inpainting, nhị phân hóa Sauvola | `code/src/cleaning/` |
| 4A. Dựng bảng phiếu xét nghiệm | Khang | Gom dòng / cột bằng Projection + Dilation, cờ bất thường | `code/src/table_extraction/` |
| 4B. Bóc tách đơn thuốc | Thắng | Tách từng khoản thuốc bằng Projection + Dilation | `code/src/prescription_segmentation/` |
| Dùng chung | Khang · Thắng | Bộ OCR + CER/WER · bộ sinh benchmark tổng hợp (bộ 5) | `code/src/ocr/` · `code/src/synthetic/` |

## Cấu trúc repo

```
.
├── code/          Mã nguồn pipeline: src/ (module từng người), tests/, experiments/, scripts/, examples/, samples/
├── data/          Dữ liệu bộ 1–5 (KHÔNG commit, tải từ Drive nhóm); chỉ theo dõi README và nguồn/license
├── docs/          Đề tài chi tiết (đã chốt), trang tra cứu lý thuyết theo chương
│   └── references/  Tài liệu của thầy: requirements/ (quy định đồ án), lectures/ (bài giảng chương 1–8), practices/ (notebook thực hành)
├── report/        Báo cáo: dàn ý, hướng dẫn viết, khung LaTeX
├── slides/        Slide thuyết trình
└── team/          Phân công tổng hợp, kế hoạch từng người, bảng tiến độ
```

`results/` và `out/` (kết quả thí nghiệm, ảnh trung gian) nằm ở gốc repo và không commit.

## Bắt đầu từ đâu

| Cần gì | Mở file |
| :--- | :--- |
| Đề tài, thí nghiệm đối chứng, độ đo, gói nộp | [docs/DeTai_XuLyAnh_ChiTiet.md](docs/DeTai_XuLyAnh_ChiTiet.md) |
| Quy định đồ án của thầy | [docs/references/requirements/DoAn_XuLyAnh.md](docs/references/requirements/DoAn_XuLyAnh.md) |
| Tra cứu lý thuyết theo chương, notebook thực hành | [docs/README.md](docs/README.md) |
| Pipeline, dữ liệu, ai làm gì | [team/PhanCong_TongHop.md](team/PhanCong_TongHop.md) |
| Hôm nay ai phải giao gì cho ai | [team/TienDo.md](team/TienDo.md) |
| Kế hoạch chi tiết từng người | [Hiếu](team/KeHoach_Hieu.md) · [Trường](team/KeHoach_Truong.md) · [Khang](team/KeHoach_Khang.md) · [Thắng](team/KeHoach_Thang.md) |
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
| 08/11/2026 | Trường, Khang, Thắng gửi phần báo cáo, slide, demo cho Hiếu |
| 09/11/2026 | Hiếu ghép báo cáo, slide, video |
| 10/11/2026 | Đóng gói `.zip`, chạy thử trên máy sạch |
| **11/11/2026** | **Nộp đồ án** qua email `uyenvnb@ptit.edu.vn` |

## Quy ước

- Tài liệu tự viết đặt vào `docs/`; tài liệu của thầy vào `docs/references/`; sản phẩm nộp vào `report/` và `slides/`.
- Tên file mới: chữ thường, không dấu, không khoảng trắng (ví dụ `chuong7_xu_ly_hinh_thai.md`). Tài liệu nhóm hiện có giữ tên cũ.
- Link giữa các tài liệu luôn dùng đường dẫn tương đối.
- Quy ước code (ảnh BGR, mask mực = 255, bbox `[x, y, w, h]`, đọc ảnh qua `common.io`) ở [code/README.md](code/README.md) mục 3.
