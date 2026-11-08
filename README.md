# MedTrack AI

Ứng dụng số hóa đơn thuốc và phiếu xét nghiệm từ ảnh chụp, giúp gia đình quản lý sức khỏe người thân: nhắc uống thuốc, theo dõi chỉ số xét nghiệm qua nhiều lần khám.

Nhóm: Hiếu (nhóm trưởng) · Trường · Khang · Thắng — Học viện Công nghệ Bưu chính Viễn thông.

Repo phục vụ **3 đầu việc dùng chung một sản phẩm**: đồ án Xử lý ảnh số (DIP), bài tập lớn Học máy (ML) và cuộc thi NEXUS 2026. Mỗi đầu việc có thư mục riêng; code, phân công và tiến độ dùng chung nằm ở gốc.

## Cấu trúc repo

```
.
├── team/                Phân công, tiến độ, kế hoạch từng người (chung cả 3 đầu việc)
├── code/                Pipeline xử lý ảnh dùng chung (DIP + ML + app): module từng người, hợp đồng dữ liệu, test, thí nghiệm
├── dip/                 Đồ án Xử lý ảnh số (hạn 11/11/2026)
│   ├── docs/            Đề tài chi tiết, tra cứu lý thuyết
│   ├── references/      requirements/ (đề bài của thầy) · lectures/ (slide bài giảng) · practices/ (notebook thực hành)
│   ├── report/          Báo cáo: outline, LaTeX
│   └── slides/          Slide thuyết trình của nhóm
├── ml/                  Bài tập lớn Học máy
│   ├── docs/            Kế hoạch bài tập lớn
│   ├── references/      requirements/ · lectures/
│   ├── report/
│   └── slides/
├── nexus/               Cuộc thi NEXUS 2026
│   ├── docs/            Bản nháp đề xuất
│   ├── references/      Thể lệ
│   └── report/          Bản nộp
├── pdlc/                Tài liệu sản phẩm theo PDLC (ý tưởng → nghiên cứu → yêu cầu → thiết kế)
└── frontend/            App Flutter (không theo dõi trên nhánh dip)
```

`data/`, `results/`, `out/` nằm ở gốc; dataset và kết quả không đưa lên Git (xem [code/README.md](code/README.md)). Riêng [kiểm kê dữ liệu](data/README.md) và [nguồn/license](data/SOURCES_AND_LICENSES.md) được theo dõi để cả nhóm tra cứu. Thống nhất tên thư mục `data/` viết thường.

## Bắt đầu từ đâu

| Cần gì | Mở file |
| :--- | :--- |
| Pipeline, dữ liệu, ai làm gì ở cả 3 đầu việc | [team/PhanCong_TongHop.md](team/PhanCong_TongHop.md) |
| Hôm nay ai phải giao gì cho ai | [team/TienDo.md](team/TienDo.md) |
| Kế hoạch chi tiết của từng người | [Hiếu](team/KeHoach_Hieu.md) · [Trường](team/KeHoach_Truong.md) · [Khang](team/KeHoach_Khang.md) · [Thắng](team/KeHoach_Thang.md) |
| Cài đặt, chạy code, hợp đồng dữ liệu | [code/README.md](code/README.md) |
| Đề tài DIP (thí nghiệm, độ đo, benchmark tổng hợp) | [dip/docs/DeTai_XuLyAnh_ChiTiet.md](dip/docs/DeTai_XuLyAnh_ChiTiet.md) |
| Quy định đồ án DIP của thầy Cường | [dip/references/requirements/DoAn_XuLyAnh.md](dip/references/requirements/DoAn_XuLyAnh.md) |
| Tra cứu lý thuyết DIP theo chương | [dip/docs/README.md](dip/docs/README.md) |
| Dàn ý báo cáo DIP | [dip/report/outline.md](dip/report/outline.md) |
| Kế hoạch bài tập lớn ML | [ml/docs/KeHoach_BTL_MachineLearning.md](ml/docs/KeHoach_BTL_MachineLearning.md) · [yêu cầu](ml/references/requirements/require.md) |
| Thể lệ và bản nháp NEXUS | [nexus/references/thele.md](nexus/references/thele.md) · [bản nháp tiếng Việt](nexus/docs/nopbai_draft_vi.md) · [bản tiếng Anh](nexus/docs/nopbai_draft.md) |
| Tài liệu sản phẩm (UX/UI, kiến trúc, chuẩn hóa y tế) | [pdlc/README.md](pdlc/README.md) |

## Hạn chót

| Ngày | Việc |
| :---: | :--- |
| 05/11/2026 | Mốc tích hợp pipeline · Đăng ký đội NEXUS |
| 07/11/2026 | Nộp đề xuất NEXUS vòng 1 |
| 11/11/2026 | Nộp đồ án Xử lý ảnh |
| Chưa rõ | Nộp bài tập lớn ML |

## Quy ước

- Tên file tài liệu sản phẩm: `kebab-case.md` (xem [pdlc/README.md](pdlc/README.md)).
- Tài liệu nhóm và môn học giữ tên hiện có; tài liệu mới đặt đúng thư mục của đầu việc: tài liệu tự viết vào `docs/`, tài liệu của thầy / BTC vào `references/`, sản phẩm nộp vào `report/` và `slides/`.
- Link giữa các tài liệu luôn dùng đường dẫn tương đối.
