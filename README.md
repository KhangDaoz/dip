# Đồ án Xử lý ảnh số — Trích xuất thông tin đơn thuốc từ ảnh

**Đề tài:** Trích xuất thông tin đơn thuốc (tên thuốc, số lượng, cách dùng, chẩn đoán, ngày) từ ảnh đơn thuốc bảo hiểm y tế,
kết hợp học sâu với xử lý ảnh cổ điển.

Môn Xử lý ảnh số, PGS.TS. Phạm Văn Cường, Học viện Công nghệ Bưu chính Viễn thông. Hình thức: **Lựa chọn 2, dự án xử lý ảnh thực tế**.
Nhóm: Hiếu (nhóm trưởng) · Trường · Khang · Thắng. Kế hoạch: [plan.md](plan.md).

## Pipeline

```
ảnh ─► [A] Tiền xử lý ─► [B] Phát hiện vùng chữ ─► [C] Nhận dạng chữ ─► [D] Ghép khoản thuốc ─► JSON
       cổ điển             YOLOv8s, 6 lớp            VietOCR tinh chỉnh     luật + từ điển tên thuốc
       (Ch. 2, 3, 5, 7)    đối chứng: Sauvola+giãn   đối chứng: Tesseract   đối chứng đầu-cuối: VLM zero-shot
```

| Khối | Người | Phương pháp chính | Đối chứng | Mã nguồn |
| :--- | :--- | :--- | :--- | :--- |
| A Tiền xử lý | Hiếu | Nắn phối cảnh (Canny), chỉnh nghiêng (lược đồ chiếu), khử bóng (phép đóng), lọc trung vị, CLAHE | Tắt khối A | `code/src/enhance/` |
| B Phát hiện | Trường | YOLOv8s phát hiện + phân 6 lớp vùng chữ | Sauvola → giãn → thành phần liên thông | `code/src/detect/` |
| C Nhận dạng | Khang | VietOCR `vgg_transformer` tinh chỉnh trên VAIPE (C3) | VietOCR gốc (C2), Tesseract 5 (C0, C1) | `code/src/recognize/` |
| D Ghép khoản | Thắng | Chọn lớp, gom theo dải ngang, chuẩn hóa, so khớp từ điển | VLM zero-shot (Qwen2.5-VL) | `code/src/assemble/` |
| Tích hợp | Hiếu | Ghép A→B→C→D, CLI, file bàn giao cho thí nghiệm | | `code/src/pipeline/`, `code/run_pipeline.py` |

Dữ liệu: phần đơn thuốc của bộ VAIPE, chia lại 820 / 176 / 176 ảnh (huấn luyện / kiểm định / kiểm tra), không rò rỉ bệnh nhân giữa các tập.

## Cấu trúc repo

```
.
├── code/       Mã nguồn: src/ (4 khối + pipeline), tools/, experiments/, notebooks/, tests/, samples/ — xem code/README.md
├── data/       Dữ liệu (KHÔNG commit, tải từ Drive nhóm); chỉ theo dõi README
├── docs/       Tổng quan đề tài, tài liệu của thầy (references/: quy định đồ án, bài giảng, notebook thực hành)
├── report/     Báo cáo LaTeX
├── slides/     Slide thuyết trình
└── plan.md     Kế hoạch v2: phương pháp, thí nghiệm, phân công
```

`results/` (kết quả thí nghiệm) và `out/` (đầu ra `run_pipeline.py`) nằm ở gốc repo và không commit.

## Chạy nhanh

```bash
cd code
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # GPU; CPU: .../whl/cpu
pip install -r requirements-dl.txt
pip install --no-deps -r requirements-vietocr.txt
pip install -e .
python tools/download_weights.py          # trọng số YOLOv8s + VietOCR từ Drive nhóm (kiểm tra SHA-256)
pytest                                    # phần thiếu dữ liệu / trọng số / Tesseract tự SKIP
python run_pipeline.py samples/           # pipeline chính trên ảnh mẫu → out/<ảnh>.json + out/<ảnh>_overlay.png
```

Đối chứng Tesseract (C0, C1, preset `classical`, `hybrid`) cần cài Tesseract 5 kèm gói ngôn ngữ `vie`.
Thí nghiệm và thứ tự chạy: [code/README.md](code/README.md).

## Nộp bài

Hạn **11/11/2026**, một file `.zip` (slide, mã nguồn + README, báo cáo 4–10 trang) gửi `uyenvnb@ptit.edu.vn`.
Quy định: [docs/references/requirements/DoAn_XuLyAnh.md](docs/references/requirements/DoAn_XuLyAnh.md).
