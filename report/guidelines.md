# HƯỚNG DẪN VIẾT BÁO CÁO DIP

> **Hạn:** mỗi người điền phần mình trong ngày **11/11/2026**, Hiếu ghép và nộp trước 23:59 (lịch: [plan.md](../plan.md) mục 8)  
> **Yêu cầu của thầy** ([DoAn_XuLyAnh.md](../docs/references/requirements/DoAn_XuLyAnh.md), slide 8): báo cáo **4–10 trang**, `.pdf`, gồm bài toán, dữ liệu, phương pháp, kết quả, thảo luận, phân công, khai báo sử dụng AI. Ghi rõ chương học phần đã dùng; mỗi bước ghi tham số và lý do chọn; có phương pháp đối chứng và độ đo định lượng.  
> **Dàn ý và phân công file:** [outline.md](outline.md) · **Thí nghiệm, bảng chính của từng khối:** [plan.md](../plan.md) mục 5, 10

## 1. Biên dịch

Cần MiKTeX hoặc TeX Live (có sẵn XeLaTeX, biber, latexmk, phông TeX Gyre Termes).

```bash
cd report/latex
latexmk            # XeLaTeX + biber, kết quả: build/main.pdf
latexmk -c         # xóa file trung gian
```

Lần đầu chạy, MiKTeX có thể hỏi cài thêm gói: chọn cài. `build/` không đưa lên git.

## 2. Cấu trúc file và phân công viết

Bảng phân công file theo v2 ở [outline.md](outline.md). Mỗi người chỉ sửa file của khối mình; `main.tex` chỉ nhóm trưởng sửa.

| Phần | Nguồn trích | Thư mục hình | Người |
| :--- | :--- | :--- | :--- |
| Dùng chung, khối A | `refs/common.bib`, `refs/preprocessing.bib`, `refs/cleaning.bib` | `figure/common/`, `figure/enhance/` | Hiếu |
| Khối B, C, D, VLM, độ đo | `refs/dl.bib` | `figure/detect/`, `figure/recognize/`, `figure/assemble/` | Trường, Khang, Thắng |

Đầu mỗi file có dòng `% Viết gì:` ghi nội dung cần có. Chỗ chưa viết đánh dấu `\TODO{...}` (chữ đỏ), số liệu chờ điền đánh dấu `\KQ{...}` (chữ xanh); **trước khi gửi không còn `\TODO`, `\KQ` nào trong file được giao.**

Bản chính tổng cộng **≤ 10 trang**, phụ lục không giới hạn. Mỗi người khoảng **1,5–2 trang**, gồm **1 bảng chính + 1 hình chính**; còn lại đưa vào phụ lục.

## 3. Mỗi phần phải có

| Mục | Nội dung |
| :--- | :--- |
| Phương pháp (3.x, ≈ 0,75 trang) | Các bước và lý do chọn; **chương + slide + công thức chính** (`\chuong{3}`); phương pháp đối chứng; **bước hậu xử lý**; bảng tham số rút gọn. Kỹ thuật ngoài bài giảng ghi nguồn theo ký hiệu ✅ / ➕ / 📘 / ⚠️ (kế hoạch mục 4.8) |
| Kết quả (4.x, ≈ 1 trang) | 1 bảng chính so sánh các phương án; 1 hình chính (đường cong theo mức hỏng); 2–3 câu nhận xét có số liệu; 1–2 ca lỗi kèm ảnh và nguyên nhân |
| Phụ lục | Bảng tham số đầy đủ (tên, giá trị chọn, khoảng đã thử, lý do), đường cong theo tham số, các thí nghiệm phụ |
| Gửi kèm (ngoài LaTeX) | 2–3 slide vào `slides/`, đoạn demo 30 giây, README module trong `code/` |

## 4. Quy ước

### 4.1. Hình và bảng

Mẫu có sẵn trong [sections/4_2_preprocessing.tex](latex/sections/4_2_preprocessing.tex).

| Hạng mục | Quy ước |
| :--- | :--- |
| File hình | `figure/<phần>/<mã thí nghiệm>_<số>.pdf` (ví dụ `figure/preprocessing/D3_1.pdf`). Ưu tiên `.pdf` (matplotlib `savefig("x.pdf")`); ảnh chụp dùng `.jpg` |
| Nhãn | `\label{fig:D3_1}`, `\label{tab:T2}`; tham chiếu bằng `Hình~\ref{fig:D3_1}`, `Bảng~\ref{tab:T2}` |
| Chú thích | Tự đủ nghĩa khi đọc riêng; ghi tập đo (bộ nào, split nào) |
| Bảng | `booktabs` (`\toprule`, `\midrule`, `\bottomrule`), không kẻ dọc; **in đậm** kết quả tốt nhất mỗi cột; dưới bảng ghi script tạo ra số liệu |
| Số | Dấu thập phân là dấu phẩy: `\num{0.953}` → 0,953; 3 chữ số có nghĩa |
| Vị trí | `[H]` để hình / bảng nằm đúng chỗ viết |

### 4.2. Trích dẫn

- Kiểu IEEE, tự đánh số. Trích bằng `\cite{zhang2007whiteboard}`.
- Thêm nguồn vào file `.bib` của phần mình (bảng mục 2); khóa đặt theo `<họ tác giả đầu><năm><từ khóa>`.
- Nguồn nhiều phần cùng dùng: nhờ Hiếu thêm vào `refs/common.bib`. Trước khi thêm, tìm trong 5 file `.bib` xem đã có chưa, tránh trùng khóa.

### 4.3. Lệnh dùng chung (khai báo trong `main.tex`)

| Lệnh | Kết quả |
| :--- | :--- |
| `\TODO{...}` | Chữ đỏ đánh dấu chỗ chưa viết |
| `\chuong{3}` | (Chương 3) |
| `\KQ{...}` | Số liệu chờ điền (chữ xanh) |
| `\khoi{A}`, `\Khoi{A}` | khối A, Khối A |

### 4.4. Thuật ngữ thống nhất

| Dùng | Không dùng |
| :--- | :--- |
| nắn phối cảnh | nắn phẳng, warp, chỉnh méo |
| chỉnh nghiêng | deskew, xoay thẳng |
| nhị phân hóa | binarize, phân ngưỡng (khi nói về cả bước) |
| khử bóng | xóa bóng, loại bóng |
| lớp mực | layer mực, kênh mực |
| khoản thuốc | block thuốc, mục thuốc |
| phiếu xét nghiệm | phiếu XN, lab report |
| khối A/B/C/D | block, module (khi nói về pipeline) |
| vùng chữ | text box, box chữ |
| tinh chỉnh | fine-tune (trong văn bản) |
| tập huấn luyện / kiểm định / kiểm tra | train / valid / test (trong văn bản) |
| bộ ảnh suy giảm | ảnh degraded |

Quy ước kỹ thuật giống [code/README.md](../code/README.md): bbox `[x0, y0, x1, y1]` (xyxy); mask mực = 255; ảnh màu BGR; 4 góc theo thứ tự TL, TR, BR, BL.

## 5. Khai báo sử dụng AI

Mỗi người ghi phần của mình vào cuối file 4.x (dạng chú thích `%` hoặc gửi Hiếu); Hiếu gộp vào mục 7.

| Công cụ | Dùng cho việc gì | Phần nào do người tự làm | Cách kiểm tra kết quả của AI |
| :--- | :--- | :--- | :--- |
| *Ví dụ: Claude* | *Gợi ý cấu trúc code, sửa lỗi, viết nháp đoạn văn* | *Chọn phương pháp, chạy thí nghiệm, đọc kết quả, phân tích lỗi* | *Unit test, so với nhãn chuẩn VAIPE, đọc lại từng đoạn văn* |

## 6. Checklist trước khi gửi phần (mỗi người)

- [ ] `latexmk` chạy không lỗi
- [ ] Không còn `\TODO`, `\KQ` trong các file được giao
- [ ] Có đủ: phương pháp, bảng chính, hình chính, nhận xét có số, ca lỗi, bảng tham số (phụ lục)
- [ ] Mọi số liệu chạy lại được từ `code/experiments/`; dưới mỗi bảng / hình ghi tên script
- [ ] Tham số chọn trên valid, số liệu báo trên test; tập đánh giá ≥ 50 ảnh (kế hoạch mục 3.3)
- [ ] Hình / bảng đúng quy ước tên, nhãn, chú thích
- [ ] Dùng đúng thuật ngữ ở mục 4.4
- [ ] Nguồn trích đã có trong file `.bib` của phần mình
- [ ] Khai báo AI đã ghi
- [ ] Độ dài phần mình ≤ 2 trang (không tính phụ lục)

## 7. Checklist trước khi nộp (Hiếu, tối 11/11)

- [ ] Bản chính ≤ 10 trang, không còn `\TODO`, `\KQ`, xuất `build/main.pdf`
- [ ] Hình / bảng đánh số liên tục, đều được nhắc tới trong văn bản
- [ ] Số liệu khớp giữa báo cáo, slide và `results/`
- [ ] Tài liệu tham khảo đầy đủ, không có `\cite` lỗi (dấu `?` trong PDF)
- [ ] Điền lớp, số nhóm, MSSV ở `sections/0_title.tex`
- [ ] Slide 10–15 trang (`.pptx` + `.pdf`), video ≤ 3 phút
- [ ] `.zip` đặt tên `LopXX_NhomYY_TenDeTai.zip`, có `code/`, `requirements.txt`, `samples/` (5–10 ảnh), `README.md`, báo cáo PDF, slide, video; chạy thử trên máy sạch
