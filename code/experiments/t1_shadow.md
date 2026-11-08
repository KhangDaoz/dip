# T1 — So sánh khử bóng

## Phiên bản đang dùng: v2 — 03/11/2026

CLI hiện mặc định v2, output `results/t1_v2/`. Có thể kiểm tra đầu vào mà không chạy đo/ghi kết quả bằng `python code/experiments/t1_shadow.py --check-inputs-only`. Lần chạy T1 mới lưu `input_contract.json` với hash tham chiếu và fingerprint code liên quan; config ràng buộc checksum contract, rồi ràng buộc selection trước khi đo test. Chọn output mới để giữ kết quả lịch sử.

```powershell
python code/experiments/t1_shadow.py --dataset-version v2 --data-dir data/5_synthetic --output-dir results/t1_v2
```

Dùng `manifest_v2.json` tại `data/5_synthetic/`. Gói v2 có 70 mẫu `single_shadow` (60 train, 10 val), **không có mẫu bóng đơn test**. Vì vậy phép đo mới sinh **chỉ bóng** từ ảnh sạch v2 cho cả ba split; không trộn snapshot cũ hay mẫu tổ hợp v14.

Giữ 100 nguồn và split 60/10/30. Sắp ID nguồn trong từng split rồi luân phiên planar/silhouette và mức 1/2/3 (darkness 0,2/0,45/0,7). Seed từng nguồn lấy 8 byte đầu SHA-256 của `42:<source_id>:derived_shadow_v2`, little-endian. Gọi `synthetic.degrade.shadow` trên BGR sạch rồi chuyển xám; giữ kích thước và H đơn vị. V00 chỉ dùng tra đường dẫn ảnh sạch, không dùng ảnh hỏng/H của v00; metadata gốc ở trường `provenance` trong `samples.json`.

T1 v2 so không khử, Median, Closing và Closing + Median; kernel 15/21/31/41/61/81: **19 cấu hình val**. Closing đứng riêng là phần bổ sung so với giao thức cũ. T2 giữ lưới 5 phương pháp × hai nhánh (**80 cấu hình val**), đọc đúng cấu hình T1 đã khóa. Train sanity → chọn trên val → khóa → đo test. Không dùng `img_test` hoặc snapshot cũ cho ảnh minh họa v2.

Lưu `samples.json`, `config.json`, checksum nguồn và checksum pixel xám đã sinh trong CSV. Script từ chối ghi đè kết quả. Đổi thư mục output khi chạy lại; T2 phải nhận `selection.json` cùng `config.json` của chính lần chạy T1 tương ứng. Các nguồn test đã từng được đo ở v1, nên đây **không phải một tập test độc lập mới**.

`clean_binary` là Otsu tự sinh, **không phải nhãn thủ công**. T2 đo mức khớp tham chiếu và có thiên lệch thuận Otsu. Chưa suy ra OCR/CER/WER, giữ chữ đỏ hoặc chất lượng ảnh thật; không tự đổi mặc định pipeline. Xem README trong kết quả v2 để đọc số liệu và giới hạn.

## Giao thức lịch sử v1 — chỉ dùng tái lập kết quả cũ


T1 đo mức độ khôi phục **ảnh xám** bằng PSNR và SSIM so với ảnh sạch cùng tọa độ. Đây là thí nghiệm riêng của bước khử bóng, chưa chọn phương pháp nhị phân hóa và chưa đo OCR.

## Chạy lại

Từ gốc repo, với dữ liệu đã tải trên máy Trường:

```powershell
python code/experiments/t1_shadow.py --dataset-version legacy --data-dir data/6_synthetic_v1_old/6_synthetic --output-dir results/t1
```

Từ thư mục `code/`, đổi thành `python experiments/t1_shadow.py --dataset-version legacy --data-dir ../data/6_synthetic_v1_old/6_synthetic --output-dir ../results/t1`.

Mặc định `--data-dir` lấy từ `common.paths.DATASETS[5]`; dùng tham số để trỏ tới vị trí dữ liệu thực tế, không sửa đường dẫn chung của nhóm. Script từ chối ghi đè thư mục kết quả không rỗng; dùng `--output-dir results/t1_repeat` nếu cần chạy lại có chủ đích.

## Giao thức đo

1. Đối chiếu ID và split với `synthetic_100_sources.json`. Giữ nguyên 60/10/30 nguồn; không chia lại.
2. Chỉ lấy `single_shadow` (v05), một biến thể cho mỗi nguồn; xác nhận H đơn vị và không có xoay/phối cảnh/mờ. Không dùng mask bóng làm đáp án ước lượng nền.
3. Chạy kiểm tra trên 6 ảnh train, mỗi tổ hợp mức bóng × dạng bóng một ảnh, kernel 41. Không dùng điểm train để xếp hạng.
4. Trên toàn bộ 10 nguồn val, thử không khử bóng và Median/Closing + Median với kernel 15, 21, 31, 41, 61, 81: tổng 13 cấu hình.
5. Chọn theo PSNR trung bình từng ảnh cao nhất; hòa thì SSIM, rồi kernel nhỏ hơn, rồi tên phương pháp. Mỗi phương án đo trên cùng các nguồn.
6. Lưu `selection.json` trước khi đọc pixel test. Trên 30 nguồn test chỉ chạy không khử và cấu hình val tốt nhất của mỗi phương pháp. Không mở rộng lưới hay chọn lại bằng test.
7. Kiểm tra thêm 10 ảnh sạch val bằng các cấu hình đã khóa; chỉ dùng để mô tả tác dụng phụ, không thay lựa chọn sau khi đã đo test.

Ảnh được chuyển BGR → xám, giữ kích thước gốc, không điều chỉnh độ sáng theo ảnh chuẩn. PSNR dùng dải 255; SSIM cửa sổ 7, không Gaussian, sample covariance. Trung bình theo ảnh, không gộp pixel của các ảnh có kích thước khác nhau. Median dùng biên lặp; Closing có SE vuông và biên lặp, sau đó Median cùng kernel.

`ink_mae` và `paper_mae` là sai số xám tuyệt đối trung bình ở vùng mực/nền, để hỗ trợ phân tích. Vùng mực lấy từ `clean_binary.png` tự sinh bằng Otsu; không phải nhãn thủ công, không dùng để chọn cấu hình T1.

## Đầu ra

- `config.json`: lưới tham số, giao thức, phiên bản thư viện và hash code/manifest.
- `validation.csv`, `validation_ranking.csv`: điểm từng ảnh và bảng xếp hạng chọn cấu hình.
- `selection.json`: lựa chọn đã khóa trên val để dùng cho T2.
- `test.csv`, `test_summary.csv`, `test_by_shadow.csv`: kết quả test từng ảnh, tổng hợp, và theo mức/dạng bóng.
- `clean_control*.csv`: kiểm tra tác dụng phụ trên ảnh sạch; PSNR vô hạn nếu ảnh ra giống hệt tham chiếu. Không hiểu PSNR trung bình vô hạn là mọi ảnh đều hoàn hảo; xem SSIM/MAE và từng dòng.
- `train_sanity.csv`, `table.csv`: kiểm tra train và tổng hợp tất cả 268 dòng của lần chạy đầu.
- `figure.png`, `figure.pdf`: biểu đồ test; `examples/`: ví dụ val định trước theo ba mức bóng.

Kết quả lần chạy 01/11/2026: Closing + Median 81 được chọn trên val; test đạt PSNR 50,206 dB, SSIM 0,999293. Đọc `results/t1/README.md` để xem bảng và giới hạn đầy đủ. Không đổi mặc định hàm hoặc ảnh fixture bàn giao theo kết quả này; T2 truyền `method='closing_median', kernel=81` hoặc đọc lựa chọn đã lưu.
