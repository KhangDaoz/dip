# T2 — So sánh 5 cách nhị phân hóa × có/không khử bóng

## Phiên bản đang dùng: v2 — 03/11/2026

CLI mặc định chọn `results/t1_v2/selection.json` và output `results/t2_v2/`. Nếu T1 được tạo bằng code mới, script kiểm tra `input_contract.json` đã được ràng buộc trong config, không cần ZIP lịch sử.

**Với kết quả T1 lịch sử đang có trên máy, cần chỉ rõ bằng chứng**, kể cả khi manifest còn giống cũ. Không có cờ bỏ qua checksum. Kiểm tra tương thích, không chạy đo và không ghi file:

```powershell
python code/experiments/t2_binarization.py --check-inputs-only --compatibility-evidence results/t1_t2_repro_audit/t1_t2_v2_handoff.zip
```

Khi thực sự chạy T2 từ T1 lịch sử, thêm cùng `--compatibility-evidence ...` vào lệnh bên dưới và chọn output mới. ZIP gốc được giữ nguyên; script đối chiếu config, selection, nguồn/split/seed/hệ tọa độ, hash tham chiếu đã đo, các phụ thuộc code sinh bóng và phiên bản thư viện. Manifest thay đổi được ghi lại riêng nếu các bằng chứng liên quan đều khớp. Thay đổi ảnh chuẩn, nhãn, code liên quan hoặc thiếu bằng chứng sẽ dừng. Không sửa config lịch sử để ghi hash mới.

```powershell
python code/experiments/t2_binarization.py --data-dir data/6_synthetic --t1-selection results/t1_v2/selection.json --compatibility-evidence results/t1_t2_repro_audit/t1_t2_v2_handoff.zip --output-dir results/t2_v2_repeat
```

Dùng `manifest_v2.json` tại `data/6_synthetic/`. Gói v2 có 70 mẫu `single_shadow` (60 train, 10 val), **không có mẫu bóng đơn test**. Vì vậy phép đo mới sinh **chỉ bóng** từ ảnh sạch v2 cho cả ba split; không trộn snapshot cũ hay mẫu tổ hợp v14.

Giữ 100 nguồn và split 60/10/30. Sắp ID nguồn trong từng split rồi luân phiên planar/silhouette và mức 1/2/3 (darkness 0,2/0,45/0,7). Seed từng nguồn lấy 8 byte đầu SHA-256 của `42:<source_id>:derived_shadow_v2`, little-endian. Gọi `synthetic.degrade.shadow` trên BGR sạch rồi chuyển xám; giữ kích thước và H đơn vị. V00 chỉ dùng tra đường dẫn ảnh sạch, không dùng ảnh hỏng/H của v00; metadata gốc ở trường `provenance` trong `samples.json`.

T1 v2 so không khử, Median, Closing và Closing + Median; kernel 15/21/31/41/61/81: **19 cấu hình val**. Closing đứng riêng là phần bổ sung so với giao thức cũ. T2 giữ lưới 5 phương pháp × hai nhánh (**80 cấu hình val**), đọc đúng cấu hình T1 đã khóa. Train sanity → chọn trên val → khóa → đo test. Không dùng `img_test` hoặc snapshot cũ cho ảnh minh họa v2.

Lưu `samples.json`, `config.json`, checksum nguồn và checksum pixel xám đã sinh trong CSV. Script từ chối ghi đè kết quả. Đổi thư mục output khi chạy lại; T2 phải nhận `selection.json` cùng `config.json` của chính lần chạy T1 tương ứng. Các nguồn test đã từng được đo ở v1, nên đây **không phải một tập test độc lập mới**.

`clean_binary` là Otsu tự sinh, **không phải nhãn thủ công**. T2 đo mức khớp tham chiếu và có thiên lệch thuận Otsu. Chưa suy ra OCR/CER/WER, giữ chữ đỏ hoặc chất lượng ảnh thật; không tự đổi mặc định pipeline. Xem README trong kết quả v2 để đọc số liệu và giới hạn.

## Giao thức lịch sử v1 — chỉ dùng tái lập kết quả cũ


## Chạy từ gốc repo

```powershell
python code/experiments/t2_binarization.py --data-dir data/6_synthetic_v1_old/6_synthetic --t1-selection results/t1/selection.json --output-dir results/t2
```

Từ `code/`, dùng `python experiments/t2_binarization.py` và thêm `../` vào đường dẫn dữ liệu/kết quả tương ứng. Mặc định dữ liệu lấy từ `common.paths.DATASETS[6]`. Script từ chối ghi đè kết quả cũ; dùng output-dir mới nếu chạy lại có chủ đích.

## Phạm vi và nguồn tham chiếu

- Bộ ⑥ v1, chỉ biến thể `single_shadow` (v05), cùng phạm vi khử bóng T1. Đây không phải kết quả cho tất cả 10 biến thể v1, ảnh nền màu, OCR hay phần tách lớp mực.
- Một mẫu cho mỗi ảnh nguồn; giữ nguyên split 60 train / 10 val / 30 test của Thắng. Sáu nguồn train được chọn định trước theo mức/dạng bóng chỉ để kiểm tra chạy được.
- Xác nhận ảnh sạch/hỏng cùng kích thước, H đơn vị và không lẫn lỗi hình học/mờ; đối chiếu ID/split với metadata 100 nguồn. Không tự sửa nhãn, resize hoặc nắn ảnh theo kết quả dự đoán.
- `clean_binary.png` được kiểm tra lại khớp Otsu trên ảnh sạch. Đây là **tham chiếu tự sinh, không phải nhãn thủ công**; phép đo ưu ái các quyết định giống Otsu. Không diễn giải F1 gần 100% là OCR gần 100% hay thuật toán tốt nhất trên mọi tài liệu.

## Hai nhánh và lưới tham số

- `none`: ảnh xám chưa khử bóng.
- `t1`: đọc đúng lựa chọn đã khóa ở T1 (Closing + Median 81 trong lần chạy đầu). Kiểm tra hash manifest để không vô tình dùng split/dữ liệu khác T1.
- Otsu không có tham số dò.
- Niblack: window 15/31/61, k −0,1/−0,2/−0,3, công thức T=mean+k×std.
- Sauvola: window 15/31/61, k 0,1/0,2/0,3/0,5, R=128 cố định.
- Wolf: window 15/31/61, k 0,2/0,5/0,8.
- Gaussian adaptive: block 15/31/61, C=5/10/15.

Có 40 cấu hình mỗi nhánh, tổng **80 cấu hình × 10 ảnh val = 800 dòng**. Mỗi phương pháp ở mỗi nhánh được chọn tham số riêng trên cùng 10 nguồn val, theo F1 trung bình cao nhất; hòa thì pFM cao hơn, DRD thấp hơn rồi ID để kết quả xác định. Không dùng PSNR có thể vô hạn để phá hòa.

Lưu `selection.json` trước khi đọc pixel test; test chỉ đo **10 cấu hình đã khóa × 30 ảnh = 300 dòng**. Không chọn lại theo test hoặc mở rộng lưới sau khi thấy kết quả. Thêm 60 dòng kiểm tra train và 100 dòng đối chứng ảnh sạch val: tổng **1.260 dòng**.

## Độ đo và trường hợp đặc biệt

Code dùng chung: `code/src/cleaning/metrics.py`, public input uint8 mực=255/nền=0.

- F1, precision, recall: tính theo mực, trung bình theo ảnh.
- `pfm_skeleton`: pseudo F-measure dùng precision thường và recall trên skeleton GT theo **H-DIBCO 2010**, skeleton Zhang của scikit-image. Không phải weighted pseudo-F của DIBCO 2013 trở đi và không tuyên bố khớp bit với executable của ban tổ chức. Nguồn: [bài của tác giả, mục III.B](https://www.researchgate.net/publication/221389513_H-DIBCO_2010_-_Handwritten_Document_Image_Binarization_Competition).
- DRD: cửa sổ 5×5, trọng số nghịch đảo khoảng cách chuẩn hóa, tâm bằng 0; số block không đồng nhất 8×8 tính cả block mép có kích thước thực. Ở biên, chỉ cộng hàng xóm nằm trong ảnh, không chuẩn hóa lại riêng cửa sổ. Nguồn: [Lu–Kot–Shi, 2004](https://doi.org/10.1109/LSP.2003.821748), [tài liệu H-DIBCO 2014 của ban tổ chức](https://iapr.org/archives/icfhr2014/wp-content/uploads/2015/02/ICFR2014-H-DIBCO.pdf). Quy ước biên được công bố và kiểm tra bằng phép cộng trực tiếp, không giả định mọi executable dùng giống nhau.
- PSNR từng ảnh = −10 log10(tỷ lệ pixel sai), bằng vô hạn khi khớp hoàn toàn. Bảng tổng hợp `psnr_db` tính **−10 log10(trung bình tỷ lệ pixel sai theo ảnh)** để không bị một ảnh khớp hoàn toàn làm trung bình dB vô hạn; đây là PSNR từ MSE trung bình, không phải trung bình dB. Vẫn lưu `mean_image_psnr_db` và số ảnh khớp `exact_matches`, không cắt PSNR bằng trần tùy ý.
- DRD không xác định nếu GT đồng nhất và dự đoán sai; trả NaN. Benchmark từ chối mẫu không có block không đồng nhất hoặc không có skeleton, không âm thầm coi điểm bằng 0.

## Đầu ra và cách dùng

- `config.json`: phạm vi, lưới, định nghĩa điểm, phiên bản thư viện và hash code/dữ liệu.
- `validation.csv`, `validation_ranking.csv`: điểm theo ảnh/cấu hình, xếp hạng val.
- `selection.json`: 10 cấu hình đã khóa và cấu hình đứng đầu theo tham chiếu hiện có; đầu vào cho bước tích hợp sau.
- `heldout_test.csv`, `test_summary.csv`: điểm test chi tiết và bảng chính 5×2.
- `val_clean_control.csv`, `clean_control_summary.csv`: tác dụng phụ trên ảnh sạch; chỉ phân tích, không đổi lựa chọn sau test.
- `train_sanity.csv`, `table.csv`: kiểm tra chạy và tổng hợp mọi dòng.
- `figure.png/pdf`: F1 và DRD test.
- `examples/`: ví dụ val định trước cho 3 mức bóng; ảnh so sánh chỉ đảo màu để xem, không thay quy ước mực/nền.
- `development_unscored/`: nếu có bộ `img_test` local, chỉ tạo ảnh đối chiếu hai nhánh tốt nhất trên val. Không có nhãn để chấm điểm các phiếu xét nghiệm màu; không dùng ảnh này để thay tham số đã khóa.

Chưa tự thay mặc định các hàm, `cleaning.pipeline.clean`, JSON bàn giao hoặc file của đồng đội. Chọn trên benchmark tự sinh là quyết định có phạm vi; giữ các ca lỗi để Trường duyệt chữ, dấu tiếng Việt, số/đơn vị và nội dung màu trước khi tích hợp.
