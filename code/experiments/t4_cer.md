# T4 phiên 6 — code và điều kiện chặn CER

## Cập nhật hiện hành: cohort nhỏ reduced_v1 (thay yêu cầu duyệt toàn bộ phía dưới)

Đã được Trường đồng ý trước OCR test: chọn một box/ảnh đủ điều kiện bằng
SHA256(`MedTrackAI-T4-CER-reduced-v1-20261104|crop_id`), tăng dần. Khóa 15 val
(10 nguồn) và 47 test (30 nguồn), tổng 62 box chính. Manifest 332 box cha và
trang review cũ giữ nguyên; không chọn theo điểm OCR hoặc đề xuất đúng/sai của AI.

File khóa: `results/t4/ocr_preparation/cohort_reduced_v1.json`, chứa toàn bộ
332 ứng viên theo thứ tự chính/dự phòng. Gate tái tính danh sách từ manifest cha
và seed, không chỉ tin số lượng hoặc hash khai báo. Trang duyệt:
`results/t4/ocr_preparation/review_reduced_v1.html`.

Mặc định CLI hiện dùng cohort này và export `t4_transcript_review_reduced_v1.json`.
Chạy val chỉ cần 15 quyết định đang hoạt động val; chạy test cần cả val và 47 test
được giải quyết. Box dự phòng chưa dùng không cần duyệt. Export gắn cohort SHA,
không nhận bản đề xuất AI hoặc export của cohort khác.

Chỉ lấy dự phòng khi các box trước đó được người duyệt loại vì crop cắt/mờ,
không đọc được, nhiều dòng hoặc che thông tin; ghi lý do cho từng box. Không
loại vì OCR khó, kết quả kém hoặc dấu/bút che nhiều. Nếu hết dự phòng, giữ lý do
và báo ảnh không có dòng đọc được, số box cuối có thể nhỏ hơn 62; không gán CER0.
Không quay lại chọn box khác sau khi xem OCR. Giữ cùng quyết định crop/ref cho
clean/none/Telea/NS/white và các mask source hợp lệ.

```powershell
code/.venv/Scripts/python.exe -B code/experiments/t4_cer.py
# Sau khi người duyệt xác nhận đủ điều kiện:
code/.venv/Scripts/python.exe -B code/experiments/t4_cer.py --run --split val --output-dir results/t4/cer_reduced_val_v1
code/.venv/Scripts/python.exe -B code/experiments/t4_cer.py --run --split test --output-dir results/t4/cer_reduced_test_v1
```

Báo cáo check-only mặc định mới: `results/t4/step6_reduced_readiness.json` (không
ghi đè). Số liệu CER sau này chỉ là tập con, không đại diện mọi dòng; bút mức1
test chỉ có một box/màu, chưa đủ kết luận theo mức. Phiên5 không thay đổi. Các
mục 81/251 và lệnh cũ phía dưới là lịch sử giao thức full-cohort, không là mặc
định hiện hành nữa. Chưa chạy OCR/CER cho cohort nhỏ.

Ngày 04/11/2026. **Chỉ chuẩn bị code/test; chưa có CER dữ liệu thật.**
Không sửa giao thức/code/số liệu đã khóa trong phiên 5. Không dùng bộ ⑤.

## Điều kiện trước khi chạy

1. Bộ ⑥, phiên 4 và phiên 5 còn đúng hash. Kiểm cả artifact pilot val để dùng
   mask/output đã đo, không sinh detector mới hay đổi radius sau test.
2. OCR thật Tesseract portable 5.5.3.20260724; model fast vie+eng, PSM7/OEM3,
   cùng binary/model/DLL SHA, wrapper SHA và pytesseract version như phiên 4.
   Chế độ kiểm tra chỉ gọi list-langs, không OCR ảnh test. Không có mock/fallback.
3. File `results/t4/ocr_preparation/t4_transcript_review.json` là export schema1
   từ trang review, có tên người duyệt và từng crop ID/hash nhãn nguồn đúng.
   Trạng thái human_confirmed có transcript không rỗng; excluded có lý do;
   pending không được vào phép đo. Bản ai_assisted_proposal bị từ chối.
4. Chạy val cần giải quyết đủ 81 quyết định val; chạy test cần giải quyết cả
   81 val và 251 test. Train không bắt buộc duyệt. Excluded không có điểm CER;
   human_confirmed được đưa vào cohort, mọi crop cùng method/branch giữ chung ref.
   Mỗi split bắt buộc có ít nhất một dòng xác nhận; không mặc định tất cả hợp lệ.
5. Người duyệt thực sự phải đọc ảnh sạch; chương trình kiểm schema/provenance,
   **không thể chứng minh một người đã nhìn ảnh chỉ qua tên/trạng thái JSON**.
   Không đổi nhãn theo output OCR test. AI gợi ý cần người xác nhận phù hợp.

## Phép đo chuẩn bị

- Chỉ dùng box/padding2/native grayscale đã khóa ở crop_manifest. Replay crop
  sạch/hỏng phải đúng pixel với PNG phiên4. Không resize/CLAHE/nhị phân thêm.
- Cùng transcript người duyệt cho clean, none, Telea, NS, fill_white; không lấy
  OCR(clean) làm GT. Dùng wrapper OCR thật của nhóm, không tự viết engine khác.
- Mask dự đoán: ứng viên dấu cho v11, màu xanh cho v12 blue. Bút đen không có
  prediction, chỉ GT chẩn đoán. Không dùng mask black toàn trang để xóa chữ ký.
- Dùng mask đã lưu ở pilot val/official test. Replay bốn full-page output phải
  khớp checksum phiên3/5 trước crop/OCR. Không tự giãn mask, radius3 không đổi.
- Clean/none OCR tính một lần/crop rồi dùng lại trong bảng riêng từng mask source;
  không cộng gộp hai nhánh để nhân đôi cỡ mẫu. Clean là baseline OCR, không là GT.
- Normalization dùng `ocr.metrics.normalize_text`: NFC và gộp whitespace;
  giữ chữ hoa/thường, dấu câu, dấu thập phân, đơn vị. Dùng edit_distance chung.
- CER = **tổng edit errors / tổng ký tự reference đã normalize**, không macro
  CER dòng. Báo n_sources/n_lines/total_errors/total_reference_characters, riêng
  mức/màu/mask_source và tổng mức. CER có thể >1, không cap.
- Delta = CER(method) − CER(none), âm nghĩa ít lỗi hơn; báo thêm delta vs clean.
  Mọi phương án phải cùng crop IDs và reference; khác cohort bị từ chối.
- Hypothesis OCR rỗng vẫn chấm đủ lỗi xóa. Engine exception/shape/hash sai dừng
  toàn lượt; giữ partial output nhưng không completion, không bỏ mẫu lỗi âm thầm.
- Trước OCR ghi config có cohort/ref/seed không áp dụng, engine/model, code/input
  hash. Không overwrite output, không chọn method riêng theo test. Sau đo kiểm
  lại review/runtime/hash và lưu completion kèm checksum.

## Chạy kiểm tra, chưa đo

Từ gốc repo:

```powershell
code/.venv/Scripts/python.exe -B code/experiments/t4_cer.py
```

Ghi `results/t4/step6_readiness.json`, từ chối ghi đè báo cáo có sẵn. Muốn kiểm tra
lại dùng `--readiness-report results/t4/step6_readiness_after_review.json`.
BLOCKED_TRANSCRIPTS không phải kết quả CER và không tạo thư mục benchmark.
Thiếu/bị đổi file prerequisite hoặc schema review sai sẽ báo lỗi, không bỏ hash.

## Lệnh đo sau khi đủ điều kiện (chưa chạy)

```powershell
code/.venv/Scripts/python.exe -B code/experiments/t4_cer.py --run --split val --output-dir results/t4/cer_val_v1
code/.venv/Scripts/python.exe -B code/experiments/t4_cer.py --run --split test --output-dir results/t4/cer_test_v1
```

Không dò engine/normalization/method theo val hoặc test trong baseline này. Val
là kiểm tra quy trình bằng nhãn đã duyệt, không phải mở lưới tham số mới.
Nếu cần đổi giao thức phải khai báo version khác trước test, giữ lịch sử.

Khi chạy thật sẽ tạo config.json, review_export.json, coverage.csv, exclusions.csv,
per_crop.csv (ref/hypothesis/raw & normalized, errors/chars/CER/pixel hash),
summary.csv, README và completion. Coverage giữ cả mẫu không có vùng chữ đủ
điều kiện, không gán CER=0. Bản raw review/transcript/result nhận riêng ngoài Git;
không tự commit ảnh, binary/model, dữ liệu hay transcript.

## Mức kiểm chứng hiện tại

Test dùng schema/chuỗi tổng hợp có đáp án biết trước để kiểm gate và phép tính,
không giả OCR hoặc báo fixture thành benchmark. Kiểm gate thật trên máy đã xác
nhận runtime/mô hình còn khớp; export người duyệt hiện chưa có. Chưa chạy nhánh
OCR/CER end-to-end trên cohort được người xác nhận. Sau khi có nhãn cần chạy val,
đối chiếu ref/hyp và tổng errors/chars trước khi chạy test chính thức.
