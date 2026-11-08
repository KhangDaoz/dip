# T4 — Phát hiện và xóa dấu/nét bút: giao thức phiên 1

Ngày 04/11/2026. **Chỉ đã kiểm kê dữ liệu, chưa có benchmark T4.**
Code kiểm kê: `code/experiments/t4_marks.py`; test hợp đồng:
`code/tests/test_t4_marks.py`. Bằng chứng: `results/t4/step1_inventory.json`,
`step1_samples.csv`, `step1_ocr_pool.csv`, `step1_tests.xml`.
Không dùng bộ ⑤, không sửa dữ liệu gốc hoặc code/kết quả T1–T3.

## 1. Dữ liệu đã kiểm tra thực tế

Root `data/6_synthetic/`, manifest v2 SHA-256
`05dd1a8da9268610d5194e334d7173716865bc14ac6d2da333f9fdb8e4dcd8ad`.
Checksum list v2 SHA
`39dbf3164f0e426beae4b2884bc0a71c137ccc2953651b3307f77179abd503fc`.
Phiên bản có sửa lóa, nhưng T4 chỉ chọn v11/v12, không dùng v10/v14.

| Nhánh | Train | Val | Test | Mức test |
|---|---:|---:|---:|---|
| v11 single_stamp | 60 | 10 | 30 | 10 ảnh/mức 1,2,3 |
| v12 single_pen | 60 | 10 | 30 | mức 1: 8 xanh + 7 đen; mức 2: 8 xanh + 7 đen |

Cùng 100 nguồn, split 60/10/30. Không suy split từ chữ TRAIN trong tên nguồn
VAIPE. Mỗi nguồn có đúng một v11 và một v12. Không có cùng nguồn ở cả ba mức
dấu: đây là benchmark theo mức với các nguồn khác nhau, KHÔNG là đường cong
ghép cặp như T3. Không sinh thêm biến thể hoặc đổi split trong T4 mặc định.

### Phạm vi kiểm tra phiên 1

- SHA 1.400 file (7 file/mẫu): ảnh hỏng, clean, clean_binary, label JSON,
  stamp_mask, pen_mask, glare_mask, đối chiếu danh sách checksum được pin.
- Header ảnh clean/degraded đúng RGB/kích thước; chưa decode các ảnh màu để
  chạy xử lý. Decode GT masks và clean_binary, kể cả test, CHỈ để kiểm kê kiểu,
  kích thước, giao vùng annotation; không gọi detector, inpaint hoặc OCR.
- Cả 200 mẫu H, H_clean_to_degraded, H_degraded_to_clean đều identity;
  không xoay/phối cảnh/bóng/mờ/lóa/nén, exposure_gain=1. Mask active đúng một
  loại; mask active không rỗng, các mask stamp/pen/glare inactive rỗng.
- Mask uint8 H×W, chỉ 0/255, cùng frame ảnh hỏng/clean; đường dẫn label và
  manifest khớp. V11/v12 cùng nguồn có clean và clean_binary giống hash.
- Box dùng [x,y,w,h]. boxes_degraded có polygon và metadata bổ sung, không
  so cả dict bằng tuyệt đối; ID/text/label/mapping và polygon kiểm identity.
- 8.438 lượt box qua 200 mẫu = 4.219 box nguồn, lặp qua v11/v12.
  Không có bbox không hợp lệ hoặc trường text trống trong bản đang có.
  Không đồng nghĩa transcript đúng hoặc box luôn là một dòng OCR hoàn chỉnh.

### GT là mask mực được tiêm, không phải mask màu nhìn thấy

`stamp_mask` / `pen_mask` đánh dấu alpha mực tổng hợp > 0,15, không phải
toàn bộ hình tròn/bbox dấu và không phải mọi pixel đỏ/xanh quan sát được.
Mực multiply có thể đè lên pixel chữ đen nhưng gần như không đổi màu pixel đó;
detector màu không thể luôn phục hồi toàn bộ support GT. Tán màu ngoài ngưỡng
alpha có thể vào prediction nhưng không vào GT. Nêu giới hạn khi giải thích IoU.
GT synthetic biết support mực nhưng KHÔNG là nhãn chữ in được người vẽ pixel.

## 2. Các nhánh phát hiện phải báo tách riêng

| ID | Prediction | Ý nghĩa và phạm vi |
|---|---|---|
| red_color | ink_masks(BGR)['red'] | baseline thu nhận màu đỏ, không gọi là detector dấu hoàn chỉnh |
| stamp_candidate | split_stamp_vs_text(red)[0] | detector vành dấu bảo thủ hiện có; primary stamp detector |
| blue_color | ink_masks(BGR)['blue'] | baseline bút xanh trên nhóm v12 blue, chưa phải detector chữ ký tổng quát |
| black_pen | chưa có detector chuyên biệt | báo unsupported, KHÔNG dùng mask black toàn trang hoặc mask rỗng thay thế |

`split_stamp_vs_text` chỉ nhận vành kín tròn/elip cô lập, không tự nhận chữ giả
hay ngôi sao bên trong dấu. IoU thấp có thể là giới hạn thiết kế thật; không
được điền kín ellipse để có điểm cao mà vẫn gọi là thuật toán cũ.
Khóa SHA ink.py hiện tại ở phiên 2 (đang có sửa v5 local chưa commit);
không trộn code đầu/giữa/cuối lần đo. Phiên 1 không đánh giá hay chọn v4/v5.

Train dùng kiểm tra và sửa lỗi thực thi; val dùng chọn tham số nếu có lưới nhỏ
được tuyên bố TRƯỚC khi xem điểm. Baseline là code hiện có. Mọi phiên bản thay
detector phải là cấu hình/version khác, có test và kiểm lại val trước test.
Không thay chữ ký hàm, không chỉnh ngưỡng sau khi xem test.

### Độ đo phát hiện

Trên GT dương: IoU, precision, recall từng ảnh và macro theo ảnh; báo theo mức,
loại/màu và tổng, kèm n_sources. Có thể lưu TP/FP/FN để diễn giải. Nếu prediction
rỗng và GT dương: IoU=0, recall=0, precision quy ước 0, ghi empty_prediction=True.
Không gộp bút đen chưa được hỗ trợ vào điểm blue_color để tạo điểm tổng giả.

Không đặt ngưỡng chất lượng nghiệm thu sau test. Task đo hoàn thành khi đủ
đúng mẫu, metric/test/ảnh/provenance đạt, kể cả phát hiện ra chất lượng thấp.

## 3. Đối chứng không tiêm dấu/bút

Lấy clean tương ứng, **một control/source** (100 = 60/10/30), tránh đếm v11/v12
hai lần. Hash clean/clean_binary theo nguồn đã đối chiếu giống nhau.
Control là không có dấu/bút được tiêm trong thí nghiệm, không phải đã được người
duyệt chắc chắn không có logo/màu/dấu thật từ nguồn gốc.

Chạy cùng detector và tham số trên clean. Báo số pixel foreground, tỉ lệ pixel,
số ảnh có bất kỳ prediction và, nếu cần, số thành phần 8-connected không lọc thêm.
Trong mục tiêu phát hiện mực TIÊM, GT control rỗng. IoU/recall ghi N/A khi union
rỗng; không tự cho IoU=1 rồi gộp với dương. Precision trên GT rỗng không dùng để
xếp hạng; bảng negative contamination riêng. Hiện chưa tính bất kỳ điểm nào.

## 4. Hai loại mask trong phép so sánh xóa

### A. Mask dự đoán — nhánh thực tế

- Stamp: stamp_candidate là primary. Có thể thêm red_color như ablation rộng
  đã đăng ký trước, phải ghi có nguy cơ xóa chữ đỏ, không biến thành policy triển khai.
- Bút xanh: blue_color, ghi đây là color baseline.
- Bút đen: không báo kết quả xóa bằng prediction khi chưa có detector.
- Prediction rỗng vẫn giữ ảnh/mẫu và báo no-op; không loại khỏi kết quả.

### B. Mask GT — chẩn đoán, không phải pipeline thực tế

Chạy trên toàn bộ v11/v12, kể cả bút đen, để khảo sát bộ xóa khi biết đúng support
mực được tiêm. Gọi là GT-mask diagnostic, KHÔNG gọi kết quả detector hoặc upper
bound chất lượng: GT-mask có thể chọn cả chữ đen bên dưới, inpainting làm hỏng
chữ, nên không bảo đảm tốt hơn mask dự đoán.

### Bốn phương án và các bất biến

Không xóa / Telea / Navier–Stokes / fill_white, radius mặc định **3 px**.
Phiên 3 mặc định không dò radius; nếu có lý do cần lưới, đăng ký trước rồi chỉ
chọn trên val. Không tự dilation/closing/fill mask. No-removal giữ ảnh nguyên;
ba phương án còn lại cùng mask chính xác cho một sample/nhánh.
Ảnh vào BGR sạch hình học, cùng độ phân giải, chưa CLAHE; không tự thêm khử
bóng hoặc nhị phân hóa trong phép so sánh chính. OCR crop từ BGR hoặc gray
được chốt thống nhất ở phiên 4, không đổi theo method.

Kiểm input không mutate, output shape/type giữ, pixel ngoài mask prediction
hoặc GT tương ứng phải giống đầu vào bit-for-bit. Fill_white =255, không phải
ước lượng nền; so sánh trên đúng ROI chung, không chọn ROI theo prediction.

## 5. Vùng đánh giá ảnh và chữ

Giữ ROI chính **R = GT mask**, chung cho baseline và mọi phương án/mask source.
Trong R, so gray(output) với gray(clean): MAE và MSE/PSNR, báo n_pixels; không
lấy PSNR toàn trang trắng làm kết luận phục hồi chữ. PSNR vô hạn khi MSE=0,
không cap tùy tiện, giữ MSE để tổng hợp rõ. Theo ảnh rồi macro; pooled pixels
nếu có là số bổ sung ghi riêng.

Vùng chữ proxy **R_text = GT mask & clean_binary_foreground**: đánh giá riêng
MAE/n_pixels. clean_binary là Otsu tự sinh, không gọi đây là pixel GT chữ thủ công.
Nếu R_text rỗng: N/A cho metric vùng chữ, vẫn giữ mẫu trong đánh giá mask/vùng R.
Ghi thêm số pixel prediction nằm ngoài GT và giao chữ proxy để báo nguy cơ xóa
nhầm; không tự diễn giải thành xác suất mất chữ hoặc CER.

Kết quả inventory: stamp có 2/100 mẫu không giao mực proxy (0/30 test);
pen có 47/100 (13/30 test). Vì vậy không thể giả mọi chữ ký đều che chữ.
Không loại các mẫu đó khỏi điểm detection hoặc điểm ảnh vùng R.

## 6. OCR — pool ứng viên, chưa phải cohort CER đã duyệt

Chỉ dùng box annotation từ ảnh sạch, không box do OCR/detector/inpaint tìm ra.
Pool đủ điều kiện: bbox nguyên dương trong frame, text không trống,
`count(GT_mask & clean_binary trong bbox) > 0`. Các box giao mask chỉ ở khoảng
trắng được ghi nhận nhưng không vào pool chữ bị che chính.

| Nhóm | Train | Val | Test | Tổng box ứng viên chữ bị che |
|---|---:|---:|---:|---:|
| Dấu | 265 | 49 | 149 | 463 |
| Bút | 175 | 32 | 102 | 309 |

Có 856 lượt box giao mask theo hình chữ nhật; 772 trong số đó giao cả mực proxy.
Test: 30/30 ảnh dấu có ứng viên; chỉ 17/30 ảnh bút có ứng viên. 13 ảnh bút còn
lại ghi no_eligible_text_region, không gán CER=0. File pool ghi bbox/id/nguồn/
hash text, không xuất toàn bộ transcript ra báo cáo kiểm kê.

Giao thức mặc định phiên 4: chuẩn bị toàn bộ 149+102 box test đủ điều kiện và
val tương ứng để review. Nếu cần giảm công review thì nhóm phải chốt quy tắc
giới hạn cố định theo nguồn/box trước khi xem OCR predictions, lưu cohort mới;
không chọn box dễ sau khi đọc điểm. Cohort cuối khóa chỉ sau review transcript,
trước lần OCR test chính thức. Không tuyên bố có 251 transcript đã xác nhận.

Crop full bbox annotation + padding cố định 2 px mỗi cạnh, clip tại frame;
lưu bbox crop thực và dùng y hệt trên clean, degraded/no-removal và mọi method.
Không OCR riêng các pixel của mask hay so các crop prediction khác nhau.
Không gộp box chồng nhau thành dòng mới mà vẫn tái dùng transcript cũ.

Trước CER, người đọc kiểm clean crop và text nguồn: số, dấu tiếng Việt, dấu chấm/
phẩy, đơn vị, crop bị redaction hoặc nhiều dòng. Review độc lập với kết quả các
phương án; lưu status/reason/reviewer. Chỉ đưa box đã xác nhận vào CER. Không
lấy OCR(clean) hoặc mock làm transcript GT. Model/language/PSM/crop/gray policy
khóa trên train/val, cùng OCR cho mọi method; clean OCR chỉ là đối chứng.

Chuẩn hóa NFC + gộp whitespace, mặc định không lower, không xóa dấu câu, dấu
thập phân hoặc đơn vị. CER corpus = tổng edit distance / tổng ký tự GT; báo thêm
số ký tự/dòng/nguồn và chênh lệch CER sau-trước. Không cap CER ở 1. Khi so method,
dùng cùng cohort và mẫu số; tuyệt đối không loại output rỗng/khó riêng method.
Không coi nhiều dòng cùng nguồn hoặc v11/v12 cùng nguồn là các tài liệu độc lập.

## 7. Đầu vào còn thiếu và giới hạn kết luận

- Có code OCR chung, nhưng inventory không tìm thấy tesseract trên PATH/vị trí
  Windows thông dụng; pytesseract/paddleocr/vietocr đều chưa cài trong venv này.
  Chưa xác minh traineddata tiếng Việt/model weights. Phiên này không cài hoặc tải.
- Có text trong tất cả box nhưng chưa có evidence human_confirmed cho cohort T4.
  Cần người đọc duyệt hoặc bằng chứng duyệt nguồn đủ tin cậy; không tự nâng trạng thái.
- Không có detector nét bút đen độc lập. Không gán việc phát hiện bút đen là đã đạt.
- Ứng viên vành dấu chưa là full-stamp detector; bảo vệ chữ đỏ thật cần T5,
  không có trong synthetic stamp này. Không mở lại bộ ⑤.
- Bộ 1 đã sàng lọc trước nhưng chưa có mẫu dấu/viết tay dương tin cậy; không
  bịa kết quả ảnh thật. V11/v12 đủ cho phần synthetic, kết luận có phạm vi.

Chưa CER thật thì chỉ được hoàn thành tiểu phần IoU/ảnh xóa, không tick toàn bộ
T4 có CER. T3 template dẫn xuất KHÔNG được dùng thay GT v11 trong T4.

## 8. Đầu ra dự kiến và điều kiện chuyển phiên

Phiên 2: code/test metric + train overlay; phiên 3: bốn phương án train/val;
phiên 4: engine/crop/transcript review; phiên 5: bảng detection/restoration test;
phiên 6: CER nếu đủ đầu vào; phiên 7: replay/ảnh/README/nghiệm thu.

Mỗi lần đo lưu config trước pixel test: manifest/checksum assets, source/split,
code ink/inpaint/metric, detector params/mask policy/radius, crop cohort và text
review hash, runtime/OCR model/lang. Output mới, không ghi đè history. Lưu lỗi
theo sample, dừng khi checksum/geometry/GT sai, không im lặng bỏ mẫu.
Thiếu transcript chỉ chặn CER; không tự chế dữ liệu hoặc đánh dấu phần đó đạt.

Lượt kiểm kê này có 19 test hợp đồng đạt, không có IoU/PSNR/CER T4.
Chạy lại kiểm kê (chọn output mới nếu đã có):

```powershell
code/.venv/Scripts/python.exe code/experiments/t4_marks.py --data-dir data/6_synthetic --output-dir results/t4/reinventory
```

Không cập nhật tiến độ T4 hoàn thành ở phiên 1. Chưa commit/push.

## 9. Phiên 2 — phép đo và TRAIN sanity đã triển khai

Code tách riêng ở `code/experiments/t4_detection.py`, giữ nguyên script kiểm kê
phiên 1 để SHA của bằng chứng cũ còn đúng. Không sửa chữ ký hoặc nội dung các
hàm trong cleaning/ink.py. Mọi ngưỡng dùng nguyên mặc định, nhận diện phiên bản
ink v5 local bằng SHA trong config; không khẳng định đây là version đã push.

Chọn trước khi decode ảnh màu: source_id train nhỏ nhất cho mỗi ô
(stamp level 1/2/3) và (pen level 1/2 × blue/black). Cụ thể:

| Ô | Mẫu |
|---|---|
| dấu 1 | VAIPE_P_TRAIN_1093_v11 |
| dấu 2 | VAIPE_P_TRAIN_1012_v11 |
| dấu 3 | VAIPE_P_TRAIN_1069_v11 |
| bút xanh 1 | VAIPE_P_TRAIN_1069_v12 |
| bút xanh 2 | VAIPE_P_TRAIN_119_v12 |
| bút đen 1 | VAIPE_P_TRAIN_1103_v12 — unsupported |
| bút đen 2 | VAIPE_P_TRAIN_1012_v12 — unsupported |

Có 5 nguồn train khác nhau; control clean một lần/source, áp cả 3 detector.
Tổng 8 dòng dương (3 dấu × 2 detector + 2 blue × 1), 15 dòng control và 2 dòng
unsupported không có điểm IoU. Không tự thêm zero-mask để chấm bút đen.

Preflight kiểm pin manifest/checksum list và SHA đầu vào đã chọn so cả danh sách
phát hành lẫn bằng chứng phiên 1. Config lưu source/split/selection rule/code/
runtime/hash trước pixel; output mới, không ghi đè. Không có CLI chọn val/test.
Mask prediction lấy trực tiếp từ code hiện có, kiểm partition stamp/keep không
thêm pixel, frame/type đúng và không mutate ảnh. Mask black trả từ ink_masks
không được đưa vào detector output hoặc dùng xóa.

IoU/precision/recall được test bằng ma trận nhỏ có đáp án thủ công. GT dương mà
prediction rỗng: cả ba bằng 0 theo giao thức. Negative metrics tách riêng,
IoU/precision/recall = N/A (ô CSV trống), FP thành phần 8-connected không lọc.
Macro theo ảnh, không trộn controls hoặc pooling pixel để tạo điểm dương.

Kết quả sanity không phải nghiệm thu chất lượng:
red_color mean IoU 0,487730 trên 3 mẫu; stamp_candidate 0,093361 trên 3 mẫu
(2 mask rỗng); blue_color 0,853369 trên 2 mẫu. Đây là các tập khác nhau, không
dùng bảng này xếp hạng detector. Controls: 0/5 ảnh có prediction cho cả 3 nhánh,
không chứng minh an toàn ngoài tập nhỏ này. Mọi FP dương bằng 0; precision macro
stamp_candidate =1/3 là do quy ước hai prediction rỗng có precision=0, không phải
phát sinh nhiều FP. Stamp level1 red-color IoU chỉ 0,019496: màu nhạt/latent GT
là giới hạn thật cần giữ để phân tích, không sửa ngưỡng cho điểm đẹp.

Đã xem crop native của cả 3 dấu và 2 bút xanh: màu overlay xanh=TP,
tím=FP, cam=FN. Ứng viên vành dấu bỏ sót chữ/ngôi sao và có thể giữ lại toàn
bộ dấu không chắc chắn; không tự fill ellipse hoặc giãn mask. Bút xanh còn thiếu
một số support GT ở nét mảnh/chỗ giao chữ đen. Không có thao tác xóa dấu/OCR.

Đầu ra: `results/t4/train_detection/` gồm config, positives/negative_controls,
summary tách riêng, unsupported, mask PNG và overlay PNG đúng kích thước,
native_overlay_review.png + crop provenance và completion hashes.
`results/t4/step2_verification.json` ghi kiểm lại 8 điểm từ mask lưu, replay 15
controls khớp hash, macro summary và code/input SHA không đổi. Tổng 167 test liên
quan đạt (35 test T4 inventory/detection +132 ink/stamp); chưa phải full repo suite.

```powershell
code/.venv/Scripts/python.exe code/experiments/t4_detection.py --data-dir data/6_synthetic --inventory results/t4/step1_inventory.json --output-dir results/t4/train_detection_repeat
```

Có thể sang phiên 3 để đo bốn phương án xóa với cùng mask; nhánh prediction
rỗng phải báo no-op, nhánh GT là diagnostic riêng. Chưa dò tham số, chưa chạy
detector trên val/test, chưa đánh giá CER hoặc công bố T4 hoàn thành.

## 10. Phiên 3 — pilot xóa trên TRAIN/VAL

Đã triển khai `code/experiments/t4_restoration.py`; không sửa detector hoặc
`remove_marks`. Chọn 7 train của phiên 2 + đủ 20 val (10 v11, 10 v12), không test.
Radius duy nhất 3, không có lưới hoặc chọn tham số. Config và preview sample IDs
khóa trước khi đọc ảnh xử lý; pin code/data theo phiên 1–2.

27 mask GT diagnostic +21 mask dự đoán (13 stamp_candidate, 8 blue_color) =48
mask cases ×4 method =192 dòng. 6 mẫu bút đen (2 train,4 val) chỉ có GT diagnostic,
ghi riêng unsupported dự đoán. Không thêm ablation red_color hoặc detector đen.

Mỗi mask case dùng cùng ảnh gốc BGR cho none/telea/ns/fill_white, cùng mask không
giãn, cùng R=GT và R_text=GT & Otsu(clean). Kiểm tra output shape/type/không alias,
input/mask không mutate, pixel ngoài mask không đổi và empty prediction no-op.
Có 10 mask prediction rỗng (2 train,8 val), giữ đủ cả 40 dòng tương ứng.
Baseline none lặp qua các nhánh được đánh dấu rõ là cùng ảnh, không độc lập.

Mỗi dòng lưu MAE/MSE/PSNR gray trên R và R_text, số pixel, SHA mask/ROI/output,
trạng thái no-op và bảo toàn ngoài mask. R_text rỗng ghi N/A; summary macro theo
ảnh có denominator riêng n_text_proxy_samples, không thay N/A bằng 0.
PSNR summary từ MSE trung bình theo ảnh, không phải trung bình dB; exact match
giữ infinity trong CSV. Không dùng toàn trang trắng hoặc PSNR để chọn cách xóa.

Đã tính lại bảng độc lập từ CSV, kiểm 48 cases đủ bốn method/cùng mask/GT ROI và
cohort; none khớp input; 10 no-op cases khớp cả bốn output SHA. Replay 14 mask
cases đã chọn làm hình (56 output) có SHA khớp, không phải replay toàn bộ192.
328 test liên quan đạt (54 T4 +274 ink/stamp/inpaint), không fail/skip.

Ảnh: sample nhỏ nhất theo source_id mỗi split cho dấu mức1/3, bút xanh/đen mức2.
Crop native 320×240 (clip nếu ảnh nhỏ hơn), không resize. Crop đầu đặt tâm box
dài có thể không hiển thị rõ phần bị che: giữ bản đầu, bổ sung renderer riêng
`t4_restoration_review.py` lấy một pixel GT∩ink thật trong box bị che nhiều nhất.
File `_focus_native.png` nên dùng để review; `focus_previews.json` lưu crop và
hash/replay. Sửa vị trí crop không đổi số liệu hoặc runner/config lịch sử.

Quan sát trên val (chưa phải kết luận test): với GT dấu n=10, MAE vùng R của
none/Telea/NS/white ≈68,70/24,95/27,09/20,10 nhưng MAE chữ proxy tăng từ24,58
lên88,74/82,56/174,10. Dùng GT mask không bảo đảm giữ được chữ: support mực dấu
chồng lên chữ đen, inpaint/tô trắng có thể xóa chính nét chữ đó. Còn vệt màu nhạt
ngoài support alpha>0,15 là giới hạn mask, không tự giãn để xóa sạch màu.

Với blue prediction val n=6, chỉ3 ảnh có R_text; chỉ diễn giải metric chữ trên3
ảnh này. Black GT val n=4, chỉ2 có R_text. Đây là proxy ảnh Otsu, không phải CER.
Không chọn method chung, không đổi radius, không suy ra mask GT là upper bound.

Đầu ra `results/t4/train_val_restoration/`: per_image.csv, summary.csv,
mask_cases.json, mask PNG, preview gốc/focus, unsupported.csv, config/completion.
Kiểm chứng: `results/t4/step3_verification.json`, test `step3_final_tests.xml`.

```powershell
code/.venv/Scripts/python.exe code/experiments/t4_restoration.py --output-dir results/t4/train_val_restoration_repeat
code/.venv/Scripts/python.exe code/experiments/t4_restoration_review.py --results-dir results/t4/train_val_restoration_repeat
```

Output mới bắt buộc. Chưa OCR/CER, chưa test chính thức, không thay mặc định
pipeline. Có thể sang phiên4 chuẩn bị OCR/transcript; không tick T4 hoàn thành.

## 11. Phiên 4 — OCR thật và bản duyệt transcript

Code chuẩn bị: `code/experiments/t4_ocr_prepare.py`; đầu ra
`results/t4/ocr_preparation/`. Không sửa wrapper OCR/metric của Khang hoặc thuật
toán làm sạch. Đã cài pytesseract 0.3.13 trong venv hiện có; Tesseract Windows
portable trích xuất từ installer vào `code/.venv/tesseract_t4/`, không cài hệ thống
hoặc đổi PATH. Bộ cài chạy thường bị hủy ở bước cài đặt, không được dùng cách ép
quyền; chỉ giải nén archive bằng công cụ 7-Zip chính thức trong venv.

### Engine/profile đã kiểm chứng, không tuyên bố giống môi trường Khang

- Binary Tesseract **v5.5.3.20260724**, leptonica1.87.0; nhóm ghi benchmark5.5.4,
  không phải cùng binary và không lấy kết quả K0 làm kết quả OCR T4 tiếng Việt.
- Gọi wrapper `ocr.engines.ocr_line`, `engine='tesseract'`, `lang='vie+eng'`.
  Wrapper truyền `--psm 7`, không truyền OEM; OEM mặc định3 auto. Model fast
  chỉ có LSTM; không ghi sai rằng đã gọi cờ `--oem 1`.
- Mô hình `tessdata_fast` chính thức, commit
  `87416418657359cb625c412a48b6e1d6d41c29bd`, eng và vie. Đã kiểm list-langs,
  hash binary, models và các DLL; profile chi tiết nằm trong config.json.
- Crop grayscale nguyên độ phân giải, padding2 cố định, không resize/CLAHE/
  nhị phân hóa/tự tìm box mới. NFC + gộp whitespace khi chấm sau này; giữ nguyên
  hoa/thường, dấu câu, số, dấu thập phân và đơn vị.
- TESSDATA_PREFIX/tesseract_cmd chỉ đặt trong process chạy script, không đổi
  PATH/env hệ thống. Runtime/model nằm trong .venv bị Git bỏ qua.

Nguồn: [Windows build do UB Mannheim cung cấp](https://github.com/UB-Mannheim/tesseract/wiki),
[installer5.5.3](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.3/tesseract-ocr-w64-setup-5.5.3.20260724.exe),
[họ model/ngôn ngữ chính thức](https://tesseract-ocr.github.io/tessdoc/Data-Files),
[model fast tại commit đã khóa](https://github.com/tesseract-ocr/tessdata_fast/tree/87416418657359cb625c412a48b6e1d6d41c29bd).
Installer SHA đã đối chiếu GitHub release digest; model URL/hash lưu config.

### Cohort ứng viên và crop

Tái kiểm annotation/GT overlap theo phiên1: **772 box** (440 train,81 val,251test),
không chọn theo OCR predictions. Tạo 1.544 PNG gray: clean/degraded cho mỗi box.
Mỗi record giữ bbox annotation, bbox crop đã pad/clip, frame, hash ảnh/nhãn và
text nguồn. Clean và degraded dùng cùng rectangle. Các output Telea/NS/white
chưa materialize crop toàn bộ ở phiên4; phiên CER bắt buộc dùng hàm same_crop
và bbox đã lưu cho mọi method, không để engine hoặc method đổi vùng crop.

Đã kiểm đủ hash/kích thước 1.544 crop; replay hai crop của một record ở mỗi split
khớp pixel. Test chỉ được cắt để chuẩn bị/duyệt annotation, **chưa gọi OCR trên
test**, chưa có CER. 13 cuộc gọi OCR thật: 3 train +3 val × clean/degraded,
cộng một ảnh chữ Việt tổng hợp tự tạo để kiểm runtime. Probe đọc được câu có dấu
nhưng không dùng làm tuyên bố accuracy. Smoke còn có lỗi nhận dạng số/dấu;
OCR(clean) không được ghi đè transcript nguồn hoặc chuyển thành GT.

### Bản đối chiếu để người đọc duyệt

Mở `results/t4/ocr_preparation/review.html` bằng trình duyệt, giữ thư mục crops
cạnh file. HTML offline, không gọi API, không hiển thị OCR prediction để tránh
lấy dự đoán của engine làm gợi ý sửa GT. Mặc định val81, có train/test/all và
phân trang20. Ảnh clean crop native cuộn ngang; text nhãn có thể sửa.

Tất cả review_status ban đầu **pending**. Người đọc nhập tên, kiểm crop/text,
chọn đã xác nhận hoặc loại có lý do, rồi tải `t4_transcript_review.json`.
Lưu định kỳ và dùng Nạp bản đã lưu để tiếp tục; import kiểm crop_id/text hash
khớp. Mỗi file dùng một reviewer, không trộn người duyệt qua đổi tên trên UI.
Có `review_template.json` tương đương cho người muốn sửa JSON trực tiếp.
Không được dùng trạng thái AI tự xác nhận thay xác nhận người đọc.

Đặc biệt kiểm tên thuốc, hàm lượng, chữ số1/7, dấu chấm/phẩy và dấu tiếng Việt.
Nếu box nguồn gồm nhiều dòng hoặc vùng bị redaction làm text không đọc được,
ghi lý do, không mặc định nhận transcript. Cohort CER chính thức chỉ được khóa
sau khi nhập bản duyệt; chưa coi cả251 box test là GT đã được chứng nhận.

### Bằng chứng và đầu vào còn chờ

`config.json`, `crop_manifest.json`, `completion.json`, `ocr_smoke.json` và
`results/t4/step4_verification.json`; 68 test liên quan đạt (63 T4 +5 OCR metric),
không fail/skip. JS của trang duyệt qua kiểm tra cú pháp Node; chưa tuyên bố
đã thực hiện thao tác duyệt trên trình duyệt hoặc có transcript người duyệt.

Engine/model và cơ chế crop đã sẵn sàng trên Windows hiện tại. **Còn chờ bản
transcript người dùng duyệt**, quyết định loại/giữ box khó trước khi tính CER.
Không thiếu mask mới, không cần tải bộ⑤. Có thể làm phiên5 IoU/ảnh xóa độc lập;
phiên6 CER vẫn chặn nếu transcript chưa được xác nhận. Chưa tick toàn bộ T4.

```powershell
code/.venv/Scripts/python.exe code/experiments/t4_ocr_prepare.py --output-dir results/t4/ocr_preparation_repeat
```

Chỉ chạy được khi runtime.json/model/binary local đã có và đúng SHA. Người khác
cần chuẩn bị cùng binary/model hoặc khai báo profile khác và nghiệm thu lại;
không copy venv giữa máy rồi mặc định tương thích. Không cần Paddle/VietOCR cho
baseline này. Không commit binary/model hoặc các crop vào Git.

## 12. Phiên 5 — giao thức khóa trước pixel benchmark test

Runner mới `t4_official.py`, renderer `t4_official_report.py`; giữ nguyên code
pilot và mọi số liệu phiên trước. Không dò tham số, không chọn phương án theo test.
Kiểm tra checksum manifest, checksum list, inventory, config pilot, mã thuật toán
và toàn bộ file của 60 mẫu test bằng byte trước khi decode ảnh cho lần đo này.
`config.json` được ghi đóng trước detector/inpaint; lưu snapshot code và phiên bản
Python/NumPy/OpenCV/Pillow/Matplotlib. Nếu sai hash hoặc lỗi mẫu: dừng, không bỏ mẫu,
không tạo completion thành công. Output đã có bị từ chối, giữ lịch sử.

- Dấu đỏ: đủ 30 test, 10/mức 1/2/3; detection red_color và stamp_candidate riêng.
- Bút xanh: 16 test, 8/mức 1/2; chỉ baseline blue_color.
- Bút đen: 14 test, 7/mức 1/2; ghi unsupported detector, chỉ GT diagnostic.
- Tổng 76 detection rows, 14 unsupported; mọi nhóm báo mức/màu và tổng macro ảnh.
- 30 clean control/source, ba detector = 90 hàng, không trộn với dương.
- Xóa: 60 GT diagnostic +30 predicted stamp_candidate +16 predicted blue_color
  =106 mask cases, mỗi case bốn phương án, 424 hàng; radius3, không giãn mask.
  Không thêm red_color ablation vào nhánh xóa chính thức.
- ROI GT chung qua bốn phương án và các mask source; text proxy Otsu tự sinh,
  vùng chữ rỗng N/A. Không xóa vẫn lặp trong mỗi case để bảo toàn so sánh ghép cặp.
- Đối chứng xóa: 30 clean ×hai detector chính ×bốn phương án =240 hàng riêng;
  đo thay đổi pixel và lỗi gray so với chính clean, không trộn với ROI GT dương.
- Kiểm input/mask bất biến, output uint8/kích thước, ngoài mask giống bit-for-bit
  ở mọi lần gọi, bao gồm no-op và đối chứng. Không OCR/CER, không dùng bộ ⑤.
- Ảnh minh họa: source đầu theo thứ tự ID ở mỗi ô kind/mức/màu (bảy ô), chọn
  crop native quanh pixel GT giao chữ ở box che nhiều nhất, fallback GT nếu rỗng.
  Overlay detection và crop bốn phương án cùng tọa độ, không chọn theo điểm.
- Rà soát: tính lại detection/negative từ mask đã lưu, tổng hợp lại từ CSV;
  replay bốn output của các case thuộc bảy mẫu minh họa và đối chứng source
  đầu/cuối. Ghi số thực tế, không gọi subset replay là rerun toàn bộ.

Đây là freeze trước **lần đo benchmark T4 test**, không tuyên bố test chưa từng
được xem: inventory phiên1 và crop phiên4 đã decode ảnh; nguồn test cũng đã dùng
ở T1–T3. Các mức trong bộ6 có nguồn khác nhau, không là đường cong severity ghép
cặp. GT alpha>0,15 có thể che chữ đen; GT diagnostic không là upper bound.
Chất lượng ảnh/IoU không chứng minh CER, độ chính xác OCR, T5 hay hiệu quả thực tế.
