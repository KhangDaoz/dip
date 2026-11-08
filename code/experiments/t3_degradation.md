# T3 — Nhị phân hóa theo mức độ hỏng

## Trạng thái hiện tại — nghiệm thu lượt 4, 04/11/2026

Đã chạy đủ 6.600 dòng test, 220 ô tổng hợp × 30 nguồn, kiểm tra độc lập khớp
CSV và hash. Xem `results/t3/README.md`, `completion.json` và
`step4_acceptance_final.json`. Đã nghiệm thu phạm vi đo: 234 test liên quan đạt,
replay 660 dòng qua 3 nguồn định trước khớp ảnh/điểm; không phải replay toàn bộ.
Giữ nguyên số liệu/runner/config đã khóa. Ca Gaussian mạnh có thể F1=0; không
đổi recipe theo test. OCR, T4/T5 và bộ ⑤ không thuộc kết luận nghiệm thu này.

Đã triển khai preflight và chạy thử TRAIN. Lượt 3 bổ sung chế độ test chính thức
theo mục 8; kết quả chạy phải đọc completion.json, không suy từ việc có code.
Gate dấu được giải quyết bằng template chuẩn lớn nhất dùng chung và co giãn
INTER_LINEAR; xem mục 7. Giao thức lượt 1 dưới đây giữ lại để truy vết lý do.

## Giao thức đã chốt ở lượt 1 — 04/11/2026

Đã chốt phạm vi, nguồn, cấu hình và quy tắc đo. Chưa có bộ sinh T3, CLI benchmark,
điểm T3 hoặc hình đường cong. Riêng ghép cặp dấu cần hoàn tất cơ chế kiểm soát
ngẫu nhiên ở lượt 2 trước khi được phép chạy test chính thức.

Code hỗ trợ: `t3_degradation.py`; test: `code/tests/test_t3_degradation.py`.
Bằng chứng local: `results/t3/step1_audit.json`, `results/t3/step1_tests.xml`.
Không dùng bộ ⑤, không thay dữ liệu gốc, không nối pipeline.

## 1. Dữ liệu đã kiểm tra thực tế

- Root: `data/6_synthetic/`, manifest `manifest_v2.json`.
- SHA-256 manifest hiện tại: `05dd1a8da9268610d5194e334d7173716865bc14ac6d2da333f9fdb8e4dcd8ad`.
- 1.500 mẫu; 100 nguồn, 60 train / 10 val / 30 test. Các nguồn khác split
  của VAIPE ban đầu được dùng theo split chung trong manifest, không suy split từ tên ID.
- Chọn đúng một hàng `variant_index=0` của mỗi nguồn để tra đường dẫn clean,
  clean_binary và nhãn gốc. V00 test có thể là phối cảnh: KHÔNG đọc ảnh degraded
  hoặc tái dùng H của v00 để sinh ảnh T3. Metadata gốc chỉ là provenance.
- Đã đối chiếu nhãn source_id/split/sample_id/kích thước và SHA của 300 file
  clean/clean_binary/label với checksums_v1.sha256 và checksums_v2.sha256.
  Đây là kiểm tra byte/metadata, không phải giải mã toàn bộ 100 cặp ảnh.
- Một nguồn train đầu tiên theo ID đã giải mã: kích thước đúng, binary 0/255,
  khớp chính xác Otsu(gray(clean)). Chưa kiểm pixel val/test, chưa chấm F1 T3.
- Test có 15 `test_single_shadow_planar`, 8 `test_single_blur_gaussian`,
  7 `test_single_blur_motion`, 30 `single_stamp`. Không nói rằng hoàn toàn thiếu
  bóng đơn test. Các nhóm này không cung cấp cùng nguồn ở mọi mức nên không dùng
  thay cho thiết kế ghép cặp chính. Không trộn mẫu combo vào đường cong đơn hỏng.

## 2. Thiết kế ghép cặp và số lượt đo

| Nhóm | Mức | Tham số thực |
|---|---|---|
| shadow_planar | 0–3 | darkness 0 / 0,20 / 0,45 / 0,70 |
| shadow_silhouette | 0–3 | cùng darkness, hình bóng khác nhóm planar |
| blur_gaussian | 0–4 | sigma 0 / 1 / 2 / 3 / 4 px; ksize từ hàm sinh |
| blur_motion | 0–4 | radius 0 / 1 / 2 / 3 / 4; lưu ksize thực và angle_deg |
| stamp | 0–3 | radius_frac 0 / 0,08 / 0,11 / 0,14; opacity 0 / 0,55 / 0,75 / 0,92 |

Mỗi nguồn có 22 vị trí đường cong, nhưng chỉ 18 đầu vào vật lý khác nhau theo
thiết kế: 17 biến thể hỏng + 1 clean dùng lại ở 5 điểm mức 0.

| Split | Nguồn | Vị trí đường cong | Điểm × 10 cấu hình | Đầu vào riêng biệt tối đa |
|---|---:|---:|---:|---:|
| train (nếu chạy đủ) | 60 | 1.320 | 13.200 | 1.080 |
| val | 10 | 220 | 2.200 | 180 |
| test chính thức | 30 | 660 | 6.600 | 540 |

Không cần chạy đủ train: sanity mặc định dự kiến dùng 2 nguồn train đầu theo ID,
44 vị trí / 440 dòng điểm / 36 đầu vào riêng biệt; val dùng đủ 10 nguồn để xác
nhận quy trình, KHÔNG dò tham số lại. Tổng dự kiến sanity + val + test:
924 vị trí, 9.240 dòng điểm. Được cache mức 0 nhưng CSV vẫn ghi rõ 5 nhóm dùng
chung input_key; không coi đó là 5 quan sát độc lập. Không gộp điểm của các nhóm.

Ảnh sinh từ BGR sạch, giữ độ phân giải, uint8, cùng hệ tọa độ, H đơn vị.
Mức 0 trả bản sao clean, không gọi hàm hỏng. Không thêm lóa/nhiễu/nén/bút/
xoay/phối cảnh. Chuyển BGR sang gray theo cùng quy ước T2, sau đó áp cấu hình.

## 3. RNG — phần đã chứng minh và điều kiện còn thiếu

Seed cơ sở 42. Token UTF-8:
`42:t3_v1:<source_id>:<group>:<stream>` → SHA-256 → 8 byte đầu little-endian.
Không có level trong token. Mỗi mức khởi tạo RNG mới từ seed, không dùng chung
RNG bị tiến trạng thái giữa các mức. Lưu token/seed và tham số thực khi triển khai.

- Shadow: chỉ định shadow_type; `_generate_shadow_map` không phụ thuộc darkness.
  Khởi tạo lại RNG giữ bản đồ S(x,y), thay darkness. Test ảnh fixture kiểm coverage,
  loại bóng, darkness, kích thước và tính tối dần; chưa phải đánh giá ảnh thật.
- Gaussian: không có góc ngẫu nhiên khi chỉ định kind; level là sigma.
- Motion: góc là lần uniform đầu tiên; khởi tạo lại RNG giữ góc, thay radius/kernel.
  Test fixture đã kiểm góc ở bốn mức. Không gọi kind=random.
- Stamp: truyền center cố định giúp giữ tâm; tuy nhiên `_stamp_alpha` gọi
  `_ink_texture(shape, rng)` trước khi lấy rotation và color. Shape phụ thuộc
  radius nên cùng seed vẫn đổi rotation/color/texture. Test chủ động chứng minh
  bất biến này KHÔNG đạt; việc test đạt nghĩa là phát hiện đúng rủi ro, không phải
  bộ sinh ghép cặp dấu đã sẵn sàng.

**Gate lượt 2:** phải có wrapper cục bộ T3 giữ tâm, góc, màu, hệ số radius jitter
và mẫu texture chuẩn hóa giữa các mức, bằng các luồng RNG độc lập/canvas chuẩn
được tái sử dụng. Không sửa code synthetic chung. Nếu dùng template chuẩn rồi
co giãn, ghi rõ đây là bộ sinh T3 dẫn xuất, không tuyên bố khớp pixel hàm stamp
gốc; ghi nội suy và ảnh hưởng lượng tử/raster. Chốt công thức cụ thể và test
hồi quy ở lượt 2 trước khi đọc test. Không được âm thầm dùng stamp cùng seed
và gọi đó là giữ nguyên kiểu dấu. Nếu chưa đạt gate thì dừng nhánh stamp,
không công bố T3 hoàn thành.

Mức dấu là gói kích thước + opacity, không tách tác động hai yếu tố, không gọi
đó là đường cong chỉ theo độ đậm. Không đòi mask các mức lồng nhau từng pixel.

## 4. Cấu hình đóng băng và độ đo

Đọc bốn file `results/t1_v2/{config,selection}.json`,
`results/t2_v2/{config,selection}.json`; SHA chính xác trong `LOCKS` của script
và báo cáo audit. Kiểm binding T2 → T1 selection. Mất file/sai SHA: dừng,
không dùng mặc định thay thế. Có 10 cặp (5 method × none/t1), giữ params nguyên
văn. T1 winner Closing kernel 81, không tự cập nhật mặc định cleaning.

Điểm chính: F1 mực (255) so clean_binary cùng nguồn. Tính từng ảnh rồi macro
theo 30 nguồn cho mỗi group/level/recipe; không micro gộp pixel, không chọn
winner riêng theo mức, không làm mượt/ép đơn điệu. Có thể lưu precision/recall
để giải thích; chưa bắt buộc DRD và các metric nặng của T2.

Không inpaint hoặc loại màu/dấu trước phép đo này. Dấu thêm vào có thể bị tính
false positive so chữ sạch: đây là độ bền nhị phân hóa trước vật cản, không phải
IoU phát hiện dấu hay chất lượng phục hồi chữ T4. Ngưỡng F1 không là tiêu chí
đạt/rớt thuật toán được đặt sau khi xem test.

Nếu có CI: bootstrap cùng nguồn qua mọi mức và recipe, seed khóa trước,
báo rõ là bất định giữa nguồn của tập này; không coi biến thể là nguồn độc lập.

## 5. Bằng chứng, kiểm tra và điều kiện dừng

Lượt 2 phải khóa manifest/checksum lists, byte SHA 300 references, bốn artifact
T1/T2, code sinh và phụ thuộc liên quan, script T3, metric/binarizer/background,
phiên bản thư viện. Dùng hash hiện tại để lập provenance chưa đủ chứng minh
tương thích: đối chiếu baseline đã khóa và hợp đồng T1/T2 khi dùng lại.
Không bỏ kiểm checksum do manifest đổi; cần bằng chứng và phiên bản mới rõ ràng.

Mỗi biến thể ghi source_id, split, provenance sample/path/hash, group, level,
seed các stream, tham số thực, H đơn vị, dimensions, SHA pixel BGR/gray;
mỗi dòng điểm thêm recipe/params/hash reference/F1. Nhãn gốc v00 phải nằm
trong provenance, không giả H gốc là identity. Không ghi mảng mask vào JSON;
nếu lưu mask thì ghi đường dẫn và hash.

Lượt 2 mới kiểm pixel: decode đúng uint8/BGR/gray, shape; binary chỉ 0/255;
reference đúng Otsu theo quy ước hiện có; generation không sửa input; level 0
khớp byte; chỉ có loại hỏng được giao; output binary đúng quy ước.
Preflight metadata không được báo là các kiểm tra pixel này đã hoàn thành.

Dừng nếu: mất/trùng nguồn, split sai/cross-split, checksum thiếu/đổi, path ra
ngoài root, nhãn sai provenance/kích thước, recipe sai/thiếu, pixel/shape sai,
không kiểm soát được RNG đã cam kết, output đã tồn tại. Không bỏ mẫu lỗi để
vẫn đủ đường cong. Chỉ tạo output mới, không sửa kết quả lịch sử T1/T2.

Lượt 3: khóa toàn bộ contract trước test; tạo config.json, samples.json,
per_image.csv, summary.csv, figure.png/pdf, README.md trong results/t3/ với
phân biệt rõ bằng chứng lượt 1/sanity và kết quả chính thức. Lượt 4 mới nghiệm thu.

## 6. Giới hạn bắt buộc trong kết luận

clean_binary là Otsu tự sinh, chưa phải GT thủ công; F1 có thiên lệch thuận Otsu.
Không suy ra CER/WER/OCR hay hiệu quả trên phiếu màu/chữ đỏ bộ ⑤. Đây là
biến thể kiểm soát từ nguồn bộ ⑥, không phải benchmark toàn bộ 1.500 ảnh hỏng.
Nguồn test đã dùng T1/T2, không phải test độc lập mới. Các mức trên cùng nguồn
phụ thuộc nhau. Đường cong dấu không chứng minh T4 hoặc T5 đã đạt.

## 7. Triển khai lượt 2 — giao thức dấu cụ thể trước test

Tên chính sách: `canonical_max_radius_alpha_v1_linear_resize_integer_fixed_anchor`.
Đây là bộ sinh T3 dẫn xuất, KHÔNG khớp bit các ảnh v11 hoặc hàm `stamp` gốc.

1. Ba stream seed riêng `geometry`, `texture`, `color`, theo token ở mục 3.
2. Geometry lấy một jitter uniform 0,9–1,1; radius chuẩn = 0,14 × cạnh ngắn × jitter.
3. Gọi renderer chung `_stamp_alpha` đúng một lần trên canvas lớn nhất bằng
   stream texture: texture và rotation được đóng băng trong alpha chuẩn này.
4. Tâm lấy theo `_pick_center` trong `medicine_region_clean` nếu có, dùng
   margin dựa trên canvas lớn nhất; làm tròn tọa độ nguyên một lần cho mọi mức.
   Nếu thiếu region thì dùng vùng trung tâm mặc định; lưu center và provenance.
5. Màu lấy một lần từ RED_INK + jitter nguyên [-15,15] trên stream color.
6. Mỗi mức co alpha chuẩn về size = round(size_chuẩn × radius_frac/0,14),
   dùng INTER_LINEAR. Đặt canvas theo anchor `(cx-size//2, cy-size//2)` rồi nhân
   opacity mức 0,55/0,75/0,92 và gọi `_multiply_ink` chung. Không cắt dấu tại mép:
   nếu canvas vượt khung thì dừng. Kích thước và opacity vẫn thay đồng thời.
7. Tâm nguyên là anchor chung; kích thước lẻ/chẵn có sai lệch tâm raster tối đa
   nửa pixel. Nội suy có thể thay nét mỏng và lỗ texture; không giả texture từng
   pixel/mask lồng nhau tuyệt đối. Mẫu texture chuẩn, rotation, màu và jitter
   được giữ, không lấy lại mẫu ngẫu nhiên theo mức.

Các hàm shadow/blur gọi code nhóm với RNG khởi tạo lại từng mức; không dùng
wrapper này cho chương trình synthetic chung hoặc T4 mà không ghi chính sách.

### Chạy từ gốc repo bằng môi trường đã kiểm chứng

```powershell
code/.venv/Scripts/python.exe code/experiments/t3_degradation.py --check-inputs-only --compatibility-evidence results/t1_t2_repro_audit/t1_t2_v2_handoff.zip
code/.venv/Scripts/python.exe code/experiments/t3_degradation.py --train-sanity --compatibility-evidence results/t1_t2_repro_audit/t1_t2_v2_handoff.zip --output-dir results/t3/train_sanity_final
```

Preflight chỉ hash/metadata, không decode ảnh và không tạo output. Cần ZIP bằng
chứng T1 gốc được chia sẻ riêng và bốn file config/selection lịch sử. Không có
cờ bỏ checksum. Pin manifest và cả hai danh sách checksum của phiên bản bộ 6
hiện tại; đối chiếu toàn bộ 300 references và archive T1 để chứng minh code,
runtime và 138 references lịch sử tương thích. Metric phải khớp hash T2.
Các hash mới trong config dùng truy vết, không thay thế phép đối chiếu baseline.

Sanity chỉ lấy hai ID train nhỏ nhất theo chuỗi, không cho người dùng chọn split
test qua CLI. CLI chưa có formal mode. Từ chối mọi output đã tồn tại, kể cả rỗng;
lần chạy lỗi để lại config complete=False và không được coi đã hoàn thành.
`completion.json` là dấu hoàn tất; trường complete trong config luôn là trạng
thái lúc khóa trước chạy. Không ghi đè config sau khi đã đọc pixel.

Mỗi source/group/level được sinh trên bộ nhớ; dùng BinaryReference.score chung
(có tính các metric nội bộ của T2), chỉ lưu F1/precision/recall cho lượt này.
Tổng 44 vị trí × 10 cấu hình = 440 dòng; 5 điểm mức 0/source trùng đầu vào là có
chủ ý, chưa tối ưu cache. CSV lưu SHA prediction/gray; samples.json lưu SHA
BGR/gray/reference, metadata seed/parameters và provenance. Định nghĩa SHA mảng
bao gồm dtype, shape và bytes C-order; không nhầm với SHA file PNG.

Ảnh minh họa: giữ nguyên kích thước, chỉ lưu hai ảnh nguồn và trên nguồn đầu,
năm loại hỏng mức cao nhất cùng hai cấu hình đã chỉ định TRƯỚC khi xem điểm:
`t1__otsu__base`, `none__sauvola__k0.3_window31`. Ảnh nhị phân PNG minh họa
đảo về chữ đen/nền trắng; đầu ra tính điểm vẫn mực=255/nền=0. Đây không phải
việc chọn winner theo sanity. Tổng 17 PNG gốc/toàn trang trước các bản crop review.

Lượt 2 không dùng val để dò lại tham số, không đọc pixel test/val. Trước lượt 3
cần bổ sung formal mode và khóa contract cho tập chạy chính thức; không dùng
điểm sanity trong báo cáo đường cong test.

## 8. Giao thức chạy chính thức lượt 3 — khóa trước khi đọc pixel

```powershell
code/.venv/Scripts/python.exe code/experiments/t3_degradation.py --formal-test --compatibility-evidence results/t1_t2_repro_audit/t1_t2_v2_handoff.zip --output-dir results/t3
```

30 nguồn test đầy đủ, sort source_id, giữ mỗi nguồn ở mọi mức. Không tái tune.
`samples.json` ghi trước toàn bộ 660 vị trí đường cong, provenance/H/seed;
`config.json` ràng buộc SHA samples và nguồn/config/code/reference/runtime trước
khi decode BẤT KỲ pixel val/test. File config không được sửa sau đó. Trong config
complete=False là trạng thái lúc khóa; completion.json đánh dấu kết thúc.
`generated_samples.json` bổ sung actual parameters và SHA pixel sau khi chạy.

Chọn minh họa trước chạy: nguồn val nhỏ nhất theo source_id, mức lớn nhất của
mỗi nhóm, hai recipe t1/Otsu và none/Sauvola31 k0.3 đã chỉ định ở lượt 2. Không
chấm val để chọn tham số, không lấy ảnh test đẹp nhất làm ví dụ. Lưu val_examples.

Test gồm 660 vị trí × 10 recipe = 6.600 dòng. Cache 10 điểm mức 0 mỗi nguồn để
dùng lại cho năm nhóm; 540 đầu vào riêng biệt × 10 = 5.400 phép chấm thực tế.
Mỗi nguồn có 220 dòng, ghi checkpoint CSV sau source; lỗi dừng không bỏ mẫu.
Không resume/ghi đè kết quả dở bằng CLI; kiểm tra lại và dùng output mới khi cần.

Summary có 220 ô, mỗi ô đúng 30 source, mean F1 theo ảnh, sample SD (ddof=1),
min/max và n_sources. **Không tạo CI hoặc dải bootstrap trong phiên bản này**;
SD là độ phân tán giữa các nguồn, không gọi là khoảng tin cậy. Hình chỉ vẽ mean,
không làm mượt/ép đơn điệu; mỗi loại hỏng một ô, màu method, nét liền none và nét
đứt t1. Mức 0 dùng chung clean, không phải quan sát độc lập giữa năm nhóm.

Đầu ra chính trực tiếp results/t3: config.json, samples.json,
generated_samples.json, per_image.csv, summary.csv, figure.png/pdf,
completion.json. Từ chối chạy nếu có bất kỳ artifact chính nào. README lượt 2
được lưu riêng trước khi cập nhật README tổng; không xóa sanity hoặc bằng chứng cũ.

Hình đầu có chú giải chồng chữ; đã tách bước sửa layout sang `t3_plot.py` để
giữ nguyên SHA runner đã khóa trước test, không sửa config hoặc số liệu đã đo.
Chạy sau formal: `python code/experiments/t3_plot.py --results-dir results/t3`.
Renderer lưu figure_initial.png/pdf, dựng lại figure.png/pdf và ghi SHA trong
figure_render_update.json. Từ chối chạy lại nếu provenance sửa layout đã tồn tại.
completion.json vẫn ràng buộc bản hình gốc; verifier kiểm cả bản gốc và bản mới.

## 9. Kiểm tra nghiệm thu có thể chạy lại

Verifier dùng chung nằm tại `code/scripts/verify_t3.py`, không phụ thuộc vào
việc người nhận có script kiểm tra tạm trong results hay không:

```powershell
code/.venv/Scripts/python.exe code/scripts/verify_t3.py --compatibility-evidence results/t1_t2_repro_audit/t1_t2_v2_handoff.zip
code/.venv/Scripts/python.exe code/scripts/verify_t3.py --compatibility-evidence results/t1_t2_repro_audit/t1_t2_v2_handoff.zip --replay --report results/t3/replay_new.json
```

Mặc định chỉ byte/metadata/CSV; --replay lấy 3 nguồn test rank 0/15/29 theo ID,
đủ 22 vị trí × 10 recipe mỗi nguồn = 660 dòng. So SHA BGR/gray/reference/prediction
và F1/precision/recall với atol 1e-12. Không bỏ hash, không ghi đè report đã có,
không dùng Python -O. Có --data-dir/--t1-dir/--t2-dir/--results-dir cho đường dẫn
đầu vào. Môi trường tái lập đã nghiệm thu là Windows/Python 3.13.7 cùng phiên bản
trong `results/t3/requirements_replay.txt`; không tuyên bố portable đa nền tảng.

Sau nghiệm thu đã cập nhật riêng mục T3 ngày 04/11 trong tài liệu tiến độ hiện
có. Không đánh dấu T4/T5 hoàn thành. Bằng chứng sửa layout, sanity, test và lịch
sử T1–T2 được giữ nguyên; không thay code sinh dữ liệu chung hoặc cleaning.
