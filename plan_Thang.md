# PLAN CHI TIẾT — Thắng — Khối D: chuẩn hóa, gom khoản thuốc, đánh giá, đối chứng VLM

> Đi kèm `plan_v2.md` (plan chung). File này **chỉ** nói về phần của Thắng.
> Mọi con số trong file đều đã đo (ngày 11/11) trên nhãn VAIPE, trên đáp án v1, hoặc trên file mẫu của Khang. Chỗ nào chỉ
> là dự kiến thì ghi rõ "dự kiến".

---

## 0. Tóm tắt

| | |
| :--- | :--- |
| **Mục tiêu** | Biến danh sách vùng chữ (box + nhãn YOLO + text OCR) của một đơn thuốc thành JSON có cấu trúc: danh sách thuốc (tên chuẩn, số lượng, đơn vị, cách dùng), chẩn đoán, ngày. Sau đó đo chất lượng đầu-cuối và so với VLM |
| **Đầu vào** | File JSON của Khang, mỗi ảnh một danh sách box |
| **Phương pháp** | (D1) chọn nhãn, ưu tiên YOLO → (D2) **gom theo dải ngang** → (D3) chuẩn hóa text → (D4) đưa tên thuốc về tên chuẩn bằng từ điển. Đối chứng: (D5) VLM zero-shot |
| **Đánh giá** | F1 khoản thuốc trên 176 ảnh test, kèm khoảng tin cậy 95 % (bootstrap) |
| **Đã kiểm chứng trước** | Nguyên mẫu D1–D3 đạt **F1 = 1,000 (443/443 khoản)** trên nhãn chuẩn của 176 ảnh test, và **3/3 khoản** trên file mẫu của Khang |

---

## 1. Phạm vi và giả định

### 1.1 Trong phạm vi

1. Đọc file của Khang.
2. Chọn nhãn lớp cho từng box (D1).
3. Gom box thành khoản thuốc (D2).
4. Chuẩn hóa text (D3).
5. Đưa tên thuốc về tên chuẩn (D4).
6. Chạy VLM làm đối chứng (D5).
7. Dựng đáp án, viết code chấm, chạy E-D1 và E-D2.
8. Viết phần D của báo cáo, phần Thảo luận, và 2–3 slide.

### 1.2 Ngoài phạm vi

- Chất lượng phát hiện box (YOLO, Trường) và chất lượng nhận dạng chữ (OCR, Khang).
- Chỉnh nghiêng ảnh (Hiếu).
- Chọn hay tinh chỉnh tham số trên valid, và các mô phỏng thêm.

Text sai thì khối D chỉ sửa **những gì sửa được bằng luật** (mục 4.3). Phần còn lại giữ nguyên và chỉ được **đo**, không xử lý.

### 1.3 Giả định (nêu rõ để kiểm soát)

| Mã | Giả định | Nếu sai thì |
| :--- | :--- | :--- |
| G1 | Ảnh đã được khối A chỉnh thẳng (lệch < 3°) | D2 hỏng: lệch 5° thì F1 gom chỉ còn 0,19 (mục 7) |
| G2 | Box ở dạng `[x0, y0, x1, y1]`, pixel trên ảnh đưa vào YOLO | `load_khang` kiểm `x1 > x0` và `y1 > y0`; sai thì báo lỗi |
| G3 | Nhãn lớp là tên (`drugname`…), hoặc `null` với bộ phát hiện cổ điển | `null` thì D1 chuyển sang chỉ dùng text |
| G4 | Trường `gt` trong file của Khang **không được dùng khi dự đoán** | Số liệu bị gian lận. Có test kiểm điều này (mục 8.3) |

---

## 2. Dữ liệu và đáp án

### 2.1 Sự thật dữ liệu liên quan khối D (đo trên 1172 ảnh nhãn VAIPE)

| # | Sự thật | Số đo |
| :-- | :--- | :--- |
| S1 | Mọi tên thuốc bắt đầu bằng `N)` | 2843/2843 |
| S2 | Mọi box số lượng bắt đầu bằng `SL:` | 2843/2843 |
| S3 | Số box số lượng = số tên thuốc trong mọi đơn | 1172/1172 |
| S4 | **2 mẫu đơn**: SL cùng hàng bên phải tên thuốc (mẫu 1), SL ở dòng riêng bên dưới (mẫu 2) | Mẫu 2: 142 ảnh, 379 khoản; riêng test: 100/443 khoản |
| S5 | Chẩn đoán luôn nằm **trên** thuốc đầu tiên, các box thành một khối liền | 1172/1172 |
| S6 | Ngày có 2 dạng: `Ngày dd/mm/yyyy` và `Ngày dd tháng mm năm yyyy` | 1029 / 143 |
| S7 | Đơn vị chỉ gồm `Viên` và `viên` | 2478 / 365 |
| S8 | Từ điển (tên thuốc trong train, đã bỏ `N)`) có 138 tên. Tỉ lệ tên test ngoài từ điển (OOV) là 0,45 % | |
| S9 | Từ điển có cặp tên chỉ khác liều: `DIAMICRON MR TAB 30MG 60'S 30mg` / `... 60MG 30'S 60mg` | |

### 2.2 Định dạng file của Khang (chưa chốt, đọc linh hoạt)

```json
{"image": "VAIPE_P_TRAIN_1.png",
 "boxes": [{"idx": 3, "class": "drugname", "conf": 0.946, "xyxy": [45.6, 386.0, 325.6, 413.0],
            "ocr": "1) GLUCOFAST 850 850mg", "gt": "1) GLUCOFAST 850 850mg", "iou": 1.0}],
 "missed_gt": []}
```

| Trường | Ánh xạ sang `TextBox` | Dùng khi dự đoán? |
| :--- | :--- | :---: |
| `image` | `image_id` (bỏ `.png`) | ✅ |
| `xyxy` | `box` (làm tròn int) | ✅ |
| `class`, `conf` | `label`, `score` | ✅ |
| `ocr` | `text` | ✅ |
| `gt` | — | ❌ (chỉ dùng ở dòng kiểm "text chuẩn", mục 6.2) |
| `iou`, `missed_gt` | — | ❌ (chỉ để xem ca sai) |

### 2.3 Đáp án khoản thuốc

- **Nguồn:** `results/s0/gt_items_test.json`, đáp án v1 do Thắng soát bằng mắt, **176/176 ảnh đã duyệt**. Đáp án này dựng
  bằng một luật **khác** với D2 (cùng hàng, nếu không thì gán vào tên thuốc gần nhất phía trên), nên dùng để kiểm D2 được.
  Không phải trường hợp tự so với chính nó.
- **⚠️ Cách đọc:**
  - Tên thuốc lấy từ phần tử `lines` có `label == "drugname"`. **Không lấy trường `name`**: trường này là box đầu tiên sau
    khi sắp theo y, nên ở 3 ảnh (`VAIPE_P_TRAIN_1161`, `_139`, `_607`) nó thành `"SL: 10 Viên"`. Đọc sai thì E-D1 chỉ còn
    0,993 thay vì 1,000.
  - Số lượng lấy từ `lines` có `label == "quantity"`.
- **`diagnose` và `date`** không có trong đáp án v1. Lấy từ nhãn gốc bằng chính hàm D3 rồi **soát bằng mắt 10 ảnh**, trong
  đó ≥ 3 ảnh mẫu 2 và ≥ 2 ảnh có ngày dạng chữ.
- **Script:** `code/tools/build_items_gt.py` ghi ra `data/1_vaipe/test/items_gt.json`, và in số ảnh vi phạm bất biến S3
  (kỳ vọng 0).
- **Lưu ý:** `data/` và `results/` nằm trong `.gitignore`. Đáp án v1 `results/s0/gt_items_test.json` **không có trong git**,
  phải lấy từ máy Thắng hoặc Drive nhóm. Ai chạy lại thì cần file này.

---

## 3. Định nghĩa bài toán

Mỗi ảnh có một tập box B = {(bᵢ, ℓᵢ, sᵢ, tᵢ)}, gồm box, nhãn YOLO, độ tin, text OCR. Cần tìm:

- tập khoản thuốc **I** = {(tên, số lượng, đơn vị, [cách dùng])}, trong đó mỗi khoản ứng với đúng một box tên thuốc;
- chẩn đoán *d* và ngày *t*.

Bài toán chia làm 3 bài con:

1. **Gán lớp:** mỗi box một lớp trong 6 lớp (D1).
2. **Gán khoản:** mỗi box `quantity` hoặc `usage` thuộc đúng một tên thuốc (D2).
3. **Chuẩn hóa:** text → giá trị chuẩn (D3, D4).

Chỉ có 1 và 2 là "gom". Bài con 3 là ánh xạ chuỗi sang chuỗi.

---

## 4. Phương pháp

Mỗi bước trình bày theo cùng một khuôn: **mô tả → tham số → lý do kèm bằng chứng → phương án bị loại**.

### 4.1 D1 — Chọn nhãn lớp, ưu tiên YOLO

**Mô tả:**
- Có nhãn YOLO: **giữ nhãn YOLO**, trừ khi YOLO gán `other` hoặc độ tin < 0,5. Khi đó hỏi text (bảng bên dưới); text cho ra
  lớp khác `other` thì lấy lớp đó.
- Không có nhãn (bộ phát hiện cổ điển): gán lớp hoàn toàn bằng text.

**Bảng nhận diện bằng text.** So trên chuỗi đã bỏ dấu và viết thường, kiểm theo thứ tự từ trên xuống, gặp mẫu nào khớp trước
thì dừng:

| Lớp | Mẫu (regex) | Chịu được lỗi OCR |
| :--- | :--- | :--- |
| date | `^(\d{1,2}\s*[)\]]\s*)?ngay\s*\d` | `Ngảy`, tiền tố giả `1)` |
| diagnose | `^(\d{1,2}\s*[)\]]\s*)?ch[a-z]{0,2}n\s*d` | `Chẩn`, `Chấn`, `Chần`, `Chản` |
| quantity | `^[s5][l1i\|]\s*[:.;]?\s*\d` | `SL.`, `SL;`, `5L:`, `Sl` |
| drugname | `^\d{1,2}\s*[)\]}l.]\s*\S`, dài > 6 ký tự, không phải `\d+/` | `1]`, `1.`, `l)`; loại số trang `1/1` |
| usage | `^(sang\|trua\|chieu\|toi\|uong)` và có chữ số | `Sảng`, `CHiều` |
| other | còn lại | |

**Tham số:** ngưỡng độ tin YOLO = 0,5, **cố định**.

**Bằng chứng** (file mẫu của Khang, 1 ảnh, 3 thuốc). OCR **tự thêm tiền tố** `N)` vào box không phải thuốc:
`"2) Chản dưỡng..."` (chẩn đoán), `"1) Ngảy 18/00 / 201g"` (ngày). YOLO gán đúng cả hai box (độ tin 0,83 và 0,94).

| Phương án | Số khoản ra | Precision |
| :--- | :---: | :---: |
| Chỉ tin YOLO | 3 | 1,00 |
| Text khớp `N)` hay `SL:` thì luôn ghi đè YOLO | 5 (chẩn đoán và ngày bị coi là thuốc) | 0,60 |
| **Ưu tiên YOLO, text chỉ dùng khi YOLO không chắc (chọn)** | **3** | **1,00** |

**Lý do chọn:** giữ ưu điểm của "chỉ tin YOLO" (miễn nhiễm với OCR bịa tiền tố), mà vẫn cứu được ca YOLO gán nhầm SL thành `other`.

### 4.2 D2 — Gom khoản theo dải ngang

**Mô tả:**
1. Sắp box tên thuốc theo **y0 (đỉnh box)**.
2. Lập mốc `tops = [y0 của từng tên thuốc] + [y0 của box ngày]`. Chỉ xét box ngày nằm **dưới** đỉnh tên thuốc cuối (phòng
   ca YOLO gán nhầm một box phía trên thành `date`). Không có box ngày hợp lệ thì mốc cuối là vô cùng.
3. Dải i = `[tops[i], tops[i+1])`. Box `quantity` hoặc `usage` có **tâm y** = (y0 + y1) / 2 nằm trong dải i thì thuộc thuốc i.
4. Trong một dải có nhiều SL thì lấy SL có tâm y nhỏ nhất. Cách dùng sắp theo thứ tự đọc: gom hàng trước (hai box lệch tâm
   y < 0,5 × chiều cao là cùng hàng), rồi trái sang phải.

**Bằng chứng 1 — chọn phương án gom** (đo trên nhãn chuẩn, 2843 khoản):

| Phương án | Đúng |
| :--- | :---: |
| SL cùng hàng, bên phải tên thuốc (plan 00:40) | 0,87 (riêng test: 0,77) |
| Ghép mỗi tên với ô SL gần nhất | 0,895 |
| **Dải ngang (chọn)** | **1,000** |

**Bằng chứng 2 — vì sao mốc là y0** (đo trên nhãn chuẩn, 2843 khoản):

| Mốc chia dải | Đúng |
| :--- | :---: |
| **y0, đỉnh tên thuốc (chọn)** | **1,000** |
| Tâm y của tên thuốc | 0,177 |
| y1, đáy tên thuốc | 0,133 |

Ở mẫu 1, tâm của box SL thường **cao hơn** tâm tên thuốc (2341/2464 ca, trung vị 1,5 px). Vì vậy mốc tâm hay mốc đáy đẩy SL
sang dải của thuốc phía trên. Ngược lại, tâm SL luôn nằm **dưới** y0 của tên thuốc ít nhất 9 px (trung vị 12 px), nên
y0 có khoảng an toàn trước sai số box của YOLO.

**Bằng chứng 3 — độ bền khi ảnh nghiêng** (xoay tọa độ trên nhãn chuẩn):

| Lệch | 0° | 1° | 2° | 3° | 5° |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dải ngang | 1,000 | 1,000 | 1,000 | 1,000 | 0,190 |

Vì vậy cần giả định G1. Đã thử sửa góc ngay trên tọa độ: sai số khoảng 1° và làm hỏng cả ảnh thẳng (1,000 → 0,905), nên
**không sửa góc ở khối D**.

**Xử lý box lộn xộn** (chạy trước D2):

| Ca | Cách xử lý | Phương án bị loại |
| :--- | :--- | :--- |
| Hai box chồng nhau (IoU > 0,7) | Giữ box có `conf` cao hơn | Giữ cả hai: sinh khoản thừa |
| Tên thuốc và `SL:` gộp làm một box | Tách phần `SL: N ...` ra thành một box SL | Bỏ qua: mất số lượng |
| Tên thuốc xuống dòng (dòng 2 không có `N)`) | Nối vào tên thuốc ngay phía trên nếu cùng lề trái | Mở khoản mới: sinh khoản thừa |
| Dải không có SL | `quantity = null` | Tìm SL dự phòng hoặc cân bằng SL giữa các dải: mô phỏng **không tăng** F1 |

### 4.3 D3 — Chuẩn hóa text

| Trường | Luật | Ví dụ thật (file của Khang) |
| :--- | :--- | :--- |
| Tên thuốc | Unicode NFC; bỏ tiền tố `^\d{1,2}\s*[)\]}l.]\s*`; gộp khoảng trắng. **Không đổi hoa thường** (đáp án ghi `GLUCOFAST`) | `1) GLUCOFAST 850 850mg` → `GLUCOFAST 850 850mg` |
| Số lượng | Lấy cụm số sau `SL`; trong cụm số đổi `O/o→0`, `I/l→1` | `SL. 60 ViÊn` → 60 |
| Đơn vị | Bỏ dấu, viết thường; là `vien` thì trả `Viên` | `ViÊn`, `Viền` → `Viên` |
| Cách dùng | Sửa từ chung theo dạng bỏ dấu: `sang→Sáng`, `chieu→Chiều`, `toi→Tối`, `trua→Trưa`, `vien→Viên`, `uong→Uống` | `Sảng 1 Viền` → `Sáng 1 Viên` |
| Chẩn đoán | Nối mọi box chẩn đoán nằm trên thuốc đầu tiên theo thứ tự đọc; bỏ tiền tố giả `N)` và `Chẩn đoán:` | `"2) Chản dưỡng..." + "phảt)"` → một chuỗi |
| Ngày | Bỏ tiền tố giả; **chỉ trong cụm số** đổi `O→0`, `l/I→1`, `g→9`; đọc 2 dạng (S6); kiểm ngày ≤ 31, tháng ≤ 12, năm trong 2000–2030; sai thì `null` | `1) Ngảy 18/00 / 201g` → tháng 00 không hợp lệ → `null` |

**Nguyên tắc:** chỉ sửa những lỗi mà luật chắc chắn sửa đúng. Sửa đổi chữ trên **cả chuỗi** bị cấm: làm thế thì `ngay` thành
`n9ay`.

### 4.4 D4 — Đưa tên thuốc về tên chuẩn (từ điển)

**Mô tả:** từ điển là 138 tên thuốc của train (S8). Lấy 2 ứng viên giống nhất theo `rapidfuzz.fuzz.ratio`, gọi điểm là s₁ ≥ s₂:
- s₁ ≥ 90: thay bằng tên chuẩn.
- 80 ≤ s₁ < 90 **và** s₁ − s₂ ≥ 5: thay.
- Còn lại: **giữ tên sau D3**.
- **Khóa liều:** nếu hai ứng viên sát nhau (s₁ − s₂ < 5) mà khác dãy số (ví dụ 30MG và 60MG, S9), thì chỉ thay khi dãy số
  trong tên OCR trùng với ứng viên.

**Tham số:** 90 / 80 / 5, **cố định trước khi chạy test, không chỉnh**.

**Lý do và bằng chứng** (mô phỏng lỗi kiểu OCR trên tên thuốc của valid, ô ghi "đúng / kéo nhầm / bỏ qua"):

| Cách so | CER tên 0,03 | 0,06 | 0,10 |
| :--- | :---: | :---: | :---: |
| `ratio`, một ngưỡng 90 | 1,000 / 0 / 0 | 1,000 / 0 / 0 | 0,631 / 0 / 0,369 |
| `ratio`, một ngưỡng 85 | 1,000 / 0 / 0 | 1,000 / 0 / 0 | 1,000 / 0 / 0 |
| `token_sort_ratio`, 85 | 0,828 / 0,007 / 0,165 | 0,741 / 0,007 / 0,252 | 0,663 / 0,007 / 0,329 |
| `ratio` + bắt dãy số trùng cứng | 0,693 / 0 / 0,307 | 0,628 / 0 / 0,372 | 0,466 / 0 / 0,534 |

Đọc bảng:
- `ratio` tốt hơn `token_sort`.
- Bắt dãy số trùng **cứng** thì hại, vì OCR hay đọc sai chính chữ số. Do đó chỉ khóa liều ở ca hai ứng viên sát nhau.
- Hai ngưỡng 90 / 80 kết hợp khoảng cách 5 điểm cho phép thay ở vùng 80–90 khi không có đối thủ sát, nên an toàn hơn một
  ngưỡng 85.

Lỗi mô phỏng không giống hệt lỗi OCR thật. Vì vậy E-D2 luôn báo cả cột **"không từ điển"** để thấy tác động thật.

### 4.5 D5 — Đối chứng VLM zero-shot

| Hạng mục | Chốt |
| :--- | :--- |
| Model | Qwen2.5-VL qua FPT AI. Không có thì dùng VLM khác của FPT AI, rồi đến OpenRouter. Chốt trước khi chạy test, ghi tên chính xác vào `config.json` |
| Đầu vào | Ảnh gốc, cạnh dài ≤ 1600 px, JPEG q 90, base64 |
| Prompt | Cố định (bên dưới), thử trên 5–10 ảnh **valid**, chốt rồi không sửa |
| Tham số | `temperature=0`, `max_tokens=1500`, 4–8 luồng, thử lại 3 lần khi lỗi mạng (chờ 2, 4, 8 s) |
| JSON hỏng | Cắt khối `{...}` ngoài cùng rồi parse lại một lần; vẫn hỏng thì tính là đơn rỗng, đếm `n_json_fail` |
| Công bằng | Đầu ra VLM đi qua **cùng D3 và D4** như pipeline, rồi chấm cùng cách |
| Lưu | Câu trả lời thô vào `results/E-D2/vlm_raw/<id>.json`; đã có file thì không gọi lại |
| Riêng tư | VAIPE đã che thông tin cá nhân từ nguồn nên gửi API được |

```
Bạn là hệ thống trích xuất thông tin từ ảnh đơn thuốc Việt Nam.
Trả về DUY NHẤT một đối tượng JSON, không giải thích, theo đúng schema:
{"items":[{"name":str,"quantity":int|null,"unit":str|null,"usage":[str]}],"diagnose":str|null,"date":"dd/mm/yyyy"|null}
Quy tắc: "name" chép nguyên văn tên thuốc, bỏ số thứ tự đầu dòng như "1)". "quantity" là số sau "SL:".
"usage" là các dòng cách dùng của thuốc đó. "diagnose" chép nguyên văn sau "Chẩn đoán:".
Không bịa; trường nào không thấy thì để null.
```

---

## 5. Giao thức đánh giá

### 5.1 Độ đo

| Độ đo | Định nghĩa |
| :--- | :--- |
| **F1 khoản (chính)** | Một khoản dự đoán **khớp** một khoản đáp án khi CER(tên) ≤ 0,1 **và** số lượng bằng nhau. Tên so sau NFC, gộp khoảng trắng, bỏ `N)`, **không phân biệt hoa thường** (casefold). Ghép tham lam theo CER tăng dần, mỗi khoản đáp án ghép tối đa 1 lần. Cộng TP, FP, FN trên cả tập rồi tính P, R, F1 (micro) |
| F1 tên | Như trên nhưng bỏ điều kiện số lượng |
| % đơn đúng hoàn toàn | Tỉ lệ ảnh có FP = FN = 0 |
| % ngày đúng | `date` dự đoán trùng đáp án sau chuẩn hóa |
| Khoảng tin cậy 95 % | Bootstrap 1000 lần, lấy mẫu lại **theo ảnh** (các khoản trong cùng ảnh không độc lập), seed 42, lấy phân vị 2,5 và 97,5 |

Vì sao chọn như vậy:
- **Ngưỡng CER 0,1** để không phạt lỗi 1–2 ký tự không đổi nghĩa.
- **Casefold** để không phạt VLM vì đổi hoa thường.
- **Ghép tham lam** là đủ, vì mỗi đơn có ≤ 5 khoản, nên ghép tham lam và ghép tối ưu (Hungarian) cho cùng kết quả.

### 5.2 Kỷ luật đánh giá

1. Mọi tham số (0,5; 0,7; 90 / 80 / 5; prompt) **chốt trước** khi chạy test và ghi vào `config.json`.
2. Test chạy **một lần** cho mỗi dòng của bảng. Không sửa luật dựa trên kết quả test. Nếu thật sự phải sửa (lỗi code), thì
   ghi lại lý do và chạy lại **mọi dòng**.
3. Code dự đoán **không đọc** trường `gt` (G4).
4. `config.json` ghi commit hash, tên model VLM, đường dẫn file đầu vào.

---

## 6. Thí nghiệm và giả thuyết

### 6.1 E-D1 — Kiểm tra luật gom khi đầu vào hoàn hảo

- **Giả thuyết H1:** với box và text chuẩn, D1 + D2 + D3 tái tạo đúng đáp án.
- **Cách làm:** đầu vào là nhãn gốc `data/1_vaipe/test/labels/*.json` (đọc bằng `vaipe.dataset.read_label_json`, đổi sang
  `TextBox` với `score = 1`). Chạy 2 chế độ: có nhãn, và chỉ text (`label = null`). **Tắt từ điển** (D4) để kiểm riêng
  luật gom. So với đáp án mục 2.3.
- **Kết quả nguyên mẫu:** F1 = **1,000** (443/443) ở cả 2 chế độ.
- **Ra:** `results/E-D1/{config.json, table.csv, bad_cases.csv}`. Cột `table.csv`: `mode, tp, fp, fn, f1, n_bad_images`.

### 6.2 E-D2 — Bảng chính đầu-cuối (176 ảnh test)

| Dòng | Đầu vào | Bắt buộc |
| :--- | :--- | :---: |
| (i) Cổ điển | file box cổ điển + Tesseract (`class = null`) | nếu nhóm có |
| (ii) Lai | YOLO + Tesseract | nếu nhóm có |
| **(iii) DL (chính)** | **YOLO + VietOCR fine-tune** | ✅ |
| (iv) DL pretrained | YOLO + VietOCR pretrained | nếu nhóm có |
| (v) VLM | ảnh → VLM → D3 + D4 | ✅ |
| (kiểm) Text chuẩn | file của dòng (iii) nhưng lấy `gt` thay cho `ocr` | ✅ |

**Cột:** `pipeline, f1_item, ci_low, ci_high, f1_item_nolex, f1_name, exact_rate, date_acc, n_json_fail`.

**Giả thuyết và cách đọc:**

| Giả thuyết | So sánh | Ý nghĩa |
| :--- | :--- | :--- |
| H2: lỗi đầu-cuối chủ yếu do text OCR | (kiểm) so với (iii) | Hiệu số = phần điểm mất do OCR. Phần còn lại (1 − (kiểm)) là do box của YOLO |
| H3: từ điển tăng F1 | `f1_item` so với `f1_item_nolex` ở (iii) | Khoảng tin cậy không chồng nhau thì khác biệt có ý nghĩa |
| H4: pipeline DL và VLM khác nhau | (iii) so với (v), theo khoảng tin cậy | Kết luận theo đúng số, không ép. Thêm phân tích lỗi bên dưới |

**Phân tích lỗi:** lấy 10 ca sai của (iii) và 10 ca (iii) và (v) ra khác nhau. Ghi vào `error_analysis.csv` với cột
`image_id, dòng, loại lỗi, mô tả`. Loại lỗi gồm: thiếu box, sai lớp, text tên sai, text số sai, luật, VLM bịa, VLM bỏ sót.
Chụp 2 hình overlay: 1 ca đúng, 1 ca sai.

**Ra:** `results/E-D2/{config.json, table.csv, preds/<dòng>.jsonl, prompt.txt, vlm_raw/, error_analysis.csv}`.

---

## 7. Bằng chứng đã có (tóm tắt, để dẫn trong báo cáo)

| Nội dung | Kết quả | Dùng ở |
| :--- | :--- | :--- |
| Nguyên mẫu E-D1, 176 ảnh test | F1 1,000 (443/443), cả có nhãn lẫn chỉ text | 6.1 |
| Nguyên mẫu trên file mẫu của Khang | 3/3 khoản; ưu tiên YOLO 1,00 so với text ghi đè 0,60 precision | 4.1 |
| Phương án gom | Dải ngang 1,000 / cùng hàng 0,87 / gần nhất 0,895 | 4.2 |
| Mốc chia dải | y0 1,000 / tâm 0,177 / đáy 0,133 | 4.2 |
| Độ bền khi nghiêng | 1,000 đến 3°; 0,190 ở 5° | 4.2, G1 |
| Đáp án v1 đọc sai trường `name` | 3 ảnh, E-D1 tụt còn 0,993 | 2.3 |
| Từ điển (mô phỏng) | Bảng mục 4.4 | 4.4 |

---

## 8. Hiện thực

### 8.1 Dùng lại code đã có (commit `ec63481` của Hiếu)

| Cần | Lấy ở đâu |
| :--- | :--- |
| Kiểu dữ liệu | `common.contracts`: `TextBox(box, label, score, text, text_conf)`, `Item(name, name_raw, quantity, unit, usage, lexicon_match, box)`, `PrescriptionResult`, `BOX_LABELS`. **Không sửa file này.** `Item.box` = khung bao cả khoản |
| IoU, P/R/F1 | `common.boxes`: `iou`, `precision_recall_f1` |
| CER | `legacy.ocr.metrics.cer` |
| Đọc nhãn VAIPE | `vaipe.dataset.read_label_json` |
| Đường dẫn | `common.paths`: `VAIPE_DIR`, `RESULTS_DIR`, `vaipe_labels` |
| Thư viện thêm | `rapidfuzz`, `openai` (thêm vào requirements) |
| Chạy test | `cd code && pip install -e . && pytest` |

### 8.2 File Thắng viết

| File | Hàm |
| :--- | :--- |
| `src/assemble/load.py` | `load_khang(path) -> dict[str, list[TextBox]]` (đọc mảng JSON hoặc JSONL; bỏ qua `gt`) |
| `src/assemble/rules.py` | `strip_accents`, `classify_text`, `clean_name`, `parse_quantity`, `clean_usage`, `parse_date` |
| `src/assemble/group.py` | `choose_labels`, `dedupe`, `split_glued_quantity`, `merge_wrapped_names`, `center_y`, `split_bands`, `reading_order`, `merge_diagnose`, `assemble` |
| `src/assemble/lexicon.py` | `Lexicon.from_split("train")`, `Lexicon.match(name)` |
| `src/assemble/vlm.py` | `vlm_extract(image_path) -> dict`, có cache `vlm_raw/` |
| `src/metrics/items.py` | `items_prf(preds, gts)`, `bootstrap_ci(per_image)` |
| `tools/build_items_gt.py` | dựng đáp án mục 2.3 |
| `experiments/e_d1.py`, `experiments/e_d2.py` | chạy thí nghiệm, ghi `results/` |
| `tests/test_assemble.py` | mục 8.3 |

Code lõi của D2 (đã chạy đúng trên file mẫu của Khang):

```python
def center_y(b):  return (b.box[1] + b.box[3]) / 2

def split_bands(boxes):
    drugs = sorted([b for b in boxes if b.label == "drugname"], key=lambda b: b.box[1])   # theo y0
    if not drugs:
        return []
    date_tops = [b.box[1] for b in boxes if b.label == "date"]
    tops = [d.box[1] for d in drugs] + [min(date_tops) if date_tops else float("inf")]
    bands = []
    for i, drug in enumerate(drugs):
        inside = [b for b in boxes if tops[i] <= center_y(b) < tops[i + 1]]
        qs = sorted([b for b in inside if b.label == "quantity"], key=center_y)
        bands.append({"drug": drug, "quantity": qs[0] if qs else None,
                      "usage": reading_order([b for b in inside if b.label == "usage"])})
    return bands

def reading_order(boxes):
    rows = []
    for b in sorted(boxes, key=center_y):
        if rows and abs(center_y(b) - center_y(rows[-1][0])) < 0.5 * (b.box[3] - b.box[1]):
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: b.box[0])]
```

### 8.3 Kiểm thử (`tests/test_assemble.py`)

| Test | Kiểm |
| :--- | :--- |
| `classify_text` | ≥ 20 chuỗi: `SL. 60 ViÊn`, `5L: 20`, `1] TÊN THUỐC`, `Chản đoán:`, `1) Ngảy 18/09/2019` (phải là `date`), `Sảng 1 Viền`, `1/1` (phải là `other`) |
| `choose_labels` | Box chẩn đoán có text `"2) ..."`, YOLO độ tin 0,83 → vẫn là `diagnose` |
| `parse_quantity`, `parse_date` | `SL; 30` → 30; `2O` → 20; cả 2 dạng ngày; `18/00/2019` → `null`; `ngay` không bị đổi thành `n9ay` |
| `split_bands` | Ảnh mẫu 1 và ảnh mẫu 2: mỗi thuốc đúng 1 SL; "Sáng" đứng trước "Chiều" |
| `assemble` trên file mẫu của Khang | Ra đúng 3 khoản GLUCOFAST 60, DIAMICRON 30, ENALAPRIL 30 |
| Không đọc `gt` | Đổi mọi `gt` thành chuỗi rác → kết quả không đổi |
| `items_prf` | Khớp hoàn hảo = 1; thiếu 1 khoản; thừa 1 khoản; sai số lượng |

---

## 9. Lịch

| Bước | Việc | Thời lượng dự kiến | Ra |
| :---: | :--- | :---: | :--- |
| 1 | `rules.py`, `group.py`, `load.py` + test | 1 h | test xanh |
| 2 | `build_items_gt.py` + soát 10 ảnh chẩn đoán và ngày | 20' | `items_gt.json` |
| 3 | **E-D1** (kỳ vọng 1,000) | 10' | `results/E-D1` |
| 4 | `lexicon.py`, `metrics/items.py` | 30' | |
| 5 | `vlm.py`; thử 5–10 ảnh valid; chốt prompt và model; **chạy VLM 176 ảnh test ở nền** | 40' | `vlm_raw/` |
| 6 | **Khi có file của Khang cho test:** E-D2 các dòng + dòng kiểm + bootstrap | 30' | `results/E-D2` |
| 7 | Phân tích 20 ca lỗi + 2 hình | 30' | `error_analysis.csv` |
| 8 | Báo cáo phần D + Thảo luận + 2–3 slide | 1 h 30' | |

File của Khang trễ thì làm bước 1–5 và phần viết phương pháp trước. Khi có file, bước 6 chỉ mất khoảng 15–30'.

---

## 10. Các nguy cơ làm sai kết quả (threats to validity)

| Nguy cơ | Ảnh hưởng | Cách kiểm soát |
| :--- | :--- | :--- |
| Đáp án sinh bằng chính luật cần đo | E-D1 luôn ra 1,0, không phát hiện lỗi | Đáp án từ v1, dựng bằng luật khác và **soát tay cả 176 ảnh** |
| Đọc sai trường của đáp án v1 | E-D1 thấp giả tạo | Lấy tên từ `lines` có nhãn `drugname` (mục 2.3) |
| Dùng `gt` khi dự đoán | Số liệu gian lận | G4 + test "không đọc `gt`" |
| Chỉnh tham số theo kết quả test | Số liệu lạc quan | Tham số chốt trước; test chạy 1 lần (mục 5.2) |
| Từ điển ưu ái pipeline so với VLM | So sánh lệch | VLM qua cùng D3 + D4; báo thêm cột "không từ điển" |
| Từ điển chỉ biết thuốc trong train | Kết quả đẹp hơn thực tế | Báo OOV 0,45 %; nêu trong phần giới hạn |
| Ảnh lệch > 3° | D2 sụp | Giả định G1; nêu là điều kiện cần của khối A |
| VAIPE chỉ có 2 mẫu đơn in máy | Khó tổng quát hóa | Nêu trong phần giới hạn; dải ngang không phụ thuộc mẫu, nhưng chưa kiểm trên đơn viết tay hay ảnh chụp |
| Chỉ có 1 file mẫu của Khang | Bằng chứng mục 4.1 mỏng | E-D2 trên 176 ảnh sẽ xác nhận; báo `f1_item` của (iii) |
| VLM không tất định, model thay đổi | Khó tái lập | `temperature=0`, lưu câu trả lời thô, ghi tên model và ngày chạy |

---

## 11. Báo cáo và slide

**Phương pháp D (≈ 0,75 trang):**
- Định nghĩa bài toán (mục 3).
- D1–D4, kèm hình dải ngang trên 1 ảnh mẫu 2.
- Bảng tham số: 0,5; 0,7; 90 / 80 / 5; casefold; CER 0,1. Mỗi tham số một câu lý do dẫn mục 4.
- VLM: model và prompt (prompt để ở phụ lục).

**Kết quả D (≈ 0,5 trang):**
- E-D1: 1 câu.
- **Bảng E-D2** có khoảng tin cậy.
- Bảng nhỏ "mốc chia dải" và "độ bền khi nghiêng" (mục 7).

**Thảo luận (≈ 0,5 trang):**
1. Luật gom đúng tuyệt đối trên đầu vào chuẩn; lỗi đầu-cuối đến từ box và text (H2).
2. Dải ngang không phụ thuộc mẫu đơn nhưng cần ảnh thẳng. Đây là chỗ khối A (xử lý ảnh cổ điển) là điều kiện cần.
3. Ưu tiên YOLO hơn text vì OCR có thể bịa tiền tố.
4. Tác dụng của từ điển (H3) và giới hạn OOV.
5. VLM mạnh và yếu ở đâu (H4, phân tích lỗi); rủi ro riêng tư khi gửi ảnh y tế ra ngoài.
6. Giới hạn: VAIPE in máy, 2 mẫu đơn, chưa thử ảnh chụp thật.

**Slide (2–3 trang):**
1. Bài toán gom: 2 mẫu đơn, dải ngang, mốc y0.
2. Chuẩn hóa, ưu tiên YOLO, ví dụ thật từ file của Khang.
3. Bảng E-D2, 1 ca VLM đúng và 1 ca VLM sai.

---

## 12. Checklist xong

- [ ] Test `tests/test_assemble.py` xanh, kể cả test "không đọc `gt`".
- [ ] `data/1_vaipe/test/items_gt.json` (đọc tên từ `lines`), 10 ảnh chẩn đoán và ngày đã soát.
- [ ] E-D1 = 1,000 hoặc có giải thích từng ca sai.
- [ ] E-D2 có ít nhất dòng (iii), (v), (kiểm), mỗi dòng có khoảng tin cậy 95 % và cột "không từ điển".
- [ ] `config.json` của mỗi thí nghiệm có commit hash, tham số, tên model VLM.
- [ ] `error_analysis.csv` (20 ca) + 2 hình.
- [ ] Báo cáo phần D + Thảo luận + phụ lục prompt; 2–3 slide; 1–2 câu khai báo AI (AI viết code và đo kiểm; Thắng chọn phương án, soát đáp án, chạy và kiểm số liệu).
