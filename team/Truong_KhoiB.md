# KẾ HOẠCH CHI TIẾT — TRƯỜNG — KHỐI B: PHÁT HIỆN + PHÂN LOẠI VÙNG CHỮ

> Bám theo [plan.md](../plan.md) mục 5.2 và 7.2. Hạn: nộp 11/11/2026 trước 23:59, nên **weights phải xong sớm nhất có thể** (Hiếu và Thắng chờ).
> Đã chốt: **YOLOv8s, `imgsz=1024`**, một mô hình duy nhất. **Bỏ train 640** (11/11): lý do chọn 1024 dựa trên số đo chiều cao chữ ở mục 0.
> Đánh dấu: ⬜ chưa làm · 🔄 đang làm · ✅ xong. Xong mỗi bước thì cập nhật [TienDo.md](TienDo.md).

---

## 0. Bối cảnh và số liệu đã đo

| Mục | Giá trị |
| :--- | :--- |
| Dữ liệu | `data/1_vaipe/` (359 MB, đã có trên máy): train 820, valid 176, test 176 ảnh; nhãn JSON `[{id, text, label, box:[x0,y0,x1,y1]}]` |
| Kích thước ảnh | 760×1070 (56 %), 770×1420 (21 %), 1150×1600 (23 %) |
| Chiều cao chữ | gốc: trung vị 22 px, p5 16 px → ở 640 còn 12 / **8 px** (= stride 8, dễ mất); ở 1024 còn 19 / 13 px |
| Số khung theo lớp (cả 3 tập) | other 28.679 · usage 3.994 · drugname 2.843 · quantity 2.843 · diagnose 1.649 · date 1.172 |
| Thứ tự lớp YOLO (cố định) | `0 drugname, 1 quantity, 2 usage, 3 diagnose, 4 date, 5 other` |

**Lưu ý:**
- **Mất cân bằng lớp:** `other` chiếm khoảng 70 % số khung, nên phải báo F1 **từng lớp**, không chỉ mAP chung.
- **`drugname` và `quantity` có số khung bằng nhau** (mỗi khoản thuốc có 1 tên và 1 số lượng). Kết quả của E-B2 trên hai lớp này dùng để kiểm tra chéo.
- Trong `split_manifest.csv` có ảnh `excluded` (trùng byte với test). `to_yolo.py` chỉ đọc theo thư mục `train/ valid/ test/`, nên ảnh này không lọt vào.

---

## 1. Lộ trình theo giờ (T = lúc bắt đầu)

| Mốc | Việc | Loại | Thời gian |
| :---: | :--- | :---: | :---: |
| T+0:00 | **B1** Nhắn Hiếu: chốt 1024, hỏi ai dọn repo sang `legacy/` | HV | 2' |
| T+0:05 | **B2** Viết `code/tools/to_yolo.py`, sinh `data/1_vaipe_yolo/` + zip | AI | 15' |
| T+0:20 | **B3** Đẩy zip lên Kaggle Datasets (Private) | HV | 20' |
| T+0:40 | **B4** Notebook `code/notebooks/train_yolo.ipynb`, chạy **YOLOv8s 1024** | AI+HV | 15' + **nền ~1–2,5h** |
| T+1:00 | **B5** `code/src/detect/yolo.py` (load + infer → `TextBox`) | AI | 25' |
| T+1:25 | **B6** `code/src/detect/classical.py` (Sauvola → giãn → CC) + chọn SE trên valid | AI+HV | 40' |
| T+2:05 | **B7** `code/src/metrics/detection.py` (nếu Hiếu chưa viết) + script `experiments/e_b1.py`, `e_b2.py` chạy thử trên **valid** | AI | 30' |
| T+2:35 | **B8** Khung báo cáo phần B + 3 slide (chưa có số) | HV+AI | 30' |
| khi 1024 xong | **B9** Tải `best.pt` → `code/weights/yolo_vaipe.pt`, đưa lên Drive, báo nhóm | HV | 10' |
| ngay sau B9 | **B10** Chọn `conf` trên valid; chạy **E-B1, E-B2 trên test một lần** | HV | 35' |
| khi có số | **B11** Điền số vào báo cáo + slide; gửi Hiếu | HV | 20' |

Tổng thời gian làm trực tiếp khoảng **4 giờ**. Train chạy nền song song với B5–B8.

---

## 2. Chi tiết từng bước

### B2. Chuyển nhãn sang YOLO — `code/tools/to_yolo.py`

- **Vào:** `data/1_vaipe/{train,valid,test}/{images,labels}`.
- **Ra:** cấu trúc chuẩn của ultralytics:
  ```
  data/1_vaipe_yolo/
  ├── images/{train,valid,test}/*.png      # copy (không symlink — Kaggle cần file thật)
  ├── labels/{train,valid,test}/*.txt      # mỗi dòng: cls cx cy w h (chuẩn hóa 0–1)
  └── vaipe.yaml                           # path, train, val, test, names (6 lớp)
  ```
- **Quy tắc:**
  - Kẹp box vào khung ảnh.
  - Bỏ box có `w ≤ 1` hoặc `h ≤ 1` px, ghi số box bị bỏ ra màn hình.
  - Nhãn lạ (không thuộc 6 lớp) → dừng và báo lỗi, không đoán.
  - Ảnh có 0 box vẫn tạo file `.txt` rỗng.
- **Zip để đẩy Kaggle:** `vaipe_yolo.zip` chỉ chứa `train` + `valid` + `vaipe.yaml`. **Không đẩy test**, để giữ kỷ luật "test chỉ chạy một lần, ở local".
- **Kiểm tra:**
  - Số file `.txt` bằng số ảnh (820 / 176 / 176).
  - Tổng số dòng nhãn bằng tổng số box trong JSON, trừ số box bị bỏ.
  - Vẽ lại box lên 3 ảnh ngẫu nhiên (seed 42), lưu `results/e_b0/check_*.png`, nhìn bằng mắt cho khớp chữ.

### B3. Kaggle Dataset

- Kaggle → Datasets → New Dataset → upload `vaipe_yolo.zip` → Private, tên `vaipe-yolo`.
- Ghi lại đường dẫn: `/kaggle/input/vaipe-yolo/`. Trong notebook phải viết lại `vaipe.yaml` với `path:` trỏ vào đó.
- Nếu upload chậm: nén ảnh PNG không giảm được nhiều, nên chịu khó chờ hoặc tách làm 2 lần upload (train / valid).

### B4. Notebook train — `code/notebooks/train_yolo.ipynb`

Các ô trong notebook:
1. `!pip install -q ultralytics` rồi in phiên bản, ghi vào `config.json`.
2. Ghi `vaipe.yaml` mới vào `/kaggle/working/` với `path: /kaggle/input/vaipe-yolo`.
   - Ultralytics cần ghi file `.cache` cạnh thư mục nhãn. Nếu nó báo lỗi thư mục chỉ đọc thì **copy dataset sang `/kaggle/working/`** trước khi train (mất khoảng 1 phút).
3. Lệnh train chính:
   ```
   yolo detect train model=yolov8s.pt data=/kaggle/working/vaipe.yaml imgsz=1024 epochs=50 batch=16 patience=15 seed=42 fliplr=0 mosaic=0.5 device=0,1 project=/kaggle/working/runs name=v8s_1024
   ```
   - `fliplr=0`: lật ngang làm chữ bị ngược.
   - `mosaic=0.5`: giảm mosaic để giữ bố cục trang.
   - Hết bộ nhớ GPU (OOM) → giảm `batch=8`, **không giảm `imgsz`**.
4. `model.val(data=..., split='val')` để in mAP50 và lưu ma trận nhầm lẫn của valid.
5. Nén `runs/v8s_1024/` (gồm `weights/best.pt`, `results.csv`, các hình) thành file để tải về.

- **Chạy:** dùng **Save Version → Save & Run All (Commit)** để notebook chạy nền, tắt máy vẫn được. Settings: Accelerator **GPU T4 ×2**, Internet **On**.
- **Theo dõi:** sau 2–3 epoch, xem thời gian mỗi epoch rồi nhân với 50 để biết giờ xong. Báo giờ dự kiến cho Hiếu.
- **Dự phòng nếu chậm:**
  - Nếu 1 epoch mất quá 3' (tức là 50 epoch mất quá 2,5h): dừng, chạy lại `epochs=30`.
  - Nếu vẫn chậm: chuyển sang `yolov8n` ở cùng `imgsz=1024`.

### B5. `code/src/detect/yolo.py`

```python
class YoloDetector:
    def __init__(self, weights="code/weights/yolo_vaipe.pt", imgsz=1024, conf=0.25, iou=0.6, device=None): ...
    def detect(self, img_bgr) -> list[TextBox]:   # box xyxy int, label = tên lớp, score
```

- Tên lớp đọc từ `model.names`, không ghi cứng trong code.
- `device=None`: tự chọn `cuda` nếu có, không thì `cpu`.
- `TextBox` lấy từ `common/contracts.py`. Nếu Hiếu chưa viết lại file đó thì tạm định nghĩa trong `detect/__init__.py` đúng như plan mục 4.2, rồi bỏ khi Hiếu đã đẩy bản chính.

### B6. `code/src/detect/classical.py` — đối chứng cổ điển (không phân lớp)

1. Ảnh xám → Sauvola (cửa sổ 31, k 0,2) → ảnh nhị phân, chữ = 1. Môn học: chương 5.
2. `h` = trung vị chiều cao các thành phần liên thông, chỉ tính các thành phần có diện tích > 10 px để bỏ nhiễu.
3. Giãn bằng SE chữ nhật cao `a·h` × rộng `b·h`, để nối ký tự trong cùng một cụm nhưng không nối sang cột bên cạnh. Môn học: chương 7.
4. Lấy thành phần liên thông → box. Lọc box quá nhỏ (cao < 0,5h), quá lớn (> 50 % trang) và đường kẻ dài (tỉ lệ rộng/cao > 40 và cao < 0,3h).
   - Dùng lại `table_extraction/geometry.py::words_from_binary`. Sau khi Hiếu dọn repo, file này nằm ở `legacy/`.
- **Trả về:** `list[TextBox]` với `label=None`, `score=1.0`.
- **Chọn tham số trên valid:** lưới `a ∈ {0.3, 0.5, 0.8}` × `b ∈ {0.8, 1.2, 1.5, 2.0, 3.0}`, đo F1 @IoU 0,5 không phân lớp, chọn cặp tốt nhất. Ghi vào `results/e_b1/classical_grid.csv`.
- **Hạn chế cần ghi vào báo cáo:** khung nhãn VAIPE là từng cụm trường, ví dụ "1) RENAPRIL 5mg" là một khung. Khung của bản cổ điển to hay nhỏ phụ thuộc SE, nên khác "độ hạt" với nhãn chuẩn. Đây là nguyên nhân chính khiến F1 của bản cổ điển thấp. Phải nói rõ điều này để đánh giá công bằng.

### B7. Đo đạc — `metrics/detection.py`, `experiments/e_b1.py`, `e_b2.py`

- `match_greedy(pred, gt, iou_thr=0.5, class_aware)`: sắp tất cả cặp theo IoU giảm dần, mỗi box GT ghép tối đa một box dự đoán. Hỏi Hiếu trước xem `common/boxes.py` đã có hàm này chưa, tránh viết hai lần.
- P / R / F1 tính **cộng dồn cả tập** (cộng TP / FP / FN của mọi ảnh rồi mới tính), không lấy trung bình từng ảnh.
- Mỗi script ghi `results/<mã>/config.json`, gồm commit hash, `conf`, `imgsz`, đường dẫn weights, cùng `table.csv`.
- Chạy thử trên **valid** trước, để khi có weights chỉ việc đổi `--split test`.

### B10. Chạy số liệu cuối

1. **Chọn `conf` trên valid:** quét từ 0,05 đến 0,9, bước 0,05, chọn `conf` cho F1 không phân lớp lớn nhất. Ghi vào `results/e_b1/conf_sweep.csv`. Giá trị đã chọn dùng chung cho E-B1, E-B2 và pipeline của Hiếu. Báo Hiếu giá trị này.
2. **E-B1** trên test, P / R / F1 @IoU 0,5, **không phân lớp**:

   | Phương án | P | R | F1 | giây/ảnh (CPU) |
   | :--- | :---: | :---: | :---: | :---: |
   | Cổ điển (Sauvola + giãn + CC) | | | | |
   | YOLOv8s 1024 (bỏ nhãn lớp) | | | | |

   Thêm dòng 3: YOLO 1024 **có phân lớp** (bắt buộc đúng lớp mới tính TP), để thấy phân lớp làm F1 giảm bao nhiêu. Plan gốc ghi 4 dòng (có YOLO 640); nhóm đã bỏ 640 nên bảng còn 3 dòng.
3. **E-B2** trên test, YOLO 1024:
   - P / R / F1 từng lớp @IoU 0,5 (6 dòng + macro).
   - mAP50 lấy từ `model.val(split='test')`, cần `vaipe.yaml` local có khai báo `test`.
   - Ma trận nhầm lẫn: lưu `results/e_b2/confusion.png`.
   - Chọn 3 ảnh lỗi điển hình để vẽ overlay cho báo cáo. Chú ý cặp dễ nhầm `drugname` ↔ `other` và `usage` ↔ `other`.
4. **Test chỉ chạy một lần**, sau khi đã chốt `conf` và weights. Nếu phải chạy lại thì ghi lý do trong `config.json`.

---

## 3. Bàn giao cho nhóm

| Cho ai | Thứ gì | Khi nào |
| :--- | :--- | :--- |
| Hiếu (pipeline, E-X1, E-A1) | `code/weights/yolo_vaipe.pt` (Drive), `YoloDetector`, giá trị `conf` | ngay khi B9 xong |
| Thắng (E-D2) | `YoloDetector.detect()` trả về `TextBox` có `label` 6 lớp | cùng lúc |
| Khang | không phụ thuộc (C dùng box chuẩn) | — |
| Cả nhóm | `vaipe.yaml`, `to_yolo.py` (Khang có thể dùng lại để cắt crop) | sau B2 |

**Không commit:** `data/1_vaipe_yolo/`, zip, `code/weights/*.pt`, `runs/`. Kiểm tra `.gitignore` trước khi commit.

---

## 4. Báo cáo (phần B) + 3 slide

**Phương pháp (khoảng 0,5 trang):**
- YOLOv8s: lý do chọn bản s, `imgsz=1024` (kèm số đo chiều cao chữ ở mục 0), các augmentation đã tắt và vì sao.
- Đối chứng cổ điển: Sauvola (chương 5) → giãn SE theo `h` (chương 7) → thành phần liên thông. Tham số chọn trên valid.

**Kết quả (khoảng 0,5 trang):**
- Bảng E-B1, bảng E-B2 từng lớp, hình ma trận nhầm lẫn.
- Nhận xét:
  - DL hơn cổ điển bao nhiêu điểm F1. Lý do: cổ điển lệch "độ hạt" khung và không phân lớp được.
  - Vì sao 1024: dẫn số đo chiều cao chữ ở mục 0 (không train 640 để so).
  - Lớp nào hay nhầm với `other`.

**Phụ lục:** đường cong loss / mAP (`results.png`), lưới tham số của bản cổ điển, bảng quét `conf`, cấu hình train.

**3 slide:**
1. Bài toán B + kiến trúc YOLO + vì sao 1024 (hình phân bố chiều cao chữ).
2. E-B1: bảng so sánh + 1 ảnh overlay cổ điển và YOLO đặt cạnh nhau.
3. E-B2: F1 từng lớp + ma trận nhầm lẫn + 1 ca lỗi.

**Khai báo AI:** viết 1–2 câu cho mục Khai báo AI. Ví dụ: *"AI hỗ trợ viết code chuyển nhãn, notebook train và script đo; tôi tự chọn kích thước ảnh, chạy train, chọn ngưỡng trên valid và kiểm tra số liệu."* Câu này bạn phải tự viết lại bằng lời của mình.

---

## 5. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
| :--- | :--- | :--- |
| Upload Kaggle chậm hoặc lỗi | quá 30' chưa xong | tách zip làm 2 phần (train / valid); hoặc train trên Colab T4 với `batch=8` |
| Thư mục `/kaggle/input` chỉ đọc làm hỏng cache | lỗi khi quét nhãn | copy dataset sang `/kaggle/working/` |
| Hết bộ nhớ GPU ở 1024 | `CUDA out of memory` | `batch=8` |
| Train quá 2,5h | thời gian 1 epoch × 50 > 2,5h | `epochs=30` → `yolov8n` |
| Session Kaggle bị ngắt | notebook dừng | chạy bằng Commit (nền); thêm `save_period=10`; train lại bằng `resume=True` từ `last.pt` |
| F1 bản cổ điển gần 0 | khung khác độ hạt | đây là kết quả thật, không phải lỗi. Báo kèm F1 @IoU 0,3 làm số tham khảo, ghi rõ trong báo cáo |
| Hết giờ | — | cắt theo thứ tự: dòng 3 của E-B1 (YOLO có phân lớp) → overlay ca lỗi. **Không cắt:** YOLO 1024, E-B1 (cổ điển với 1024), E-B2 |

---

## 6. Định nghĩa "xong" của khối B

- [ ] `yolo_vaipe.pt` (1024) đã lên Drive, nhóm tải được và chạy được `YoloDetector` trên 1 ảnh.
- [ ] `results/e_b1/` và `results/e_b2/` có `config.json` + `table.csv` (tính trên test, chạy một lần).
- [ ] `conf` đã chọn trên valid và báo cho Hiếu.
- [ ] Phần B trong báo cáo đã có số + hình; 3 slide đã gửi Hiếu.
- [ ] Đã viết câu khai báo AI.
- [ ] `TienDo.md` đã cập nhật.
