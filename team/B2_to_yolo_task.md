# TASK B2 — `code/tools/to_yolo.py`: chuyển nhãn VAIPE sang định dạng YOLO + đóng zip cho Kaggle

> **Giao cho:** agent viết code (làm một mạch, không cần hỏi lại).
> **Người nhận kết quả:** Trường (khối B). Bối cảnh: [Truong_KhoiB.md](Truong_KhoiB.md) mục B2, [plan.md](../plan.md) mục 5.2.
> **Máy:** Windows 10, repo ở `D:\dip`, Python 3.13 hệ thống. Đã có sẵn `numpy`, `opencv-python` 5.0, `Pillow` 12.3.
> **Phạm vi:** chỉ tạo **2 file mới** là `code/tools/to_yolo.py` và `code/tools/__init__.py` (file rỗng, có thể bỏ). **Không sửa file nào khác** trong repo, vì Hiếu đang dọn repo song song.

---

## 1. Dữ liệu vào (đã kiểm tra)

```
data/1_vaipe/
├── train/images/*.png  train/labels/*.json   # 820 ảnh
├── valid/images/*.png  valid/labels/*.json   # 176 ảnh
├── test/images/*.png   test/labels/*.json    # 176 ảnh
├── test_unlabeled/                           # BỎ QUA
└── split_manifest.csv                        # BỎ QUA (có ảnh "excluded"; chỉ đọc theo thư mục)
```

- Ảnh là PNG, cùng tên với file nhãn: `VAIPE_P_TRAIN_0.png` ↔ `VAIPE_P_TRAIN_0.json`.
- Kích thước ảnh: 760×1070, 770×1420 hoặc 1150×1600. **Phải đọc kích thước thật** của từng ảnh (Pillow `Image.open(p).size`, không cần giải mã cả ảnh), không được ghi cứng.
- Nhãn là một list JSON, file mã hóa **UTF-8** (có tiếng Việt). Phải mở bằng `encoding="utf-8"`:
  ```json
  [{"id": 1, "text": "1/1", "label": "other", "box": [58, 28, 113, 58]}, ...]
  ```
  - `box` = `[x0, y0, x1, y1]`, đơn vị pixel, tính trên ảnh gốc.
  - `label` chỉ có 6 giá trị. Số khung trên cả 3 tập: other 28.679 · usage 3.994 · drugname 2.843 · quantity 2.843 · diagnose 1.649 · date 1.172.

## 2. Dữ liệu ra

```
data/1_vaipe_yolo/
├── images/train/*.png   images/valid/*.png   images/test/*.png    # COPY file (shutil.copy2), không symlink
├── labels/train/*.txt   labels/valid/*.txt   labels/test/*.txt
├── vaipe.yaml           # bản local, có cả test
└── stats.json           # thống kê để kiểm tra (mục 4)
data/vaipe_yolo.zip      # bản đẩy Kaggle: CHỈ train + valid + vaipe.yaml (bản Kaggle)
```

### 2.1 File nhãn `.txt`

- Mỗi khung là một dòng: `cls cx cy w h`, các giá trị chuẩn hóa về khoảng 0–1 theo `W` và `H` của ảnh. Ghi **6 chữ số thập phân**, ngăn cách bằng một dấu cách, xuống dòng bằng `\n`.
- **Thứ tự lớp cố định** (khai báo một hằng `CLASSES` ở đầu file):
  `0 drugname, 1 quantity, 2 usage, 3 diagnose, 4 date, 5 other`
- **Quy tắc xử lý khung:**
  1. Kẹp tọa độ vào khung ảnh: `x0, x1` trong `[0, W]`, `y0, y1` trong `[0, H]`. Nếu `x0 > x1` hoặc `y0 > y1` thì đổi chỗ hai giá trị.
  2. Sau khi kẹp, nếu `x1 - x0 <= 1` hoặc `y1 - y0 <= 1` thì **bỏ khung đó** và đếm vào `dropped`.
  3. Gặp `label` không thuộc `CLASSES` thì **dừng chương trình** (`raise ValueError`), báo rõ tên file và nhãn. Không được đoán hay gộp vào `other`.
- Ảnh không có khung nào vẫn phải có file `.txt` rỗng.
- Ảnh không có file nhãn, hoặc nhãn không có ảnh tương ứng: dừng và báo lỗi.

### 2.2 `vaipe.yaml`

Hai bản, nội dung chỉ khác nhau ở `path` và `test`. Tự viết chuỗi YAML (không cần cài `pyyaml`):

```yaml
# bản local: data/1_vaipe_yolo/vaipe.yaml
path: D:/dip/data/1_vaipe_yolo     # đường dẫn tuyệt đối, dùng dấu "/"
train: images/train
val: images/valid
test: images/test
names:
  0: drugname
  1: quantity
  2: usage
  3: diagnose
  4: date
  5: other
```

- **Bản trong zip:** `path: /kaggle/input/vaipe-yolo` và **không có dòng `test`**.
- Notebook train sẽ ghi đè `path` sau. Bản trong zip chỉ cần đúng cấu trúc.

### 2.3 Zip cho Kaggle

- Tạo `data/vaipe_yolo.zip` bằng `zipfile` với `ZIP_STORED`. Ảnh PNG đã nén sẵn, nên nén thêm (deflate) chỉ tốn thời gian mà không giảm được bao nhiêu dung lượng.
- **Bố cục bên trong zip**, ở gốc, không có thư mục bao ngoài:
  ```
  images/train/...  images/valid/...  labels/train/...  labels/valid/...  vaipe.yaml
  ```
- **Tuyệt đối không đưa `images/test` hay `labels/test` vào zip.** Test chỉ được chạy một lần ở máy local.
- Tên file trong zip dùng dấu `/`, không dùng `\`. Trên Windows, `arcname` phải tạo bằng `Path.as_posix()`.
- `.gitignore` đã chặn cả `/data/*` lẫn `*.zip`, nên không lo commit nhầm. Không cần sửa `.gitignore`.

## 3. Giao diện dòng lệnh

```
python code/tools/to_yolo.py [--src data/1_vaipe] [--dst data/1_vaipe_yolo] [--zip data/vaipe_yolo.zip]
                             [--no-zip] [--check N] [--force]
```

- **Mặc định**: tự suy đường dẫn repo từ vị trí file, tức `Path(__file__).resolve().parents[2]`. **Không import `common.paths`**, vì Hiếu có thể đang chuyển module đó. Script phải chạy độc lập, không phụ thuộc gì trong `code/src`.
- `--force`: nếu `dst` đã tồn tại thì xóa rồi tạo lại. Không có `--force` mà `dst` đã có thì dừng và báo lỗi, để tránh trộn file cũ với file mới.
- `--check N` (mặc định 3): vẽ lại khung từ file `.txt` (không lấy từ JSON) lên N ảnh train chọn ngẫu nhiên với `random.Random(42)`. Mỗi lớp một màu, có ghi tên lớp. Lưu vào `results/e_b0/check_<image_id>.png`.
- Có `if __name__ == "__main__":` và hàm `main(argv=None)`, để test hoặc notebook gọi được.
- Hàm cần tách riêng để dùng lại (Khang có thể cần):
  - `xyxy_to_yolo(box, W, H) -> tuple[float, float, float, float] | None` (trả `None` nếu khung bị bỏ)
  - `convert_split(src, dst, split) -> dict`
  - `write_yaml(path, root, include_test)`
  - `make_zip(dst, zip_path)`
- In tiến độ đơn giản cho từng tập (`train: 820 ảnh, N khung, bỏ k`). Không cần cài thêm thư viện như `tqdm`.

## 4. Kiểm tra bắt buộc (script tự chạy ở cuối; sai thì `sys.exit(1)`)

1. Số ảnh bằng số file `.txt` trong mỗi tập, và phải đúng **820 / 176 / 176**.
2. Trong mỗi tập: tổng số dòng `.txt` + `dropped` = tổng số khung trong JSON.
3. Mọi giá trị `cx, cy, w, h` nằm trong `(0, 1]`, riêng `cx, cy` nằm trong `[0, 1]`. Mọi `cls` thuộc `0..5`.
4. **Kiểm tra quay ngược:** lấy 1 ảnh mỗi tập, đổi file `.txt` về lại pixel. Mỗi khung phải lệch so với box trong JSON (sau khi kẹp) không quá **1 px**.
5. Trong zip không có tên file nào chứa `test`, và số file ảnh trong zip = 820 + 176.
6. Ghi `stats.json`:
   - Số ảnh, số khung, số bị bỏ theo từng tập.
   - Số khung theo lớp, theo từng tập.
   - Dung lượng zip (MB).
   - Thời gian chạy.

   Đồng thời in một bảng tóm tắt ra màn hình. Cộng số khung theo lớp của cả 3 tập (cộng cả phần bị bỏ) thì phải **khớp đúng các số ở mục 1**.

## 5. Ràng buộc

- Chỉ dùng thư viện chuẩn + Pillow. OpenCV chỉ dùng cho phần vẽ `--check`.
- Đặt seed 42 cho mọi chỗ có ngẫu nhiên.
- Viết tiếng Việt cho docstring và chú thích, theo phong cách của repo. Tên biến và hàm viết tiếng Anh.
- **Không** sửa `plan.md`, `team/*`, `code/src/*`, `.gitignore`. **Không** commit, **không** push. Trường sẽ tự review rồi commit.
- Tên file có thể có ký tự đặc biệt: dùng `pathlib` trong mọi thao tác đường dẫn.

## 6. Cách chạy để nghiệm thu

```bash
cd /d/dip
python code/tools/to_yolo.py --force
python code/tools/to_yolo.py            # chạy lại không có --force → phải dừng với thông báo rõ
```

**Báo lại cho Trường:**
- Bảng `stats.json`, gồm số khung bị bỏ và dung lượng zip.
- Đường dẫn 3 ảnh trong `results/e_b0/`. Agent tự mở xem: khung có ôm đúng chữ không, màu của từng lớp có hợp lý không (dòng "1) TÊN THUỐC" phải là `drugname`, "SL: 28" phải là `quantity`).
- Thời gian chạy.
- Mọi điểm đã phải tự quyết mà task này không nói rõ.

## 7. Định nghĩa "xong"

- [ ] `python code/tools/to_yolo.py --force` chạy hết, thoát với mã 0, mọi kiểm tra ở mục 4 đều qua.
- [ ] `data/vaipe_yolo.zip` chỉ chứa train + valid + `vaipe.yaml` (`path` là đường dẫn Kaggle, không có `test`).
- [ ] 3 ảnh kiểm tra khung trông đúng bằng mắt.
- [ ] Chỉ có file mới trong `code/tools/`, `git status` không cho thấy file nào khác bị sửa.
