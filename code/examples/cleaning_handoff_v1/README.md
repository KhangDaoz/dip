# Bộ bàn giao mẫu CleaningResult v1 — 01/11/2026

**Mục đích:** Khang và Thắng kiểm tra đọc JSON, nạp ảnh, lấy vùng chữ và crop OCR theo đúng hợp đồng. Bộ này có hai tài liệu **giả lập**, không có thông tin bệnh nhân và không phải dữ liệu đánh giá thuật toán. Ảnh xám/nhị phân là kết quả chạy thuật toán thật: Median 41 → chia nền → Sauvola window 31, k 0.2, R 128.

## 1. Mở file nào?

| Người nhận | JSON đầu vào chính thức theo cấu trúc CleaningResult |
|---|---|
| Khang — phiếu xét nghiệm | `handoff_lab_report_001/cleaning_result.json` |
| Thắng — đơn thuốc | `handoff_prescription_001/cleaning_result.json` |

Trong mỗi thư mục:

- `rectified.png`: ảnh màu BGR khi đọc bằng `common.io.read_image`, đã phẳng ngay từ lúc sinh mẫu, chưa CLAHE.
- `preprocess.json`: PreprocessResult đi kèm để nối image_id và hệ tọa độ.
- `clean_gray.png`: ảnh xám sau hiệu chỉnh nền, dùng crop OCR.
- `binary.png`: ảnh nhị phân, dùng tách dòng/span và morphology.
- `cleaning_result.json`: đối tượng CleaningResult, có đúng các khóa của `contracts.py`.
- `ocr_crop_example.png`: crop mẫu từ ảnh xám, theo bbox minh họa trong manifest; không phải kết quả OCR.
- `preview.png`: ba cột **Đầu vào màu | Xám đã khử bóng | Nhị phân đảo màu để xem**. Không đưa preview vào thuật toán.

`manifest.json` là mục lục/provenance/hash và bbox minh họa; **không** truyền cả manifest vào `CleaningResult.from_dict`. `validation.json` là kết quả kiểm tra lúc tạo bộ mẫu.

Checksum SHA-256 của văn bản `.md`, `.py`, `.json` được tính sau khi đổi CRLF thành LF để không báo sai khi Git đổi xuống dòng trên Windows. File ảnh được kiểm tra nguyên byte. Manifest khai báo `sha256_text_newlines="lf"`; cả reader và verifier dùng cùng quy ước này. Hai file `manifest.json` và `validation.json` không nằm trong danh sách checksum.

## 2. Quy ước cần giữ

1. Đường dẫn trong JSON tính từ **gốc repo**, không tính từ thư mục chứa JSON. Mở bằng `REPO_DIR / result.binary_path` ngay cả khi terminal đang ở thư mục `code/`.
2. `image_id` giống nhau giữa PreprocessResult và CleaningResult. Loại tài liệu nằm trong PreprocessResult/manifest, không thêm trường `doc_type` vào CleaningResult.
3. Ảnh xám và nhị phân đều là uint8, một kênh, **1100 hàng × 800 cột**; khớp kích thước ảnh màu đầu vào.
4. Nhị phân: **mực = 255, nền = 0**. Nếu thư viện cần giấy trắng/chữ đen thì đảo `255 - binary` trên bản sao. Không sửa file bàn giao.
5. Bbox là `[x, y, w, h]`. Crop NumPy: `gray[y:y+h, x:x+w]`. Mọi tọa độ tính trên ảnh màu đầu vào đã phẳng; làm sạch không resize/cắt/xoay thêm.
6. Dùng nhị phân để tìm span; lấy cùng bbox crop trên `clean_gray` cho OCR. Ảnh đã dilation chỉ dùng tìm vùng, không dùng làm ảnh OCR. Muốn xét màu, crop trên ảnh màu trong `preprocess.rectified_path`.
7. `masks={}` và `components=[]` nghĩa là **chưa tính các đầu ra này**, không phải đã phát hiện không có dấu/chữ ký/chữ đỏ. Cần Khang/Thắng xác nhận module nhận xử lý được bản tối thiểu này trước khi chốt tích hợp. Nếu module cần mask/component để chạy, bộ mẫu hiện tại chưa đáp ứng; không mặc định đã có `masks['stamp']`.
8. Khi có mask đầy đủ, khóa dự kiến là `stamp`, `handwriting`, `red_text`; nhãn component là `printed`, `handwritten`, `stamp`, `other`. Mask theo vùng, không trộn thành nhãn vai trò từ/dòng.

Ảnh giả lập đã thẳng nên H là ma trận đơn vị, góc nghiêng 0 và `corner_method='full_image'`. Đây là **đầu vào giả lập**, không phải kết quả chạy pipeline Hiếu. `doc_type` và `doc_confidence=1.0` được gán cho fixture để kiểm tra định tuyến, không phải dự đoán mô hình. Không dùng các giá trị này để kết luận chất lượng phân loại/nắn ảnh.

## 3. Đọc và kiểm tra nhanh

Từ gốc repo, dùng Python trong môi trường đã có NumPy/OpenCV và package dự án:

```powershell
python code/examples/cleaning_handoff_v1/read_sample.py
```

Lệnh kiểm tra cả hai mẫu: parse hợp đồng, checksum, kích thước/kênh, giá trị nhị phân, nối image_id và crop ảnh xám. Nó không gọi OCR và chưa chạy module tách bảng/tách đơn vì hai pipeline nhận hiện còn stub trong checkout lúc tạo mẫu.

Để kiểm tra thêm tham số khai báo và tính lại ảnh bằng thuật toán:

```powershell
python code/scripts/build_cleaning_handoff.py --verify-only
```

Verifier này chỉ chấp nhận cấu hình cố định của **bộ mẫu v1** (Median 41, Sauvola window 31, k 0.2, R 128 và các cờ trạng thái trong JSON). Đây không phải giới hạn tham số của `CleaningResult` dùng chung hoặc của thuật toán cho dữ liệu khác. `read_sample.py` kiểm tra khả năng đọc bộ mẫu, không thay cho bước kiểm tra lại thuật toán này.

Code sử dụng ở phía nhận:

```python
from common.contracts import CleaningResult
from common.io import load_json, read_image
from common.paths import REPO_DIR

path = REPO_DIR / 'code/examples/cleaning_handoff_v1/handoff_lab_report_001/cleaning_result.json'
cleaned = CleaningResult.from_dict(load_json(path))
binary = read_image(REPO_DIR / cleaned.binary_path, gray=True)
gray = read_image(REPO_DIR / cleaned.clean_gray_path, gray=True)

# Bbox của span do module nhận tự tìm. Ví dụ minh họa đã có trong manifest:
x, y, w, h = [45, 315, 710, 45]
ocr_crop = gray[y:y+h, x:x+w]
# Không coi bbox minh họa này là component do phần cleaning phát hiện.
```

## 4. Xác nhận bàn giao

Khang/Thắng cần xác nhận: mở được hai file ảnh từ JSON; xử lý đúng chiều mực/nền; crop đúng vùng; module không lỗi khi masks/components rỗng. Sau đó thử module của mình. Chỉ đánh dấu tích hợp thành công khi người nhận thực sự chạy được; bộ mẫu tự kiểm tra đạt chưa thay cho xác nhận của người nhận.

## 5. Chia sẻ và giới hạn

Thư mục này đặt ở `code/examples/` vì là fixture giả lập có thể chia sẻ cùng code. Ảnh y tế thật dùng phát triển nằm ở `img_test`, kết quả chạy thử nằm ở `results`; chúng không thuộc bộ bàn giao mẫu này. Không thay đổi `common/contracts.py` hoặc `cleaning.pipeline.clean`.

Bản ZIP local: `results/cleaning_handoff_v1.zip`. Giải nén **tại gốc repo** để được đúng `code/examples/cleaning_handoff_v1/`; không giải nén thêm một tầng thư mục. Để chạy reader, cần các module `common` sẵn trong repo của nhóm.

Bộ mẫu chưa tách mực, xóa dấu/chữ ký, gán component, chạy OCR hay đo độ chính xác. Nó kiểm chứng giao diện tối thiểu, chưa phải bản đầy đủ đạt mốc tích hợp 05/11.
