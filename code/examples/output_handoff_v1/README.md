# Output handoff v1 — JSON đầu ra ③ và ④

> **Trạng thái:** bản đề xuất 02/11, **chưa chốt**: chờ Khang xác nhận phần ③ và Trường (bên nhận) xác nhận cả hai.
> **Hợp đồng:** `LabReportResult` (Khang ③) và `PrescriptionResult` (Thắng ④) trong [`src/common/contracts.py`](../../src/common/contracts.py) — bộ này **không đổi** hợp đồng, chỉ cung cấp ví dụ khớp ảnh thật.
> **Bên nhận:** component xác nhận dùng chung của Trường (`m05`, `m06`, `l02`, `l03`, `l10`) — xem [KeHoach_Truong.md](../../../team/KeHoach_Truong.md).

## 1. Nội dung

| File | Nội dung |
| :--- | :--- |
| `prescription_sample.json` | `PrescriptionResult` của ảnh `cleaning_handoff_v1/handoff_prescription_001` (2 khoản DEMO) |
| `lab_report_sample.json` | `LabReportResult` của ảnh `cleaning_handoff_v1/handoff_lab_report_001` (3 dòng DEMO) |
| `verify_output_handoff.py` | Nghiệm thu: validate hợp đồng, round-trip, bbox khớp chữ trên `binary.png`, xuất overlay |
| `preview_*.png` | Overlay bbox + nội dung JSON lên ảnh, để người duyệt đối chiếu |
| `manifest.json` | Danh sách mẫu và đường dẫn |

Hai JSON là **ground truth viết tay theo đúng chữ trên ảnh fixture** của Trường (ảnh tổng hợp, ghi rõ "NOT A REAL MEDICAL DOCUMENT"), nên `confidence = 1.0`. Trường nào ảnh không có thì để `null` (đơn thuốc không có bệnh nhân / chẩn đoán; phiếu xét nghiệm không có loại phiếu và mã LOINC).

## 2. Quy ước

* `bbox = [x, y, w, h]` (int) trên **ảnh đã nắn** `rectified.png` của ①; crop: `rectified[y:y+h, x:x+w]`.
* `items[].bbox` bao trọn một khoản thuốc (tên + hàm lượng + liều + cách dùng); `rows[].bbox` bao trọn một dòng chỉ số.
* `dose`: số đơn vị mỗi buổi (`morning` / `noon` / `evening`), không dùng thì `0`.
* `flag` ∈ `LOW`, `NORMAL`, `HIGH`, `UNKNOWN`; `range = [thấp, cao]` hoặc `null`.

## 3. Kiểm tra

```
cd code
python examples/output_handoff_v1/verify_output_handoff.py
```

Tự động bắt: sai hợp đồng, bbox ra ngoài ảnh, bbox trỏ vào vùng trống hoặc lệch so với dòng chữ (IoU giữa bbox và khung mực < 0,5). **Không** tự kiểm được nội dung chữ — người duyệt xem `preview_*.png`.

## 4. Cần nhóm quyết trước khi chốt

| # | Vấn đề | Căn cứ | Đề xuất |
| :---: | :--- | :--- | :--- |
| 1 | `Dose` thiếu buổi **chiều** | Đơn VN ghi Sáng / Trưa / Chiều / Tối (GT 30 ảnh có "Chiều 2 Viên"); UX `02-medication` có giờ nhắc chiều 17:00 | Thêm `afternoon: float = 0` vào `Dose` |
| 2 | Không có **số lượng** (`SL: 28 Viên`) | Đơn VAIPE thường không ghi số ngày; UX tính thời gian dùng = số lượng ÷ liều mỗi ngày | Thêm `quantity: Optional[str] = None` vào `MedicineItem` |

Đổi hợp đồng theo quy ước `code/README.md` mục 4: báo nhóm, sửa `contracts.py` + `tests/test_contracts.py` trong cùng một commit, rồi cập nhật hai JSON mẫu.
