# Nguồn và license — bộ dữ liệu ①–③

Đối chiếu ngày **01/11/2026**. Mục đích là ghi chính xác khai báo của bên phát hành và điểm còn thiếu của gói nhóm đang dùng. Không sửa `data.yaml`, không gán license cho dữ liệu chỉ dựa trên tên tương tự.

## Kết quả

| Bộ local | Nguồn đã đối chiếu | License được nguồn khai báo | Trạng thái |
|---|---|---|---|
| `1_vaipe/` | Kaggle `tommyngx/vaipepill2022`, version 1, dataset ID 2347404 | API metadata: `DbCL-1.0`; API view: `Database: Open Database, Contents: Database Contents` | **Đã xác minh khai báo của bản Kaggle**. Chưa có tài liệu xác nhận riêng điều khoản của ban tổ chức VAIPE 2022 cho gói này. |
| `2_roboflow_prescription/` | Roboflow `prescription-image-vkedm/prescription-image-labelling-xh0np`, trang version 1 `detection` | Trang version 1 ghi **CC BY 4.0**; YAML local ghi **`Private`** | **Đã xác minh khai báo công khai; còn mâu thuẫn với metadata gói local**. Cần người xuất/tải gói xác nhận. |
| `3_roboflow_report_keywords/` | Roboflow `world-of-ai/report-keywords-detection`, trang dự án và metadata version 1 local | **CC BY 4.0** ở cả trang công khai và YAML local | **Khớp khai báo giữa nguồn và gói local**. Trang version 1 riêng chưa đọc được qua công cụ tra cứu. |

“Đã xác minh khai báo” nghĩa là đọc được thông tin license của nguồn; không phải đã kiểm toán quyền sở hữu từng ảnh, xác minh bằng checksum toàn bộ gói nguồn, hoặc xác nhận quyền công bố thông tin người bệnh.

## ① VAIPE: ghi đúng bản Kaggle và giới hạn truy nguyên

Nguồn bản nhóm ghi nhận: [vaipepill2022 — Tommy NgX / tommyngx](https://www.kaggle.com/datasets/tommyngx/vaipepill2022).

Hai API công khai đã trả JSON, không cần đăng nhập:

- [Metadata](https://www.kaggle.com/api/v1/datasets/metadata/tommyngx/vaipepill2022): `info.datasetId = 2347404`, `ownerUser = tommyngx`, `datasetSlug = vaipepill2022`, `licenses[0].name = DbCL-1.0`. Trường nguồn do người đăng khai báo (`userSpecifiedSources`) rỗng.
- [View](https://www.kaggle.com/api/v1/datasets/view/tommyngx/vaipepill2022): người đăng `Tommy NgX`, version hiện tại **1**, ngày tạo version **18/07/2022 UTC**; `licenseName = Database: Open Database, Contents: Database Contents`.

[Tài liệu metadata chính thức của Kaggle](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md) ánh xạ đúng tên khai báo này sang `DbCL-1.0`. Không đổi nó thành CC BY 4.0, CC0 hay MIT.

DbCL đề cập điều kiện tuân thủ ODbL. Vì vậy cần giữ cả dẫn chiếu [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/) và [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). Khi công bố cơ sở dữ liệu/phần dẫn xuất, cần đối chiếu điều khoản về thông báo nguồn, license và chia sẻ tương tự; không suy rằng license này tự động đổi license của toàn bộ code MedTrack.

Về nguồn ban đầu: [trang tài nguyên VinUni–Illinois Smart Health Center](https://smarthealth.vinuni.edu.vn/resources/) có giới thiệu các bộ VAIPE, nhưng chưa cung cấp bằng chứng gắn điều khoản cụ thể với đúng 1.345 ảnh đơn thuốc local. Các bộ VAIPE khác có quy mô khác; không dùng chúng để thay số liệu hoặc suy license cho gói 2022. Trang `vaipe.org` không truy cập được trong lần kiểm tra; link `vaipe.io` từ trang tài nguyên chuyển sang một miền khác, không dùng miền chuyển hướng làm bằng chứng license.

**Cách ghi trong báo cáo nhóm:** “Sử dụng phần đơn thuốc từ bản `tommyngx/vaipepill2022` trên Kaggle, version 1; metadata Kaggle khai báo `DbCL-1.0` (Database: Open Database, Contents: Database Contents). Chưa xác minh điều khoản riêng của nguồn VAIPE ban đầu.”

## ② Roboflow đơn thuốc: nguồn công khai đã rõ, YAML còn mâu thuẫn

[Trang version 1 chính xác](https://universe.roboflow.com/prescription-image-vkedm/prescription-image-labelling-xh0np/dataset/1) ghi bên đăng hiển thị **Prescription Image**, tên version `detection`, ngày tạo **08/10/2025**, tổng **9.362 ảnh**, chia **8.427 / 560 / 375**, license **CC BY 4.0**. Ngày tạo, số công bố, resize, grayscale và tăng cường khớp mô tả README xuất local; gói local thực đếm ít hơn nguồn một file train, chưa xác định lý do.

[Trang tổng quan cùng dự án](https://universe.roboflow.com/prescription-image-vkedm/prescription-image-labelling-xh0np) hiển thị 3.597 ảnh và 7 lớp. Đây là số ở màn hình tổng quan, không phải số ảnh xuất của version 1. Không thể chỉ từ hai con số này kết luận đó là hai dự án khác nhau. Bản local version 1 chỉ có một lớp `medicine-names`, như kiểm kê đã ghi.

**Mâu thuẫn chưa giải quyết:** `data/2_roboflow_prescription/data.yaml` ghi `roboflow.license: Private`, mặc dù workspace/project/version trùng đường dẫn công khai. `Private` ở trường metadata không tự mô tả các quyền sao chép, sửa đổi hoặc phân phối. Chưa biết vì sao gói xuất ghi như vậy; không tự kết luận đây là một license khác, lỗi xuất hay quyền truy cập workspace.

**Cách ghi trong báo cáo nhóm:** “Nguồn công khai Roboflow version 1 khai báo CC BY 4.0; gói local ghi `Private` trong YAML, đang cần xác nhận nguồn xuất.” Giữ nguyên YAML làm chứng cứ. Nếu cần đưa ảnh/gói này ra ngoài nhóm, làm rõ mâu thuẫn với người cung cấp trước.

**Thông tin cần người tải/xuất gói bổ sung:** link hoặc bản ghi nguồn xuất cụ thể của gói đang dùng, kèm phần license và lịch sử thay đổi nếu có; xác nhận có fork/đổi lớp hoặc thao tác xuất nào khiến YAML ghi `Private`. Chưa đủ căn cứ để yêu cầu tải lại bộ dữ liệu.

## ③ Report Keywords: nguồn và metadata khớp

[Trang công khai dự án](https://universe.roboflow.com/world-of-ai/report-keywords-detection) ghi bên đăng **world of AI**, license **CC BY 4.0**, 57 ảnh ở tổng quan và 5 lớp. YAML local trỏ đúng workspace/project này, version 1 và cùng license. Gói xuất gồm 135 ảnh có tăng cường; không phải 135 ảnh gốc khác nhau.

Trang version 1 riêng không trả nội dung đọc được qua công cụ web; yêu cầu HTTP trực tiếp bị 403. Do đó chưa nói đã xác minh độc lập mọi chi tiết version 1 trên web. Căn cứ license hiện tại là trang dự án công khai **và** YAML của chính gói local.

**Cách ghi trong báo cáo nhóm:** “Report Keywords Detection — world of AI, Roboflow Universe, bản xuất version 1 theo metadata local; CC BY 4.0, đối chiếu trang dự án ngày 01/11/2026.” Dẫn link nguồn và ghi rõ các biến đổi nhóm thực hiện thêm.

## Ghi nguồn khi sử dụng

Với các phần được nguồn cấp [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), cần ghi bên tạo/đăng được cung cấp, link dataset, link license và việc đã sửa đổi. Không mô tả tác giả dataset là người bảo trợ hoặc xác nhận kết quả MedTrack. Các quyền về thông tin cá nhân không tự được giải quyết bởi license dữ liệu.

Mẫu ngắn cho bộ ③:

> Dữ liệu: report keywords detection, world of AI, Roboflow Universe; bản xuất v1 theo metadata local. Nguồn: https://universe.roboflow.com/world-of-ai/report-keywords-detection . License: https://creativecommons.org/licenses/by/4.0/ . MedTrack thực hiện [ghi các phép biến đổi thực sự đã dùng].

Bộ ② ghi cùng loại thông tin nhưng phải kèm ghi chú mâu thuẫn `Private` ở trên. VAIPE ghi tên người đăng Kaggle và cả hai link DbCL/ODbL; không gán người đăng Kaggle là tác giả gốc của toàn bộ ảnh.

## Bằng chứng và trạng thái công việc

Chứng cứ bổ sung nằm tại `results/primary_dataset_audit/license_evidence/` trên máy Trường: JSON phản hồi API Kaggle, bản sao metadata local và `evidence_index.json`. Thư mục này bị gitignore, không đi kèm repo; các URL nguồn và trường metadata cần đối chiếu đã ghi trong tài liệu này. Với Roboflow, thông tin web được ghi lại dưới dạng ghi nhận có URL; không có bản HTML tải thành công. Phân biệt ghi nhận từ công cụ web với phản hồi HTTP thô.

Tài liệu đã đối chiếu khai báo nguồn và ghi rõ giới hạn. Còn cần người cung cấp bộ ② làm rõ YAML `Private`; nguồn VAIPE ban đầu chưa có xác nhận quyền riêng cho gói 2022. Không xem các trạng thái còn mở là đã được xác nhận chỉ vì README đã hoàn thiện.
