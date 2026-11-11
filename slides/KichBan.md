# Kịch bản thuyết trình — Trích xuất thông tin đơn thuốc từ ảnh

Nhóm 01 · E23TTNT01 · 16 slide · khoảng 10 phút trình bày.
Nguồn số liệu: báo cáo `report/latex` (bản build 11/11) và `report/latex/result/`. Lời thoại này sẽ được đưa vào phần ghi chú người nói (speaker notes) của file PPTX.

---

## 1. Bìa
Kính chào thầy cô và các bạn. Nhóm 01 xin trình bày đề tài "Trích xuất thông tin đơn thuốc từ ảnh bằng học sâu kết hợp xử lý ảnh cổ điển", thuộc lựa chọn 2 của học phần: dự án xử lý ảnh thực tế. Nhóm gồm bốn thành viên, mỗi người phụ trách một bước của chuỗi xử lý.

## 2. Bối cảnh và động cơ
Mỗi lần đi khám, người bệnh mang về một tờ đơn thuốc; sau vài năm, những tờ đơn này thành một xấp giấy rời rạc từ nhiều cơ sở. Người chịu ảnh hưởng nhiều nhất là người cao tuổi và người bệnh mạn tính: Việt Nam có 14,2 triệu người từ 60 tuổi trở lên, 26,2% người từ 18 đến 69 tuổi bị tăng huyết áp, và một nghiên cứu năm 2023 ghi nhận 41,6% bệnh nhân tăng huyết áp quên uống thuốc trong hai tuần. Hồ sơ bệnh án chưa được tập trung, nên tờ đơn giấy vẫn là nguồn đầy đủ nhất. Nhóm hướng tới ứng dụng MedTrack AI; trong phạm vi môn học, đồ án làm khâu lõi: đọc ảnh đơn thuốc thành dữ liệu có cấu trúc.

## 3. Phát biểu bài toán
Đầu vào là ảnh chụp một đơn thuốc. Đầu ra là bản ghi JSON gồm danh sách khoản thuốc, mỗi khoản có tên, số lượng, đơn vị và cách dùng, cùng chẩn đoán và ngày kê đơn. Ví dụ bên phải là đầu ra thật của hệ thống trên chính ảnh bên trái.
Bài toán có ba khó khăn. Thứ nhất, chữ rất nhỏ, chỉ cao 20 đến 32 pixel, trong khi nhiều thuốc chỉ khác nhau ở hàm lượng, như CEPHALEXIN 250MG và CEPHALEXIN PMP 500: sai một chữ số là thành thuốc khác liều. Thứ hai, tên, số lượng và cách dùng của một khoản nằm rời nhau nên phải ghép lại. Thứ ba, ảnh chụp thật có thể nghiêng, mờ hoặc đổ bóng.

## 4. Dữ liệu
Nhóm dùng phần đơn thuốc của bộ VAIPE, gồm 1.172 ảnh đơn bảo hiểm y tế in máy có nhãn. Tập kiểm tra gốc không có nhãn, nên nhóm chia lại thành 820 ảnh huấn luyện, 176 ảnh kiểm định và 176 ảnh kiểm tra; ảnh của cùng một bệnh nhân luôn nằm trong cùng một tập để tránh rò rỉ dữ liệu. Mỗi vùng chữ thuộc một trong sáu lớp, gần 70% là lớp "khác" như tiêu đề hay mã số. Tham số được chọn trên tập kiểm định; tập kiểm tra chỉ dùng một lần ở cuối. Ngoài ra, ảnh kiểm tra được làm suy giảm thành hai mức nhẹ và nặng để đo độ bền.

## 5. Tổng quan phương pháp
Chuỗi xử lý gồm bốn bước theo khung của học phần. Tiền xử lý dùng các kỹ thuật cổ điển của chương 2, 3, 5 và 7, và là bước tùy chọn. Xử lý chính gồm YOLOv8s phát hiện và phân lớp vùng chữ, rồi VietOCR đã tinh chỉnh đọc chữ trong từng vùng. Hậu xử lý ghép các vùng thành khoản thuốc bằng luật vị trí và từ điển tên thuốc. Mỗi bước có một phương pháp đối chứng, và toàn bộ chuỗi được so với mô hình thị giác–ngôn ngữ Qwen3.8-27B ở chế độ zero-shot, đọc thẳng ảnh ra JSON.

## 6. Tiền xử lý ảnh
Tiền xử lý đưa ảnh chụp về gần ảnh quét qua năm bước. Nắn phối cảnh dùng biên Canny để tìm tứ giác tờ giấy rồi kéo về hình chữ nhật. Chỉnh nghiêng nhị phân hóa Sauvola, rồi chọn góc làm cực đại phương sai lược đồ chiếu ngang: khi dòng chữ nằm ngang, mỗi hàng hoặc nhiều mực hoặc gần như trống. Khử bóng dựa trên mô hình chiếu sáng–phản xạ của chương 2: ước lượng nền bằng phép đóng 81×81, lớn gấp 2,5 đến 4 lần chiều cao chữ nên xóa nét chữ, rồi chia ảnh cho nền. Cuối cùng là lọc trung vị và CLAHE. Hình minh họa một ảnh suy giảm nặng qua các bước.

## 7. Phát hiện vùng chữ
Phương pháp chính là YOLOv8s, bộ phát hiện một giai đoạn: mạng nhìn toàn ảnh một lần và dự đoán luôn khung bao cùng lớp. Nhóm khởi tạo từ trọng số COCO, huấn luyện 50 epoch; giữ ảnh vào 1024 pixel vì nếu thu nhỏ hơn, chữ chỉ còn khoảng 14 pixel; tắt lật ngang vì làm ngược chữ; ngưỡng tin cậy 0,443 được chọn trên tập kiểm định.
Đối chứng là bộ phát hiện hình thái học: nhị phân hóa Sauvola, phép mở loại nhiễu, rồi phép giãn theo chiều cao chữ h, rộng 2,5h và cao 0,2h, để nối ký tự thành cụm mà không nối qua cột. Bộ này không phân lớp được.

## 8. Nhận dạng chữ
Mỗi vùng chữ được cắt theo khung, đệm 2 pixel, rồi đưa vào VietOCR: VGG19 trích đặc trưng, Transformer sinh từng ký tự có dấu. Phương án chính là tinh chỉnh VietOCR trên toàn bộ vùng chữ của tập huấn luyện, 20.000 vòng lặp, và nâng chiều rộng tối đa lên 1024 pixel để dòng chẩn đoán dài không bị nén mất chữ. Ba đối chứng là VietOCR gốc, chưa tinh chỉnh; Tesseract xám, đọc ảnh xám phóng hai lần; và Tesseract nhị phân, đọc ảnh đã nhị phân hóa Sauvola, để xem nhị phân hóa có giúp đọc tốt hơn không.

## 9. Hậu xử lý
Hậu xử lý biến danh sách vùng chữ thành bản ghi đơn thuốc qua bốn bước. Lớp lấy từ YOLOv8s, chỉ xét lại bằng biểu thức chính quy khi YOLO gán lớp "khác" hoặc độ tin cậy thấp. Trong hình, mỗi tên thuốc, khung đỏ, mở một dải ngang tới tên kế tiếp, khung vàng; số lượng và cách dùng nằm trong dải thì thuộc khoản đó. Sau đó chữ được chuẩn hóa, và tên thuốc được so khớp với từ điển 138 tên của tập huấn luyện bằng độ tương tự Levenshtein; nhóm không thay nếu có nguy cơ kéo sang thuốc khác hàm lượng. Chỉ 2 trong 443 khoản kiểm tra nằm ngoài từ điển.

## 10. Thiết lập thực nghiệm và độ đo
Mọi số liệu tính trên 176 ảnh kiểm tra. Với phát hiện, cặp khung có IoU từ 0,5 là dương tính đúng; Precision, Recall và F1 theo chương 8 được cộng dồn trên toàn tập. Với nhận dạng, nhóm dùng tỉ lệ lỗi ký tự CER dựa trên khoảng cách Levenshtein. Độ đo chính của đồ án là F1 khoản thuốc: một khoản được tính là đúng khi tên có CER không quá 0,1 và số lượng đúng; khoảng tin cậy 95% lấy bằng bootstrap 1.000 lần.

## 11. Kết quả phát hiện và nhận dạng
Ở phát hiện, YOLOv8s đạt F1 0,993, cao hơn hẳn bộ hình thái học với 0,818. Bộ hình thái học bỏ sót 1.364 trên 6.086 vùng và tạo 744 khung thừa, còn YOLOv8s chỉ sót 30 vùng.
Ở nhận dạng, đo trên vùng cắt theo khung chuẩn để tách riêng bước này, tinh chỉnh giảm CER từ 0,086 xuống 0,033, tức khoảng 62%, và tỉ lệ đọc đúng nguyên chuỗi tăng từ 0,816 lên 0,923; tên thuốc gần như không còn lỗi. Tesseract nhị phân không tốt hơn Tesseract xám. Lỗi còn lại tập trung ở dòng chẩn đoán, một phần do chính nhãn sai, ví dụ ảnh in dấu cộng nhưng nhãn ghi dấu phần trăm.

## 12. Kết quả đầu-cuối và so sánh với VLM
Đây là kết quả chính. Thay bộ phát hiện hình thái học bằng YOLOv8s nâng F1 khoản từ 0,9487 lên 0,9865. Khi dùng VietOCR, cả bản gốc lẫn bản tinh chỉnh đều đạt 0,9966 nếu có từ điển, ngang VLM zero-shot. Khác biệt nằm ở vạch nét đứt, tức khi bỏ từ điển: bản gốc giảm còn 0,951, VLM còn 0,988, còn pipeline đề xuất giữ 0,997 và đúng 100% số lượng và ngày kê đơn.
Nhóm cũng nói rõ: pipeline đề xuất đúng hoàn toàn 173 trên 176 đơn, thấp hơn VLM hai đơn, và ba cấu hình tốt nhất có khoảng tin cậy chồng nhau, nên chưa thể khẳng định bên nào chính xác hơn. Ưu điểm rõ của pipeline đề xuất là chạy cục bộ, không gửi ảnh y tế ra ngoài, và không lỗi định dạng JSON như VLM ở 1 trên 176 ảnh.

## 13. Phân tích cắt bỏ và độ bền
Bảng bên trái thay lần lượt từng bước bằng nhãn chuẩn: dùng riêng khung YOLO hoặc riêng chữ VietOCR đều đạt 1,0; lỗi chỉ xuất hiện khi kết hợp cả hai. Nguyên nhân là một vùng "khác" trong mục "Cận lâm sàng" có chữ bắt đầu bằng "1)", nên bị luật xác định lớp đổi thành tên thuốc. Lưu ý rằng đáp án khoản được dẫn xuất từ chính luật ghép, nên các dòng nhãn chuẩn đạt 1,0 là hiển nhiên.
Về tiền xử lý, các bước hình học hoạt động tốt: sai số chỉnh nghiêng trung vị 0,063 độ. Nhưng biểu đồ bên phải cho kết quả chưa như kỳ vọng: bật đủ tiền xử lý làm F1 giảm, ở mức suy giảm nhẹ từ 0,959 xuống 0,769. Giả thuyết của nhóm là khử bóng và CLAHE để lại vệt nền, mà YOLOv8s, vốn chỉ học trên ảnh gốc, nhận nhầm thành chữ: số khung mỗi ảnh tăng từ 34,5 lên 50,5. Vì vậy kết quả chính dùng ảnh gốc.

## 14. Thảo luận và kết luận
Tóm lại, chuỗi xử lý học sâu đạt F1 khoản 0,9966, cao hơn cấu hình dùng Tesseract và cấu hình hình thái học, và ngang VLM zero-shot trong phạm vi khoảng tin cậy. Tinh chỉnh VietOCR giảm 62% lỗi ký tự, còn từ điển bù lỗi OCR cho các cấu hình chưa tinh chỉnh.
Đồ án còn ba hạn chế chính: dữ liệu là ảnh in máy cùng một mẫu đơn; đáp án khoản thuốc dẫn xuất từ chính luật ghép; và tiền xử lý chưa cải thiện độ bền. Hướng tiếp theo là thu thập ảnh chụp thật từ nhiều bệnh viện, huấn luyện lại trên ảnh suy giảm, sửa luật xác định lớp để loại khoản thừa, và bỏ qua vùng "khác" trước khi nhận dạng, vì nhận dạng chiếm 3,3 trong khoảng 3,5 giây xử lý mỗi ảnh.

## 15. Phân công công việc
Về phân công: Hiếu phụ trách tiền xử lý, bộ ảnh suy giảm và tích hợp; Trường huấn luyện YOLOv8s và làm đối chứng hình thái học; Khang tinh chỉnh VietOCR và làm đối chứng Tesseract; Thắng làm luật ghép khoản, từ điển và đối chứng VLM. Mỗi thành viên đóng góp 25% khối lượng công việc.

## 16. Cảm ơn và hỏi đáp
Phần trình bày của nhóm đến đây là kết thúc. Xin cảm ơn thầy cô và các bạn đã lắng nghe; nhóm rất mong nhận được câu hỏi và góp ý.
