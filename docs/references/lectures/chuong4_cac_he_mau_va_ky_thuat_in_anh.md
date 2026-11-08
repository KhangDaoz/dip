> [!NOTE]
> **Lưu ý về hình ảnh minh họa:** File markdown này được trích xuất nội dung văn bản từ tài liệu gốc. Do định dạng văn bản chưa nhúng trực tiếp toàn bộ ảnh minh họa, biểu đồ và sơ đồ thị giác, nếu cần xem chi tiết hình ảnh trực quan, vui lòng mở file PDF gốc tại: [4_XLA4_ Cac he mau va ky thuat in anh.pdf](4_XLA4_ Cac he mau va ky thuat in anh.pdf)
>

---

# Slide 1 — XỬ LÝ ẢNH SỐ: Chương 4. Hệ màu và kỹ thuật in ấn

**XỬ LÝ ẢNH SỐ:**
### Chương 4. Hệ màu và kỹ thuật in ấn

1

---

# Slide 2 — NỘI DUNG

### NỘI DUNG
- Các phương pháp biểu diễn ảnh
- Các định dạng ảnh cơ bản
- Các mô hình màu
- Kỹ thuật in ảnh

2

---

# Slide 3 — CÁC PHƯƠNG PHÁP BIỂU DIỄN ẢNH: BITMAP VÀ VECTOR

### CÁC PHƯƠNG PHÁP BIỂU DIỄN ẢNH:
### BITMAP VÀ VECTOR

3

---

# Slide 4 — Giới thiệu

### Giới thiệu
• Cấu trúc dữ liệu để lưu trữ thông tin ảnh trong bộ nhớ có ảnh hưởng rất lớn đến việc hiển thị, in ấn và xử lý.
• Tiêu chí biểu diễn ảnh:
- Tiết kiệm bộ nhớ
- Giảm thời gian xử lý

4

---

# Slide 5 — Giới thiệu

### Giới thiệu
• Dữ liệu không gian có thể biểu diễn:
- (1) các đặc điểm của thế giới thực có đường biên rời rạc (như đường xá, tòa nhà, hồ, sông, ranh giới hành chính)
- (2) các hiện tượng của thế giới thực có sự thay đổi liên tục (chẳng hạn như lượng mưa và mức độ dinh dưỡng, địa hình).

Nói chung:
• mô hình dữ liệu kiểu vector được sử dụng để biểu diễn loại (1)
• mô hình dữ liệu kiểu raster để biểu diễn loại thứ (2).

5

---

# Slide 6

![Thế giới thực có đường biên rời rạc và hiện tượng liên tục](images/slide-06-figure-01.png)

6

---

# Slide 7 — Ảnh Bitmap (1)

### Ảnh Bitmap (1)
• Trong một ảnh bitmap (hay raster), dữ liệu hình ảnh là một ma trận các điểm màu hay pixel.
• Khi phóng đại trên một màn hình máy tính, hình ảnh bitmap trông giống như khảm.
• Càng nhiều các điểm ảnh, chi tiết hình ảnh sẽ càng sắc nét.

![Chi tiết ảnh bitmap phóng đại](images/slide-07-figure-01.png)

Enlarged bitmap image

| Pixel 1 | Pixel 2 | Pixel 3 |
| :--- | :--- | :--- |
| Red 80% | Red 36% | Red 93% |
| Green 80% | Green 36% | Green 91% |
| Blue 77% | Blue 13% | Blue 0% |

7

---

# Slide 8 — Ảnh Bitmap (2)

### Ảnh Bitmap (2)
• Bởi vì hình ảnh có mức độ chi tiết cao với một loạt các tông và màu sắc nên biểu diễn dưới dạng bitmap là phù hợp nhất.
• Scanner và máy ảnh kỹ thuật số tạo ra hình ảnh bitmap và Photoshop là một chương trình bitmap.
• Các định dạng bitmap bao gồm:
  PSD, TIFF, JPEG, PNG, GIF, PICT, BMP

8

---

# Slide 9 — Vector graphics (1)

### Vector graphics (1)
• Đồ họa vector tạo ra hình ảnh là một tập hợp của đường nét, hình elip, hình tam giác, đa giác...
• Đồ họa vector sử dụng các công thức toán học để mô tả các phác thảo và điền đầy các đối tượng ảnh.
• Đồ họa vector có thể phóng to hoặc thu nhỏ mà không bị mất mát dữ liệu và không làm thay đổi chất lượng ảnh.

![Đồ họa vector phóng đại](images/slide-09-figure-01.png)

Enlarged vector graphic

9

---

# Slide 10 — Vector graphics (2)

### Vector graphics (2)
• CorelDraw, Illustrator, Freehand, AutoCAD & Flash tạo ra vector graphics
• Đồ họa vector là sự lựa chọn tốt nhất khi vẽ các bản vẽ, logo, kiểu chữ... và các hình ảnh không có mô hình phức tạp hay một dải rộng các tông màu thay đổi liên tục.
• Các định dạng file kiểu vector gồm: EPS, AI, SVG, DXF, DWG

10

---

# Slide 11 — Bitmap và Vector

### Bitmap và Vector

![So sánh chi tiết 7x Magnification giữa Vector và Bitmap](images/slide-11-figure-01.png)

7x Magnification
- Vector: `er ice` `nts`
- Ice Cream
- Bitmap: `er ice` `nts`

11

---

# Slide 12 — Bitmap vs Vector

### Bitmap vs Vector
- File dạng vector phù hợp với các mô tả yêu cầu các phép đo chính xác.
- File dạng bitmap phù hợp với các ảnh photo-realistic, loại ảnh có sự thay đổi màu sắc phức tạp.

12

---

# Slide 13 — CÁC ĐỊNH DẠNG ẢNH CƠ BẢN

### CÁC ĐỊNH DẠNG ẢNH CƠ BẢN

13

---

# Slide 14 — GIỚI THIỆU

### GIỚI THIỆU
Các định dạng file ảnh thông dụng nhất, quan trọng nhất đối với camera, in ấn, scanning và internet là:
- JPG - Joint Photographic Experts Group
- TIFF - Tagged Image File Format
- PNG - Portable Graphics Format
- GIF - Graphics Interchange Format

14

---

# Slide 15 — Nén không tổn thất và có tổn thất

### Nén không tổn thất và có tổn thất
- **Nén không tổn thất** nghĩa là không bị mất mát thông tin.
  - Ví dụ: thuật toán nén không tổn thất tìm ra các mẫu lặp đi lặp lại trong file và thay thế các mẫu đó bằng các ký hiệu ngắn hơn và do đó giảm được kích thước file.
- **Nén có tổn thất** chấp nhận sự giảm cấp của ảnh để đạt được kích thước ảnh nhỏ hơn
  - Thuật toán nén có tổn thất có thể lưu giữ thông tin màu ở độ phân giải nhỏ hơn bản thân của bức ảnh và vì vậy mắt không thể nhận ra sự thay đổi của màu sắc ở khoảng cách nhỏ.

15

---

# Slide 16 — Số màu

### Số màu
- Ảnh đơn giản nhất có thể chỉ chứa hai màu (ví dụ trắng và đen) và sẽ chỉ cần 1 bit để biểu diễn cho mỗi pixel.
- Ảnh phức tạp hơn có thể sử dụng 24 bit để biểu diễn cho 1 pixel và vì vậy có khả năng hiển thị $2^{24}$ hay 16 triệu màu.
- Vì mắt khó có thể phân biệt các màu sắc tương tự nhau nên 24 bit/ 1 pixel hay 16 triệu màu thường được gọi là màu thực (truecolor).

16

---

# Slide 17 — TIFF

### TIFF
• TIFF là định dạng ảnh không tổn thất. Thực tế TIFF thường được sử dụng như định dạng ảnh không nén.
• Hầu hết các chương trình đồ họa sử dụng TIFF không nén. Do đó kích thước file khá lớn (đôi khi TIFF sử dụng thuật toán nén không tổn thất LZW nhưng nó không được hỗ trợ rộng rãi).

17

---

# Slide 18 — JPG

### JPG
• Máy ảnh số và các trang web thông thường sử dụng file JPG vì JPG nén có tỉ lệ nén rất cao (kích thước file nhỏ hơn rất nhiều). Để đạt được điều này, JPG sử dụng các thuật toán nén có tổn thất.
• Kích thước file càng nhỏ chất lượng ảnh càng giảm và ngược lại. JPG cho phép lựa chọn giữa chất lượng cao và kích thước file lớn với chất lượng kém và kích thước file nhỏ.

18

---

# Slide 19 — GIF

### GIF

![Hình minh họa GIF Homer Simpson](images/slide-19-figure-01.png)

- GIF tạo một bảng tối đa 256 màu trong 16 triệu màu.
- Nếu một bức ảnh có ít hơn 256 màu, GIF có thể tạo ra bức ảnh một cách chính xác. Nếu bức ảnh có nhiều màu, phần mềm có thể tạo ra ảnh GIF sử dụng các thuật toán để xấp xỉ số màu trong ảnh với bảng màu giới hạn 256 màu.

19

---

# Slide 20 — GIF

### GIF
• GIF đạt được khả năng nén theo 2 cách:
  - Giảm số màu trong bức ảnh nhiều màu và vì thế giảm số bit cần cho mỗi pixel.
  - Nó thay thế các mẫu lặp lại thường xuyên (đặc biệt là một vùng lớn đồng màu) bằng cách viết tắt: "white white white white white " $\rightarrow$ "5 white".
• GIF là không tổn thất đối với các ảnh có ít hơn 256 màu, còn đối với các bức ảnh màu chân thực nó có thể mất 99,998% màu.

20

---

# Slide 21 — PNG

### PNG
• PNG cũng là một định dạng không tổn thất.
• Nó tìm kiếm các mẫu trong ảnh để có thể sử dụng để nén file.
• Quá trình nén là thuận nghịch, do đó có thể khôi phục bức ảnh một cách chính xác.
• So với TIFF, PNG có dung lượng nhỏ hơn một chút nhưng thời gian đọc và ghi chậm hơn.

21

---

# Slide 22 — Kết luận

### Kết luận
• GIF và JPG là các định dạng được sử dụng cho hầu hết các ảnh web.
• PNG được hỗ trợ bởi hầu hết các trình duyệt thế hệ mới nhất.
• TIFF không được sự hỗ trợ rộng rãi của các trình duyệt web và vì vậy ít được sử dụng cho web.
• PNG có thể làm được mọi điều như GIF và thậm chí tốt hơn, vì vậy trong tương lai PNG có thể thay thế cho GIF.
• PNG không thể thay thế cho JPG vì JPG có khả năng nén ảnh tốt hơn nhiều.

22

---

# Slide 23 — CÁC CÁCH SỬ DỤNG

### CÁC CÁCH SỬ DỤNG

| | Photographic Images | Graphics, including Logos or Line art |
| :--- | :--- | :--- |
| **Đặc điểm** | Các bức ảnh có tông màu liên tục, 24 bit màu hoặc 8bit đối với ảnh xám, không có chữ viết, ít đường nét và biên. | Đồ họa thường là các màu đơn với số lượng màu ít, tối đa là 256 màu, thường có chữ viết hoặc đường nét, biên sắc nét. |
| **Chất lượng tốt nhất** | TIFF hoặc PNG (nén không tổn thất và không có nhiễu JPG) | TIFF hoặc PNG (nén không tổn thất và không có nhiễu JPG) |
| **Kích thước file nhỏ nhất** | JPG với hệ số chất lượng phù hợp. | TIFF LZW or GIF or PNG (đồ họa/logos không có gradient thường sử dụng 2-16 màu để đạt được kích thước file nhỏ nhất) |
| **Độ tương thích tối đa (PC, Mac, Unix)** | TIFF or JPG | TIFF or GIF |

23

---

# Slide 24 — CÁC MÔ HÌNH MÀU

### CÁC MÔ HÌNH MÀU

24

---

# Slide 25 — Cảm nhận màu sắc

### Cảm nhận màu sắc
- Mắt người cảm nhận màu sắc dựa vào tế bào hình nón (cones) và tế bào hình que (rods) trên hoàng điểm.
- Tế bào hình que chủ yếu cảm nhận về độ chói (luminance).
- Tế bào hình nón chịu trách nhiệm chính về màu sắc. Có 3 loại tế bào hình nón cảm biến chính tương ứng với màu đỏ (red), xanh lá cây (green) và xanh da trời (blue) với tỷ lệ 65%, 33% và 2% (nhưng tế bào nón xanh da trời là loại nhạy cảm nhất).

25

---

# Slide 26

Do đặc tính hấp thụ của mắt người, màu được coi như sự kết hợp của các màu sắc cơ bản là Red (R), Green (G) và Blue (B).

Ba màu cơ bản này có thể cộng với nhau để tạo ra các màu thứ cấp của ánh sáng là:
- Magenta (đỏ thẫm) = red + blue
- $Cyan(xanh\ nhạt\ hay\ lục\ lam) = green + blue$
- Yellow (vàng) = red + green

![Màu sắc và các giá trị RGB](images/slide-26-figure-01.png)

| R | G | B |
| :---: | :---: | :---: |
| 61 | 183 | 228 |

| R | G | B |
| :---: | :---: | :---: |
| 255 | 136 | 73 |

| R | G | B |
| :---: | :---: | :---: |
| 105 | 190 | 40 |

RGB values of the same three colors

26

---

# Slide 27 — MÔ HÌNH MÀU (1)

### MÔ HÌNH MÀU (1)
- Một mô hình màu (hay hệ màu, không gian màu) là một hệ thống có quy tắc để tạo ra tất cả các màu sắc từ một nhóm nhỏ các màu cơ bản.
- Các mô hình màu hiện nay tập trung hướng vào phần cứng (ví dụ phần cứng có chức năng thu nhận và hiển thị hình ảnh) hoặc các ứng dụng ở đó mục đích là thao tác trên màu.

27

---

# Slide 28 — MÔ HÌNH MÀU (2)

### MÔ HÌNH MÀU (2)
- Mô hình theo hướng phần cứng được sử dụng nhiều nhất trong thực tế là mô hình RGB (red, green, blue) cho các màn hình màu và camera video màu.
- CMY (cyan, magenta, yellow) và CMYK (Cyan, magenta, yellow, black) cho in màu.
- HSV (Hue, saturation, value) và HSL (hue, saturation, lightness) gần tương ứng với cách con người mô tả và biên dịch màu sắc nên phù hợp với các hệ thống phần mềm hỗ trợ biên soạn ảnh.

28

---

# Slide 29 — RGB

### RGB

![Thí nghiệm phối màu cộng RGB bằng vỏ CD](images/slide-29-figure-01.png)

Additív színkeverés CD-tokokkal  
Additive color mixing with CD covers

---

# Slide 30 — RGB

### RGB
• Trong hệ màu RGB, các màu sắc đều được tạo ra từ 3 màu cơ bản là red, green và blue.
• 3 màu RGB nằm ở 3 góc, 3 màu thứ cấp CMY nằm ở 3 góc đối diện. Màu đen ở gốc tọa độ và màu trắng ở góc xa gốc nhất.
• Các màu khác nhau trong mô hình này là các điểm nằm trên hoặc trong hình lập phương và được xác định bởi vector tính từ gốc tọa độ.

![Hệ màu cộng RGB và khối lập phương màu](images/slide-30-figure-01.png)

- Magenta
- Red
- Black
- Blue
- Cyan
- Gray Scale
- White
- Green
- Yellow

30

---

# Slide 31 — Đặc điểm của hệ màu RGB

### Đặc điểm của hệ màu RGB
• Mô hình màu cộng
• Dùng cho hiển thị máy tính, TV, camera số
• Sử dụng ánh sáng để hiển thị màu.
• Màu sắc là kết quả của ánh sáng truyền qua.
• $Red + Green + Blue = White.$
• Bởi vì hệ màu cộng hiển thị màu như là kết quả của ánh sáng truyền qua (cộng vào) nên khi không có ánh sáng sẽ được coi là màu đen.
• Các giá trị màu là một vector gồm 3 thành phần R, G, B có giá trị nằm trong khoảng từ 0 đến 1.

31

---

# Slide 32 — CMY

### CMY
• Hệ màu trừ
• Sử dụng trong in ấn.
• Sử dụng mực để hiển thị màu
• Màu là kết quả của ánh sáng bị phản xạ (không bị hấp thụ)
• Cyan + Magenta + Yellow = Black
• Hệ màu trừ hiển thị màu như là kết quả của ánh sáng bị hấp thụ (bị trừ) bởi mực in nên khi thêm vào càng nhiều mực in thì ánh sáng phản xạ càng ít. Khi không có mực in thì ánh sáng bị phản xạ (từ bề mặt trắng) sẽ được coi là màu trắng.

![Sơ đồ và khối lập phương hệ màu CMY](images/slide-32-figure-01.png)

- Cyan, Blue, Green, Magenta, Red, Yellow, Black
- Red, White, Magenta, Blue, Gray Scale, Black, Cyan, Yellow, Green

32

---

# Slide 33 — CMY

### CMY

![Sơ đồ hệ màu trừ CMY và ánh sáng qua kính lọc](images/slide-33-figure-01.png)

- Cyan
- Blue
- Green
- Magenta
- Red
- Yellow

33

---

# Slide 34 — CMY

### CMY
• Khi một bề mặt phủ màu cyan được rọi bởi ánh sáng trắng, sẽ không có ánh sáng màu đỏ phản chiếu từ bề mặt. Đó là bởi vì cyan đã triệt tiêu phần màu đỏ phản xạ khi có tia sáng trắng mà bản chất là tổng của 3 màu RGB.
• $Cyan = 1 - red = blue + green$

![So sánh vòng tròn giao thoa CMY và RGB](images/slide-34-figure-01.png)

- Cyan, Blue, Green, Magenta, Red, Yellow
- Red, Magenta, Yellow, Blue, Cyan, Green

34

---

# Slide 35 — RGB và CMY

### RGB và CMY
- Sự khác biệt nằm ở mục đích sử dụng.
- Với các ứng dụng liên quan đến trình chiếu trên các màn hình thì lựa chọn RGB. RGB sẽ làm việc tốt với các thiết bị phát quang sử dụng ánh sáng trắng làm cơ sở.
- Để phục vụ cho in ấn thì CMY (CMYK) là lựa chọn tối ưu.

35

---

# Slide 36 — RGB và CMY

### RGB và CMY
• Chuyển đổi giữa hệ màu RGB và CMY:

$$\begin{bmatrix} C \\ M \\ Y \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} - \begin{bmatrix} R \\ G \\ B \end{bmatrix}$$

$$\begin{bmatrix} R \\ G \\ B \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} - \begin{bmatrix} C \\ M \\ Y \end{bmatrix}$$

![So sánh hiển thị dải màu và phong cảnh giữa RGB và CMYK](images/slide-36-figure-01.png)

- RGB en monitor
- CMYK en monitor
- CMYK impreso
- RGB
- CMYK

---

# Slide 37 — HSV (Hue, Saturation, Value)

### HSV (Hue, Saturation, Value)
• Ngoài RGB, CMY còn có thể mô tả màu bằng các thuộc tính khác:
  - Hue (độ màu)
  - Saturation (độ bão hòa)
  - Value (giá trị độ sáng)

37

---

# Slide 38 — HUE (Độ màu)

### HUE (Độ màu)

![Lăng kính tán sắc ánh sáng tạo dải màu](images/slide-38-figure-01.png)

- **Định nghĩa:** một màu cụ thể trong dải màu của mô hình màu, mô tả độ chân thực của màu sắc.
- Được xác định bởi độ dài bước sóng xác định.
- Là cái mà hầu hết chúng ta nghĩ đến khi nói “màu sắc".
- ROY G. BIV =
  - Red
  - Orange
  - Yellow
  - Green
  - Blue
  - Indigo
  - Violet

38

---

# Slide 39 — Saturation (Độ bão hòa)

### Saturation (Độ bão hòa)

![Mô hình hình trụ biểu diễn Hue, Saturation, Value](images/slide-39-figure-01.png)

- **Định nghĩa:**
  - “là lượng ánh sáng trắng được trộn với hue (độ màu)”
  - đại diện cho lượng màu; mức độ trội của độ màu.
- Độ bão hòa là thước đo độ chân thực bị làm loãng đi hay bị giảm bớt đi bởi sự tác động của ánh sáng trắng.
- Hue
- Saturation
- Value

39

---

# Slide 40 — Value (Độ sáng)

### Value (Độ sáng)
- **Định nghĩa:** biểu diễn mức độ sáng tối của màu.
- Cũng thường gọi là độ sáng "brightness" hay cường độ “intensity".
- Đây là khái niệm chủ quan khó có thể đo đạc được, dùng để mô tả cường độ màu sắc cảm nhận bởi mắt người.

![So sánh không gian màu RGB và HSV](images/slide-40-figure-01.png)

Federal University of Rio de Janeiro (UFRJ) Campus Cidade Universitária Rio de Janeiro Fundão Island: COPPE/PESC/Laboratory of Computer Graphics March/2015  
RGB VS HSV (Color Spaces - a common mistake...)  
RGB Color Space  
HSV Color Space  
Blue, Red, Green  
A cone is commonly adopted but cylindrical representation is more mathematically accurate.  
Hue, Saturation, Value  
Michel Alves - Five Minute Speech (FMS): Overview of Activities Developed in Disciplines and Guided Studies Graduate Program in Systems Engineering and Computing  

40

---

# Slide 41 — HSV

### HSV
- Mô hình này tách biệt giá trị cường độ ra khỏi các thông tin chứa màu sắc (độ màu và độ bão hòa) trong ảnh màu.
- HSV trở thành công cụ lý tưởng cho xử lý ảnh dựa trên sự mô tả màu sắc một cách tự nhiên và trực quan nhất với con người.
- HSV thường được sử dụng trong các phần mềm biên soạn ảnh, hoặc các công cụ trình diễn dữ liệu như bản đồ hoặc ảnh y tế.

41

---

# Slide 42 — KỸ THUẬT IN ẢNH

### KỸ THUẬT IN ẢNH
### Kỹ thuật nửa cường độ số (Digital Halftoning)

![Ví dụ minh họa Digital Halftoning chân dung](images/slide-42-figure-01.png)

42

---

# Slide 43 — KỸ THUẬT IN ẢNH

![Video minh họa: What is a halftone?](images/slide-43-figure-01.png)

What is a halftone?  
0:55 / 2:04  
0:54  
CC  

### KỸ THUẬT IN ẢNH
### Kỹ thuật nửa cường độ số (Digital Halftoning)

---

# Slide 44 — Digital Halftoning: Giới thiệu

### Digital Halftoning: Giới thiệu
• Kỹ thuật nửa cường độ là một quá trình mô phỏng các sắc thái của màu xám bằng cách thay đổi kích thước của chấm đen nhỏ sắp xếp theo một mô hình chung.
• Kỹ thuật này được sử dụng trong máy in, cũng như các ngành công nghiệp xuất bản.
• Điều này thực hiện được do sự tổng hợp không gian được thực hiện bởi đôi mắt. Đôi mắt sẽ pha trộn các chi tiết sắc nét và ghi lại cường độ tổng thể.
• Cũng được áp dụng cho ảnh màu.

44

---

# Slide 45 — Các tham số của halftoning

### Các tham số của halftoning
- **Hình dạng của thành phần lưới:**
  Hình vuông, hình tròn, hình elip, dấu thập, dấu chéo, tam giác, đường thẳng.
- **Mật độ đường thẳng trên lưới**, được xác định bằng lines per inch (lpi).
  - Báo giấy: 85 lpi, tạp chí: 200 – 300 lpi
- **Góc màn hình:** góc phổ biến nhất là $45^\circ$

![Góc xoay tram halftoning 45 độ và 90 độ](images/slide-45-figure-01.png)

$45^\circ$  
$90^\circ$  

45

---

# Slide 46 — Ví dụ halftoning

### Ví dụ halftoning
• Trường hợp lưới $6 \times 6$.
• Có thể hiển thị 37 mức sáng.

![Mô phỏng 37 mức sáng bằng ma trận 6x6](images/slide-46-figure-01.png)

46

---

# Slide 47 — Ví dụ halftoning

### Ví dụ halftoning
• Mức độ bao phủ của mực trong vùng sẽ xác định mức xám.

![Kích thước chấm halftone tương ứng mức xám](images/slide-47-figure-01.png)

$$\text{gray} = 0.7 \quad \text{gray} = 0.5 \quad \text{gray} = 0.3$$

Identical?

47

---

# Slide 48 — Ví dụ

### Ví dụ
• Dưới đây là các ví dụ về kỹ thuật nửa cường độ đối với ảnh màu với sự phân tách CMYK.

![Phối hợp tram các màu CMYK với góc xoay khác nhau](images/slide-48-figure-01.png)

$75^\circ + 105^\circ / 90^\circ / 75^\circ + 45^\circ + 15^\circ = \dots \approx \dots$  
$\dots + 15^\circ + 0^\circ = \dots \approx \dots$  
$75^\circ + 105^\circ / 90^\circ + 45^\circ + 45^\circ + 15^\circ / 165^\circ = \dots \approx \dots$  
$0^\circ$

---

# Slide 49 — Các kỹ thuật nửa cường độ số

### Các kỹ thuật nửa cường độ số
- Thresholding (phân ngưỡng)
- Patterning (chọn mẫu)
- Dithering (phối màu)
- Error diffusion (Khuếch tán lỗi)

49

---

# Slide 50 — Kỹ thuật phân ngưỡng (Thresholding)

### Kỹ thuật phân ngưỡng (Thresholding)
• Kỹ thuật đơn giản nhất để cải thiện độ phân giải hình ảnh là sử dụng một mẫu ngưỡng.
• Ta tạo ra mẫu này và so sánh với ảnh gốc. Nếu pixel trong ảnh gốc lớn hơn pixel tương ứng trong mẫu ngưỡng thì pixel này được thay bằng 1, ngược lại thay bằng 0.

![So sánh ảnh mức xám và kết quả phân ngưỡng](images/slide-50-figure-01.png)

- GrayScale
- Threshold

50

---

# Slide 51 — Kỹ thuật chọn theo mẫu (Patterning)

### Kỹ thuật chọn theo mẫu (Patterning)
• Được sử dụng khi thiết bị đầu ra có độ phân giải không gian cao hơn ảnh gốc.
• Độ phân giải hình ảnh của các ảnh được tạo ra bằng máy tính có thể tăng lên nhờ sử dụng kỹ thuật này.
• Trong kỹ thuật này, thực hiện chọn một nhóm các mẫu (patterns) để tạo ra nhiều mức cường độ xám hơn.

51

---

# Slide 52 — Patterning

### Patterning
• Với mỗi cell gồm 4 pixel, cách sử dụng này tạo ra 5 mức xám:  
  $0, 1, 2, 3, 4$
• Thông thường sử dụng ma trận chọn mẫu hồi quy của Rylander:
  - Khối $n \times n$ tạo ra $n^2 + 1$ mức xám.
  - Yêu cầu lượng tử hóa pixel đầu vào.
  - Ví dụ khối $4 \times 4$ tạo ra 17 mức cường độ. Do đó phải chia pixel đầu vào cho 15 ($15 \times 17 = 255$). Khi đó sẽ tạo ra các pixel có giá trị từ 0 đến 17.

52

---

# Slide 53 — Ma trận chọn mẫu hồi quy của Rylander

### Ma trận chọn mẫu hồi quy của Rylander

![17 mẫu ma trận 4x4 từ 0 đến 16](images/slide-53-figure-01.png)

0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16

53

---

# Slide 54 — Quá trình chọn mẫu

### Quá trình chọn mẫu

![Sơ đồ khối quá trình chọn mẫu sử dụng bảng tra LUT](images/slide-54-figure-01.png)

- Input Image
- LUT Address Generation
- Output Image
- Pattern LUT
- LUT: Look Up Table

**input Image:**
| 7 | 7 | 5 | 4 | 2 |
| :-: | :-: | :-: | :-: | :-: |
| 7 | 6 | 4 | 3 | 0 |
| 7 | 6 | 4 | 3 | 1 |
| 6 | 6 | 4 | 2 | 0 |
| 5 | 5 | 3 | 1 | 0 |

**LUT:**
| 0 | 0 |
| :-: | :-: |
| 1 | 0 |
| 2 | 1 |
| 3 | 1 |
| 4 | 2 |
| 5 | 3 |
| 6 | 4 |
| 7 | 5 |

**output image:**
| 5 | 5 | 3 | 2 | 1 |
| :-: | :-: | :-: | :-: | :-: |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

54

---

# Slide 55 — Nhận xét

### Nhận xét
• Việc chọn kích thước của nhóm như vậy sẽ làm giảm độ mịn của ảnh.
• Vì vậy kỹ thuật này chỉ áp dụng trong trường hợp mà độ phân giải của thiết bị ra lớn hơn độ phân giải của ảnh nguồn.
• Ví dụ thiết bị ra có độ phân giải $256 \times 256$ và sử dụng ma trận $2 \times 2$, ta phải sử dụng ảnh gốc $128 \times 128$.

![So sánh ảnh gốc và ảnh sau khi áp dụng patterning](images/slide-55-figure-01.png)

55

---

# Slide 56 — Kỹ thuật phối màu (Dithering)

### Kỹ thuật phối màu (Dithering)
- Không giống như pattern, dithering tạo ra hình ảnh đầu ra với cùng số lượng điểm ảnh trong hình ảnh đầu vào.
- Dithering có thể được coi là phân ngưỡng ảnh với một ma trận phối màu.
- Kỹ thuật này tạo ra ảnh đa cấp sáng khi độ phân giải nguồn và đích là như nhau.
- Để tạo ảnh, mỗi phần tử của ảnh gốc sẽ được so sánh với phần tử tương ứng của ma trận Dither. Nếu lớn hơn, phần tử ở đầu ra sẽ sáng (trong in ấn sẽ là pixel tối với số điểm in nhiều) và ngược lại.

56

---

# Slide 57 — Ví dụ

### Ví dụ

![Ví dụ phối màu Dithering với ma trận lặp](images/slide-57-figure-01.png)

**input image:**
| 12 | 51 | 34 | 121 |
| :-: | :-: | :-: | :-: |
| 78 | 254 | 10 | 97 |
| 45 | 113 | 110 | 16 |
| 90 | 200 | 206 | 34 |

**repeated dither matrix:**
| 0 | 60 | 0 | 60 |
| :-: | :-: | :-: | :-: |
| 45 | 110 | 45 | 110 |
| 0 | 60 | 0 | 60 |
| 45 | 110 | 45 | 110 |

**output image:**
(Bi-level Output Image Segment)

**Input Image Segment:**
| 12 | 51 | 14 | 31 | 16 | 50 | 60 | 70 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 30 | 55 | 23 | 100 | 13 | 99 | 79 | 83 |
| 77 | 65 | 199 | 203 | 202 | 200 | 85 | 99 |
| 66 | 58 | 43 | 11 | 15 | 65 | 89 | 91 |
| 57 | 81 | 64 | 24 | 98 | 56 | 80 | 19 |
| 41 | 98 | 31 | 30 | 18 | 16 | 9 | 0 |
| 78 | 43 | 51 | 61 | 45 | 53 | 87 | 15 |
| 64 | 53 | 41 | 43 | 61 | 13 | 71 | 115 |

**Tiled Dither Matrices:**
| 128 | 32 | 160 | 0 | 128 | 32 | 160 | 0 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 192 | 64 | 224 | 96 | 192 | 64 | 224 | 96 |
| 48 | 176 | 16 | 144 | 48 | 176 | 16 | 144 |
| 240 | 112 | 208 | 80 | 240 | 112 | 208 | 80 |
| 128 | 32 | 160 | 0 | 128 | 32 | 160 | 0 |
| 192 | 64 | 224 | 96 | 192 | 64 | 224 | 96 |
| 48 | 176 | 16 | 144 | 48 | 176 | 16 | 144 |
| 240 | 112 | 208 | 80 | 240 | 112 | 208 | 80 |

**Bi-level Output Image Segment**

---

# Slide 58 — Khuếch tán lỗi (error diffusion)

### Khuếch tán lỗi (error diffusion)
- Khuếch tán lỗi cho phép giảm thiểu mức độ mất chi tiết của ảnh khi tách ngưỡng bằng cách phân tán lỗi (do sai số xấp xỉ) ra các điểm ảnh xung quanh (bên phải và bên dưới pixel hiện thời).
- Bằng cách này tổng giá trị điểm ảnh của một vùng nhỏ được giữ tương đối gần với giá trị trên ảnh gốc.
- **Lưu ý:** Khuếch tán lỗi là toán tử láng giềng trong khi chọn mẫu và phối màu là toán tử điểm (pixel).

58

---

# Slide 59 — Mô tả quá trình khuếch tán lỗi

### Mô tả quá trình khuếch tán lỗi

![Sơ đồ hồi tiếp mô tả quá trình khuếch tán lỗi](images/slide-59-figure-01.png)

- $I(x,y) \rightarrow (+) \rightarrow u(x,y) \rightarrow \text{Quantize (threshold)} \rightarrow b(x,y)$
- Error calculation: $e(x,y) \rightarrow \text{Distribute error} \rightarrow (+)$
- $0 \le I(x,y) \le 1 \rightarrow \text{Threshold} = 0.5$; $\text{Black} = 0$; $\text{White} = 1$
- $0 \le I(x,y) \le 255 \rightarrow \text{Threshold} = 127$; $\text{Black} = 0$; $\text{White} = 255$

59

---

# Slide 60 — Khuếch tán lỗi một chiều (1D)

### Khuếch tán lỗi một chiều (1D)
- Ảnh được duyệt từ trái qua phải, từ trên xuống dưới.
- Tại mỗi điểm ảnh, giá trị điểm ảnh được tách theo ngưỡng có sẵn.
- Phần dư do sai số phân ngưỡng được chuyển sang điểm ảnh tiếp theo trên cùng dòng.
- Các bước được lặp lại cho đến hết dòng, phần dư của điểm ảnh cuối cùng sẽ được loại bỏ.

60

---

# Slide 61 — Ví dụ

### Ví dụ
$$I = \begin{bmatrix} 0.7 & 0.7 & 0.3 & 0.5 & 0.1 & 0.1 & 0.1 \end{bmatrix}$$

- $I(1) = u(1) = 0.7 \rightarrow \text{threshold at } 0.5 \rightarrow b(1) = 1$
- $e(1) = b(1) - u(1) = 0.3$
- Since pixel $I(1)$ was over represented, compensate by subtracting error from next pixel $I(2)$
- $u(2) = I(2) - e(1) = 0.4$
- $u(2) = 0.4 \rightarrow \text{threshold at } 0.5 \rightarrow b(2) = 0$
- $e(2) = b(2) - u(2) = -0.4$
- $u(3) = I(3) - e(2) = 0.7$
- and so on....

$$\begin{matrix}
b = & [1 & 0 & 1 & 0 & 0 & 0 & 1]
\end{matrix}$$

61

---

# Slide 62 — Bài tập

### Bài tập
- Thực hiện khuếch tán lỗi một chiều với ảnh $I$ sử dụng ngưỡng 127, được biết ảnh này là ảnh 256 mức xám với mức nhỏ nhất là 0 và lớn nhất là 255.

$$I = \begin{bmatrix} 1 & 13 & 156 & 22 & 45 \\ 133 & 13 & 12 & 12 & 232 \\ 12 & 222 & 127 & 32 & 21 \end{bmatrix}$$

62

---

# Slide 63 — Khuếch tán lỗi hai chiều (2D)

### Khuếch tán lỗi hai chiều (2D)
Các bước được thực hiện như khuếch tán lỗi một chiều, tuy nhiên lỗi do sai số xấp xỉ sẽ được phân tán ra các điểm xung quanh (tối thiểu là 4 pixel) theo tỷ lệ xác định bởi ma trận khuếch tán.

Một số ma trận khuếch tán phổ biến:

**Floyd & Steinberg:**
$$\frac{1}{16} \times \begin{bmatrix} & X & 7 \\ 3 & 5 & 1 \end{bmatrix}$$

**Jarvis, Judice & Ninke:**
$$\frac{1}{48} \times \begin{bmatrix} & & X & 7 & 5 \\ 3 & 5 & 7 & 5 & 3 \\ 1 & 3 & 5 & 3 & 1 \end{bmatrix}$$

**Stucki:**
$$\frac{1}{42} \times \begin{bmatrix} & & X & 8 & 4 \\ 2 & 4 & 8 & 4 & 2 \\ 1 & 2 & 4 & 2 & 1 \end{bmatrix}$$

*X: Pixel being processed*

63

---

# Slide 64 — Floyd & Steinberg

### Floyd & Steinberg

![Sơ đồ phân tán lỗi 2D theo Floyd & Steinberg](images/slide-64-figure-01.png)

Current Pixel Image:
- Phải: $+ \text{ERROR} \times \frac{7}{16}$
- Dưới - trái: $+ \text{ERROR} \times \frac{3}{16}$
- Dưới: $+ \text{ERROR} \times \frac{5}{16}$
- Dưới - phải: $+ \text{ERROR} \times \frac{1}{16}$

```text
FOR Y = 0 TO HEIGHT
  FOR X = 0 TO WIDTH
    IF INPUT_IMAGE[X][Y] < THRESHOLD
      OUTPUT_IMAGE[X][Y] = 0
      ERROR = INPUT_IMAGE[X][Y]
    ELSE
      OUTPUT_IMAGE[X][Y] = 1
      ERROR = INPUT_IMAGE[X][Y] - MAX_PIXEL
    INPUT_IMAGE[X+1][Y]     = INPUT_IMAGE[X+1][Y]     + ERROR * 7/16
    INPUT_IMAGE[X-1][Y+1]   = INPUT_IMAGE[X-1][Y+1]   + ERROR * 3/16
    INPUT_IMAGE[X][Y+1]     = INPUT_IMAGE[X][Y+1]     + ERROR * 5/16
    INPUT_IMAGE[X+1][Y+1]   = INPUT_IMAGE[X+1][Y+1]   + ERROR * 1/16
```

64

---

# Slide 65 — NHẬN XÉT

### NHẬN XÉT
- Các kỹ thuật phân ngưỡng, chọn mẫu hay phối màu đều là các toán tử pixel nên tính toán nhanh nhưng cho kết quả chưa được tốt.
- Kỹ thuật khuếch tán lỗi cho kết quả tốt nhất nhưng là toán tử láng giềng nên cần các tính toán chuyên sâu hơn.

65

---

# Slide 66 — NỘI DUNG ĐÃ HỌC

### NỘI DUNG ĐÃ HỌC
- Các phương pháp biểu diễn ảnh
- Các định dạng ảnh cơ bản
- Các mô hình màu
- Kỹ thuật in ảnh.

66