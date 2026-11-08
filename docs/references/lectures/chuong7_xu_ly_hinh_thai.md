> [!NOTE]
> **Lưu ý về hình ảnh minh họa:** File markdown này được trích xuất nội dung văn bản từ tài liệu gốc. Do định dạng văn bản chưa nhúng trực tiếp toàn bộ ảnh minh họa, biểu đồ và sơ đồ thị giác, nếu cần xem chi tiết hình ảnh trực quan, vui lòng mở file PDF gốc tại: [7_XLA7_Xu ly hinh thai.pdf](7_XLA7_Xu ly hinh thai.pdf)
>

---

Slide 1 — XỬ LÝ ẢNH SỐ Chương 7. Xử lý ảnh hình thái

# XỬ LÝ ẢNH SỐ

## Chương 7.

## Xử lý ảnh hình thái

Slide 2 — Nội dung

2 of 50

# Nội dung

Thuật toán xử lý ảnh hình thái được sử dụng để loại bỏ những khiếm khuyết trong bức ảnh sau khi được phân vùng và bổ sung thêm các thông tin bị mất cho bức ảnh.

### Các nội dung chính

* Khái niệm xử lý ảnh hình thái (morphology)
* Các phép toán xử lý hình thái đơn giản
* Các phép toán xử lý hình thái ghép
* Các thuật toán xử lý hình thái

Slide 3 — Khái niệm xử lý ảnh hình thái

3 of 50

# Khái niệm xử lý ảnh hình thái

* Xử lý ảnh hình thái (morphology) mô tả một loạt các kỹ thuật xử lý trên hình dạng của các đối tượng trên bức ảnh.
* Các phép toán hình thái thường được sử dụng để loại bỏ những khiếm khuyết của đối tượng trong quá trình xử lý phân vùng ảnh.

Slide 4 — Ví dụ

4 of 50

# Ví dụ

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

* Image after segmentation
* Image after segmentation and morphological processing

Slide 5 — Các phần tử có cấu trúc (Structuring Elements)

5 of 50

# Các phần tử có cấu trúc

## (Structuring Elements)

* Các SE - phần tử cấu trúc - là các mảng (ma trận) có cấu trúc được tạo bởi 2 mức xám 0 và 1
* Thông thường SEs có dạng đối xứng với trọng tâm ở giữa.

$$
\begin{bmatrix}
0 & 0 & 1 & 0 & 0 \\
0 & 1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1 & 1 \\
0 & 1 & 1 & 1 & 0 \\
0 & 0 & 1 & 0 & 0
\end{bmatrix}
\qquad
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
\qquad
\begin{bmatrix}
0 & 1 & 0 \\
1 & 1 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

Slide 6 — Hits & Fits

6 of 50

# Hits & Fits

* **Structuring Element**
* **Fit:** Fit xảy ra khi tất cả pixel 1 của SE trùng với các pixel 1 của bức ảnh
* **Hit:** Hit xảy ra khi tồn tại pixel 1 của SE trùng với pixel 1 của ảnh

**Tất cả các phép toán xử lý hình thái đều được dựa trên 2 hiện tượng trên!**

* A: fit
* B: hit
* C: none

Slide 7 — Fitting & Hitting

7 of 50

# Fitting & Hitting

$$
S_1 = \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
\qquad
S_2 = \begin{bmatrix}
0 & 1 & 0 \\
1 & 1 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

| | | A | B | C |
| :--- | :--- | :--- | :--- | :--- |
| **fit** | $S_1$ | yes | no | no |
| | $S_2$ | yes | yes | no |
| **hit** | $S_1$ | yes | yes | yes |
| | $S_2$ | yes | yes | no |

Slide 8 — Các phép xử lý hình thái cơ bản

8 of 50

# Các phép xử lý hình thái cơ bản

* Phép xử lý ảnh hình thái về cơ bản tương tự như phép lọc không gian.
* SE được dịch chuyển qua tất cả các pixel của ảnh gốc để tạo ra bức ảnh mới.
* Giá trị của các pixel mới phụ thuộc vào phép toán xử lý hình thái
* Hai phép toán xử lý hình thái cơ bản là: **co (erosion, shrink, reduce)** và **giãn (dilation, grow, expand)**

Slide 9 — Erosion- Phép co

9 of 50

# Erosion- Phép co

* Phép co bức ảnh $f$ bởi SE $s$ được biểu diễn bởi phép toán: $f \ominus s$
* **Mô tả phép toán:**
  Giả sử SE $s$ đang ở vị trí $(x, y)$. Pixel mới sau khi thực hiện phép toán có giá trị như sau:

$$
g(x,y) = \begin{cases}
1 & \text{if } s \text{ fits } f \\
0 & \text{otherwise}
\end{cases}
$$

Slide 10 — Phép co: Ví dụ 1

10 of 50

# Phép co: Ví dụ 1

* Original Image
* Processed Image With Eroded Pixels
* Structuring Element

Slide 11 — Phép co: Ví dụ 1

11 of 50

# Phép co: Ví dụ 1

* Original Image
* Processed Image
* Structuring Element

Slide 12 — Phép co: Ví dụ 2

12 of 50

# Phép co: Ví dụ 2

* Original image
* Erosion by $3*3$ square structuring element
* Erosion by $5*5$ square structuring element

Slide 13 — Phép co: Ví dụ 3

13 of 50

# Phép co: Ví dụ 3

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

* Original image
* After erosion with a disc of radius 10
* After erosion with a disc of radius 5
* After erosion with a disc of radius 20

Slide 14 — Ứng dụng của phép co hình

14 of 50

# Ứng dụng của phép co hình

* Phép co hình làm tách rời các đối tượng gần nhau
* Phép co hình cắt bớt các phần dư thừa của đối tượng
* Chú ý: Phép co làm thu nhỏ kích thước của các đối tượng

Slide 15 — Bài tập 1

15 of 50

# Bài tập 1

Cho tập A và phần tử cấu trúc B như hình vẽ. Tìm kết quả của phép co A với cấu trúc B.

* A
* B

Slide 16 — Bài tập 1

16 of 50

# Bài tập 1

$$
A \ominus B
$$

* $B$

Slide 17 — Bài tập 2

17 of 50

# Bài tập 2

Cho tập A như hình vẽ (a). Tìm các phần tử cấu trúc để sau phép co, tập A có dạng như hình (b) và (c).

* $2d$
* $d$
* $6d$
* (a)
* (b)
* (c)

Slide 18 — Bài tập 2

18 of 50

# Bài tập 2

* $2d$
* $d$
* $6d$
* (a)
* (b)
* (c)

Slide 19 — Phép giãn hình- Dilation

19 of 50

# Phép giãn hình- Dilation

* Phép giãn hình $f$ bởi phần tử cấu trúc $s$ được cho bởi công thức $f \oplus s$
* **Mô tả phép toán:**
  Nếu phần tử cấu trúc $s$ tại pixel có tọa độ $(x, y)$ của ảnh gốc thì sau phép giãn hình, pixel đó có giá trị:

$$
g(x,y) = \begin{cases}
1 & \text{if } s \text{ hits } f \\
0 & \text{otherwise}
\end{cases}
$$

Slide 20 — Phép giãn hình: Ví dụ 1

20 of 50

# Phép giãn hình: Ví dụ 1

* Original Image
* Processed Image
* Structuring Element

Slide 21 — Phép giãn hình: Ví dụ 1

21 of 50

# Phép giãn hình: Ví dụ 1

* Original Image
* Processed Image With Dilated Pixels
* Structuring Element

Slide 22 — Phép giãn hình: Ví dụ 2

22 of 50

# Phép giãn hình: Ví dụ 2

* Original image
* Dilation by $3*3$ square structuring element
* Dilation by $5*5$ square structuring element

Slide 23 — Phép giãn hình: Ví dụ 3

23 of 50

# Phép giãn hình: Ví dụ 3

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

### Original image
Historically, certain computer programs were written using only two digits rather than four to define the applicable year. Accordingly, the company's software may recognize a date using "00" as 1900 rather than the year 2000.

### After dilation
Historically, certain computer programs were written using only two digits rather than four to define the applicable year. Accordingly, the company's software may recognize a date using "00" as 1900 rather than the year 2000.

### Structuring element

$$
\begin{bmatrix}
0 & 1 & 0 \\
1 & 1 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

Slide 24 — Ứng dụng của phép giãn hình

24 of 50

# Ứng dụng của phép giãn hình

* Phép giãn hình có thể sửa những đối tượng bị đứt gãy
* Phép giãn hình có thể bổ sung những phần thiếu của đối tượng
* Lưu ý: Phép giãn hình có tác dụng phóng to các đối tượng

Slide 25 — Bài tập 3

25 of 50

# Bài tập 3

Cho tập A và phần tử cấu trúc B như hình vẽ. Tìm kết quả của phép co A với cấu trúc B.

* A
* B

Slide 26 — Bài tập 3

26 of 50

# Bài tập 3

* A

Slide 27 — Các phép xử lý hình thái kết hợp

27 of 50

# Các phép xử lý hình thái kết hợp

* Hai phép toán xử lý hình thái co và giãn có thể kết hợp với nhau để tạo ra phép xử lý hình thái mới:
  * Phép mở (Opening)
  * Phép đóng (Closing)

Slide 28 — Phép mở - Opening

28 of 50

# Phép mở - Opening

Phép toán mở ảnh $f$ bởi phần tử cấu trúc $s$, được biểu diễn bởi công thức $f \circ s$, bao gồm phép co hình và theo sau bởi phép giãn:

$$
f \circ s = (f \ominus s) \oplus s
$$

$$
A \circ B = (A \ominus B) \oplus B
$$

* Original shape: $A$
* After erosion: $A \ominus B$
* After dilation (opening): $A \circ B = (A \ominus B) \oplus B$
* *Note a disc shaped structuring element is used*

Slide 29 — Phép mở ảnh: Ví dụ

29 of 50

# Phép mở ảnh: Ví dụ

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

* Original Image
* Image After Opening

Slide 30 — Phép mở ảnh: ví dụ

30 of 50

# Phép mở ảnh: ví dụ

* Original Image
* Structuring Element

Slide 31 — Phép mở ảnh: Ví dụ

31 of 50

# Phép mở ảnh: Ví dụ

* Structuring Element

Slide 32 — Phép mở ảnh: Ví dụ

32 of 50

# Phép mở ảnh: Ví dụ

* Original Image
* Structuring Element

Slide 33 — Phép mở ảnh: Ví dụ

33 of 50

# Phép mở ảnh: Ví dụ

* Original Image
* Processed Image
* Structuring Element

Slide 34 — Ứng dụng của phép mở ảnh

34 of 50

# Ứng dụng của phép mở ảnh

Phép mở ảnh có tác dụng:
* Làm trơn các đường gãy khúc.
* Cắt đứt các dải eo hẹp nối giữa các đối tượng
* Loại bỏ các mảnh nhỏ và các góc nhọn trên ảnh

Slide 35 — Phép đóng - Closing

35 of 50

# Phép đóng - Closing

Phép đóng hình $f$ bởi phần tử cấu trúc $s$, được biểu diễn bởi công thức $f \bullet s$, bao gồm phép giãn hình và theo sau bởi phép co hình:

$$
f \bullet s = (f \oplus s) \ominus s
$$

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

$$
A \cdot B = (A \oplus B) \ominus B
$$

* Original shape: $A$
* After dilation: $A \oplus B$
* After erosion (closing): $A \cdot B = (A \oplus B) \ominus B$
* *Note a disc shaped structuring element is used*

Slide 36 — Phép đóng ảnh: Ví dụ

36 of 50

# Phép đóng ảnh: Ví dụ

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

* Original Image
* Image After Closing

Slide 37 — Phép đóng ảnh: Ví dụ

37 of 50

# Phép đóng ảnh: Ví dụ

* Original Image
* Processed Image
* Structuring Element

Slide 38 — Ứng dụng của phép đóng ảnh

38 of 50

# Ứng dụng của phép đóng ảnh

Phép đóng ảnh có tác dụng:
* Làm trơn các đường gãy khúc.
* Làm nổi bật các dải eo hẹp nối giữa các đối tượng
* Loại bỏ các khoảng trống nhỏ trên ảnh

Slide 39 — Ví dụ về xử lý ảnh hình thái

39 of 50

# Ví dụ về xử lý ảnh hình thái

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

* $A$
* Phần tử cấu trúc $B$:

$$
B = \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
$$

* $A \ominus B$
* $(A \ominus B) \oplus B = A \circ B$
* $(A \circ B) \oplus B$
* $[(A \circ B) \oplus B] \ominus B = (A \circ B) \bullet B$

Slide 40 — THUẬT TOÁN XỬ LÝ HÌNH THÁI Tách biên

40 of 50

# THUẬT TOÁN XỬ LÝ HÌNH THÁI

## Tách biên

Thuật toán tách biên được cho bởi công thức:

$$
\beta(A) = A - (A \ominus B)
$$

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

* $A$
* $B$ (với Origin tại tâm)
* $A \ominus B$
* $\beta(A)$

Slide 41 — Bài tập 4

41 of 50

# Bài tập 4

Cho bức ảnh sau A. Thực hiện phép co ảnh A bởi phần tử cấu trúc có dạng 3x3. Tách biên của đối tượng trong ảnh A.

$$
A = \begin{bmatrix}
1 & 1 & 1 & 0 & 1 & 1 & 1 & 1 & 1 & 0 \\
1 & 1 & 1 & 0 & 1 & 1 & 1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1
\end{bmatrix}
$$

Slide 42 — Giải bài tập 4

42 of 50

# Giải bài tập 4

```matlab
>> se = strel('square', 3)
>> E = imerode(A, se)
>> Ec = not(E)
>> beta = Ec & A
```

$$
E = \begin{bmatrix}
1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 1 \\
1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 1 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
1 & 1 & 1 & 0 & 1 & 1 & 1 & 1 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
\qquad
A = \begin{bmatrix}
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 0 & 1 & 1 & 1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1
\end{bmatrix}
$$

$$
\beta(A) = A \cap E^c = E^c \cap A = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 1 & 1 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 1 & 1 & 0 & 0 \\
0 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 1 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1
\end{bmatrix}
$$

Slide 43 — THUẬT TOÁN XỬ LÝ HÌNH THÁI Làm đầy vùng

43 of 50

# THUẬT TOÁN XỬ LÝ HÌNH THÁI

## Làm đầy vùng

Thuật toán làm đầy vùng là quá trình tìm tất cả các pixel bên trong của một vùng và gán các pixel tìm được giá trị bằng 1

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

* *Cho 1 điểm bên trong hình tròn, liệu có thể tô toàn bộ hình tròn này không?*

Slide 44 — THUẬT TOÁN XỬ LÝ HÌNH THÁI Làm đầy vùng

44 of 50

# THUẬT TOÁN XỬ LÝ HÌNH THÁI

## Làm đầy vùng

Thuật toán: Xác định 1 điểm bên trong vùng, sau đó thực hiện liên tục các phép giãn hình

$$
X_0 = p
$$

$$
X_k = (X_{k-1} \oplus B) \cap A^c, \quad k = 1, 2, 3, ...
$$

$$
\text{Until } X_k = X_{k-1}
$$

*Bound the growth*

Slide 45 — THUẬT TOÁN XỬ LÝ HÌNH THÁI Làm đầy vùng (cont...)

45 of 50

# THUẬT TOÁN XỬ LÝ HÌNH THÁI

## Làm đầy vùng (cont...)

Công thức của thuật toán:

$$
X_k = (X_{k-1} \oplus B) \cap A^c \qquad k = 1, 2, 3, ...
$$

Trong đó $X_0$ là điểm bắt đầu trong vùng, $B$ là phần tử cấu trúc và $A^c$ là phần bù của $A$.

Phương trình được lặp cho đến khi $X_k$ bằng $X_{k-1}$.

Kết quả cuối cùng là hợp của $X_k$ và $A$.

Slide 46 — THUẬT TOÁN XỬ LÝ HÌNH THÁI Làm đầy vùng (cont...)

46 of 50

# THUẬT TOÁN XỬ LÝ HÌNH THÁI

## Làm đầy vùng (cont...)

**FIGURE 9.15 Region filling.**
* (a) Set A.
* (b) Complement of A ($A^c$).
* (c) Structuring element B.
* (d) Initial point inside the boundary ($X_0$).
* (e)-(h) Various steps of Eq. (9.5-2) ($X_1, X_2, X_6, X_7$).
* (i) Final result [union of (a) and (h)] ($X_7 \cup A$).

Slide 47 — THUẬT TOÁN XỬ LÝ HÌNH THÁI Làm đầy vùng (cont...)

47 of 50

# THUẬT TOÁN XỬ LÝ HÌNH THÁI

## Làm đầy vùng (cont...)

* $X_0$
* $X_1$
* $X_2$
* $X_3$
* $X_4$
* $X_5$
* $X_6$
* $X_7$
* $X_8$

Slide 48 — THUẬT TOÁN XỬ LÝ HÌNH THÁI Làm đầy vùng (cont...)

48 of 50

# THUẬT TOÁN XỬ LÝ HÌNH THÁI

## Làm đầy vùng (cont...)

* Original Image
* One Region Filled
* All Regions Filled

Slide 49 — Bài tập 5

49 of 50

# Bài tập 5

Cho bức ảnh sau A. Thực hiện các bước làm đầy vùng cho đối tượng trong ảnh A bởi cấu trúc B

$$
A = \begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 1 & 0 & 1 & 1 & 0 \\
0 & 1 & 0 & 1 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 1 & 1 & 1 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
\qquad
B = \begin{bmatrix}
0 & 1 & 0 \\
1 & 1 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

Slide 50 — Kết luận

50 of 50

# Kết luận

* Phép co và phép giãn
* Phép đóng và phép mở ảnh
* Các thuật toán xử lý ảnh hình thái