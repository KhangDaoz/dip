> [!NOTE]
> **Lưu ý về hình ảnh minh họa:** File markdown này được trích xuất nội dung văn bản từ tài liệu gốc. Do định dạng văn bản chưa nhúng trực tiếp toàn bộ ảnh minh họa, biểu đồ và sơ đồ thị giác, nếu cần xem chi tiết hình ảnh trực quan, vui lòng mở file PDF gốc tại: [3_XLA3_Nang cao chat luong anh.pdf](3_XLA3_Nang cao chat luong anh.pdf)
>

---

## Slide 1 — XỬ LÝ ẢNH SỐ CHƯƠNG 3. NÂNG CAO CHẤT LƯỢNG ẢNH

# XỬ LÝ ẢNH SỐ
## CHƯƠNG 3. NÂNG CAO CHẤT LƯỢNG ẢNH

---

## Slide 2 — Nội dung

**Original Slide Number:** 2

- Trong phần này chúng ta sẽ tìm hiểu về các kỹ thuật tăng cường ảnh trong miền không gian:
  - Thế nào là tăng cường ảnh?
  - Các kiểu tăng cường ảnh
  - Cải thiện ảnh sử dụng toán tử điểm ảnh
  - Cải thiện ảnh sử dụng toán tử láng giềng
  - Khôi phục ảnh

---

## Slide 3 — Thế nào là tăng cường ảnh?

**Original Slide Number:** 3

- Tăng cường ảnh là quá trình làm cho bức ảnh trở nên hữu ích hơn.
- ✓ Lý do để thực hiện tăng cường ảnh bao gồm:
  - Làm nổi bật những chi tiết cần quan tâm trong bức ảnh
  - Loại bỏ nhiễu khỏi các bức ảnh
  - Làm cho bức ảnh hấp dẫn hơn về mặt trực quan

---

## Slide 4 — Ví dụ về tăng cường ảnh (1)

**Original Slide Number:** 4

![MRI of a fractured human spine and an enhanced image](images/slide-04-figure-01.png)

Magnetic resonance image (MRI) of a fractured human spine and an enhanced image

---

## Slide 5 — Ví dụ về tăng cường ảnh (2)

**Original Slide Number:** 5

![Image of whole body bone scan (from front and back)](images/slide-05-figure-01.png)

Image of whole body bone scan (from front and back)

---

## Slide 6 — Ví dụ về tăng cường ảnh (3)

**Original Slide Number:** 6

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

![An image of an electronic board before and after enhancement](images/slide-06-figure-01.png)

An image of an electronic board

---

## Slide 7 — Ví dụ về tăng cường ảnh (4)

**Original Slide Number:** [7]

![Portrait image enhancement showing reduction in fine skin lines](images/slide-07-figure-01.png)

**Embedded Text in Diagram:**
- Higher attraction
- Reduction in fine skin lines in the magnified sections

---

## Slide 8 — Miền không gian và miền tần số

**Original Slide Number:** 8

- Các kỹ thuật tăng cường ảnh có thể chia ra 2 nhánh:
  - Các kỹ thuật tăng cường ảnh trong miền không gian
    - Thao tác trực tiếp với các điểm ảnh
  - Các kỹ thuật tăng cường ảnh trong miền tần số
    - Đưa ảnh về miền tần số bằng phép biến đổi fourier hoặc wavelet sau đó xử lý trên miền tần số
- Chương này sẽ tập trung vào các kỹ thuật xử lý ảnh trong miền không gian

---

## Slide 9 — CẢI THIỆN ẢNH SỬ DỤNG TOÁN TỬ ĐIỂM

**Original Slide Number:** 9

# CẢI THIỆN ẢNH SỬ DỤNG TOÁN TỬ ĐIỂM

---

## Slide 10 — Toán tử điểm

**Original Slide Number:** 10

- Xử lý ảnh trên miền không gian
  - Phép toán xử lý ảnh có dạng: $g(x,y)=T[f(x,y)]$
    - $f(x,y)$ là ảnh đầu vào
    - $g(x,y)$ là ảnh đầu ra
    - $T$ là một số toán tử tác động lên các điểm ảnh xung quanh $(x,y)$

![Spatial domain neighborhood around point (x,y) in Image f](images/slide-10-figure-01.png)

**Embedded Labels in Diagram:**
- Origin
- $x$
- $y$
- $(x, y)$
- Image $f(x, y)$

---

## Slide 11 — Xử lý điểm ảnh

**Original Slide Number:** 11

- Xử lý ảnh trên miền không gian
  $$g(x,y)=T[f(x,y)]$$
- ✓ Nếu $T$ chỉ tác động lên 1 điểm ảnh tại $(x,y)$ thì $T$ là hàm biến đổi mức xám hay toán tử xử lý điểm ảnh
- ✓ Các toán tử xử lý điểm ảnh có dạng:
  $$s=T(r)$$
  - $r$: mức xám của ảnh đầu vào tại $(x,y)$
  - $s$: mức xám của ảnh đầu ra tại $(x,y)$

![Contrast stretching transformation curve](images/slide-11-figure-01.png)

**Embedded Labels in Graph 1:**
- **Y-axis:** $s = T(r)$, Dark $\longleftrightarrow$ Light, $s_{0}=T(r_{0})$
- **X-axis:** $r$, Dark $\longleftrightarrow$ Light, $k$, $r_{0}$
- **Curve Label:** $T(r)$

![Thresholding transformation curve](images/slide-11-figure-02.png)

**Embedded Labels in Graph 2:**
- **Y-axis:** $s = T(r)$, Dark $\longleftrightarrow$ Light
- **X-axis:** $r$, Dark $\longleftrightarrow$ Light, $k$
- **Curve Label:** $T(r)$

---

## Slide 12 — MỘT SỐ TOÁN TỬ ĐIỂM ẢNH

**Original Slide Number:** 12

- Biến đổi âm bản
- Phân ngưỡng
- Một số hàm biến đổi mức xám cơ bản: tuyến tính, logarithm, hàm mũ.

---

## Slide 13 — Phép biến đổi âm bản

**Original Slide Number:** 13

- Phép biến đổi âm bản (negative image) sử dụng để tăng cường các chi tiết trắng hoặc xám trong nền tối
- Phép biến đổi: $s=T(r)=Intensity_{max}-r$

![Original digital mammogram](images/slide-13-figure-01.png)

**Embedded Flow Label:**
- Ảnh gốc $\xrightarrow{s=1.0-r}$ Ảnh âm bản

![Negative image of digital mammogram](images/slide-13-figure-02.png)

A digital mammogram and its negative image

---

## Slide 14 — Phân ngưỡng (Thresholding)

**Original Slide Number:** 14

- Biến đổi phân ngưỡng được sử dụng cho việc làm nổi bật các đổi tượng cần quan tâm trên nền của ảnh
- Phép biến đổi:

$$s=\begin{cases}1.0 & r>threshold\\ 0.0 & r<=threshold\end{cases}$$

![Original moon image](images/slide-14-figure-01.png)

![Thresholded binary moon image](images/slide-14-figure-02.png)

---

## Slide 15 — Các hàm cơ bản biến đổi mức xám

**Original Slide Number:** 15

- Có 3 hàm biến đổi mức xám thông dụng nhất:
  - Biến đổi tuyến tính
    - Negative/Identity
  - Biến đổi logarith
    - Log/Inverse log
  - Biến đổi hàm mũ
    - $n^{th}\text{ power}/n^{th}\text{ root}$

![Basic gray-level transformation functions plot](images/slide-15-figure-01.png)

**Embedded Labels in Graph:**
- **Y-axis:** Output gray level, $s$ [Values: $0, L/4, L/2, 3L/4, L-1$]
- **X-axis:** Input gray level, $r$ [Values: $0, L/4, L/2, 3L/4, L-1$]
- **Curves:**
  - Negative
  - Log
  - $n\text{th root}$
  - Identity
  - $n\text{th power}$
  - Inverse log

---

## Slide 16 — Biến đổi logarith

**Original Slide Number:** 16

- Dạng tổng quát của biến đổi log:
  $$s=c*\log(1+r)$$
- Biến đổi log ánh xạ một dải hẹp mức xám thấp của ảnh đầu vào sang một dải rộng hơn cho ảnh đầu ra
- Biến đổi ngược của hàm log thực hiện điều ngược lại

---

## Slide 17 — Biến đổi logarith (tiếp...)

**Original Slide Number:** [17]

- Các hàm log Log được sử dụng khi giá trị mức xám của ảnh đầu vào nằm trong 1 dải rất rộng.
- Biến đổi Fourier của ảnh được đưa qua biến đổi log để hiển thị các chi tiết rõ ràng hơn

![Fourier spectrum before and after log transformation](images/slide-17-figure-01.png)

**Embedded Formula:**
$$s=\log(1+r)$$

---

## Slide 18 — Biến đổi hàm mũ (1)

**Original Slide Number:** 18

- Biến đổi hàm mũ có dạng:
  $$s=c*r^{\gamma}$$
- Biến đổi hàm mũ ánh xạ một dải hẹp mức xám thấp của ảnh đầu vào đầu vào sang một dải rộng hơn cho ảnh đầu ra và ngược lại
- Khác với biến đổi log, có thể thay đổi giá trị $\gamma$ của biến đổi hàm mũ để tạo ra các đồ thị biến đổi khác nhau.

![Plots of power-law equation for various values of gamma](images/slide-18-figure-01.png)

**Embedded Labels in Graph:**
- **Y-axis:** Output gray level, $s$ [Values: $0, L/4, L/2, 3L/4, L-1$]
- **X-axis:** Input gray level, $r$ [Values: $0, L/4, L/2, 3L/4, L-1$]
- **Curves:**
  - $\gamma=0.04$
  - $\gamma=0.10$
  - $\gamma=0.20$
  - $\gamma=0.40$
  - $\gamma=0.67$
  - $\gamma=1$
  - $\gamma=1.5$
  - $\gamma=2.5$
  - $\gamma=5.0$
  - $\gamma=10.0$
  - $\gamma=25.0$

---

## Slide 19 — Biến đổi hàm mũ (2)

**Original Slide Number:** 19

![Power-law transformation of MRI spine image with gamma = 0.6, 0.4, 0.3](images/slide-19-figure-01.png)

**Embedded Labels on Arrows:**
- $s=r^{0.6}$
- $s=r^{0.4}$
- $s=r^{0.3}$

---

## Slide 20 — Biến đổi hàm mũ (3)

**Original Slide Number:** 20

![Power-law transformation of aerial image with gamma = 3.0, 4.0, 5.0](images/slide-20-figure-01.png)

**Embedded Labels on Arrows:**
- $s=r^{3.0}$
- $s=r^{4.0}$
- $s=r^{5.0}$

---

## Slide 21 — Tăng độ tương phản

**Original Slide Number:** 21

- Thay vì sử dụng 1 hàm toán học, ta có thể sử dụng nhiều hàm toán học tại các phân đoạn khác nhau để biến đổi mức xám

![Low contrast pollen image](images/slide-21-figure-01.png)

![Piecewise linear contrast stretching transformation function](images/slide-21-figure-02.png)

**Embedded Labels in Graph:**
- **Y-axis:** Ouput gray level, $s$ [Values: $0, L/4, L/2, 3L/4, L-1$]
- **X-axis:** Input gray level, $r$ [Values: $0, L/4, L/2, 3L/4, L-1$]
- **Curve & Points:** $T(r)$, $(r_{1},s_{1})$, $(r_{2},s_{2})$

![Contrast stretched pollen image](images/slide-21-figure-03.png)

---

## Slide 22 — LƯỢC ĐỒ XÁM VÀ CÂN BẰNG LƯỢC ĐỒ XÁM

**Original Slide Number:** 22

# LƯỢC ĐỒ XÁM VÀ CÂN BẰNG LƯỢC ĐỒ XÁM

---

## Slide 23 — Histogram (Lược đồ mức xám)

**Original Slide Number:** 23

- Lược đồ xám của ảnh có mức xám trong khoảng $[0, L - 1]$ là hàm rời rạc có dạng như sau:
  $$p(r_{k})=\frac{n_{k}}{n}$$
  với
  $$\begin{cases} n_{k} \text{ là số lượng pixel có giá trị xám } r_{k} \\ n \text{ là tổng số pixel của ảnh} \end{cases}$$
- Hàm $p(r_{k})$ mô tả tỷ lệ giữa tổng số pixel có mức xám $r_{k}$ và tổng số pixel của ảnh
- Histogram mô tả tổng thể phân bố mức xám của ảnh

---

## Slide 24 — Lược đồ xám của ảnh

**Original Slide Number:** 24

- Ví dụ về biểu diễn histogram:

![Example histogram chart showing Frequencies vs Grey Levels](images/slide-24-figure-01.png)

**Embedded Labels in Chart:**
- **Y-axis:** Frequencies
- **X-axis:** Grey Levels

---

## Slide 25 — Ví dụ lược đồ xám (cont...)

**Original Slide Number:** 25

![Grayscale portrait image](images/slide-25-figure-01.png)

![Scilab Graphic histogram of grayscale portrait image](images/slide-25-figure-02.png)

**Embedded Text in Screenshot:**
- **Window Title:** Scilab Graphic (0)
- **Menu:** File Tools Edit
- **Y-axis:** $0, 1, 2, 3, 4, 5, 6, 7$
- **X-axis:** $0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9$

---

## Slide 26 — Ví dụ lược đồ xám (cont...)

**Original Slide Number:** 26 [partially cut off as 20 in text extraction]

![Dark pollen image](images/slide-26-figure-01.png)

![Scilab Graphic histogram of dark pollen image](images/slide-26-figure-02.png)

**Embedded Text in Screenshot:**
- **Window Title:** Scilab Graphic (0)
- **Menu:** File Tools Edit
- **Y-axis:** $0, 2, 4, 6, 8, 10, 12, 14, 16, 18$
- **X-axis:** $0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0$

---

## Slide 27 — Ví dụ lược đồ xám (cont...)

**Original Slide Number:** 27 [partially cut off as 21 in text extraction]

![Bright pollen image](images/slide-27-figure-01.png)

![Scilab Graphic histogram of bright pollen image](images/slide-27-figure-02.png)

**Embedded Text in Screenshot:**
- **Window Title:** Scilab Graphic (0)
- **Menu:** File Tools Edit
- **Y-axis:** $0, 5, 10, 15, 20, 25, 30$
- **X-axis:** $0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0$

---

## Slide 28 — Ví dụ lược đồ xám (cont...)

**Original Slide Number:** 28

![Low-contrast pollen image](images/slide-28-figure-01.png)

![Scilab Graphic histogram of low-contrast pollen image](images/slide-28-figure-02.png)

**Embedded Text in Screenshot:**
- **Window Title:** Scilab Graphic (0)
- **Menu:** File Tools Edit
- **Y-axis:** $0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20$
- **X-axis:** $0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0$

---

## Slide 29 — Ví dụ lược đồ xám (cont...)

**Original Slide Number:** 29

![High-contrast pollen image](images/slide-29-figure-01.png)

---

## Slide 30 — Ví dụ lược đồ xám (cont...)

**Original Slide Number:** 30

![High-contrast pollen image](images/slide-30-figure-01.png)

![Scilab Graphic histogram of high-contrast pollen image](images/slide-30-figure-02.png)

**Embedded Text in Screenshot:**
- **Window Title:** Scilab Graphic (0)
- **Menu:** File Tools Edit
- **Y-axis:** $0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10$
- **X-axis:** $0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0$

---

## Slide 31 — Ví dụ lược đồ xám (cont...)

**Original Slide Number:** [31]

- Bức ảnh có độ tương phản cao thì phân bố xác xuất mức xám đều và đồng nhất

![Star graphic with text Tăng độ tương phản](images/slide-31-figure-01.png)

**Embedded Text in Star:**
- Tăng độ tương phản

![Four pollen images and their corresponding histograms](images/slide-31-figure-02.png)

**Embedded Labels in Histograms:**
- Dark image
- Bright image
- Low-contrast image
- High-contrast image

---

## Slide 32 — Cân bằng lược đồ xám

**Original Slide Number:** 32

- Việc phải làm là tìm hàm biến đổi
  $$s=T(r)$$
  $$p_{R}(r) \xrightarrow{T} \text{uniform } p_{S}(s)$$

![Histogram transformation T from narrow distribution to uniform distribution](images/slide-32-figure-01.png)

**Embedded Label in Diagram:**
- $T$

---

## Slide 33 — Cân bằng lược đồ xám

**Original Slide Number:** 33

$$s=T(r)=(L-1)\int_{0}^{r}p_{r}(w)dw$$

$$s_{k}=T(r_{k})=(L'-1)\sum_{j=0}^{k}p_{r}(r_{j})=(L'-1)\sum_{j=0}^{k}\frac{n_{j}}{n}$$

- $r_{k}$: mức xám đầu vào
- $s_{k}$: mức xám đầu ra
- $k$: dải cường độ sáng
- $n_{j}$: tần suất của cường độ $j$
- $n$: tổng số điểm ảnh
- $L'$: số mức xám ở ảnh đầu ra (thông thường bằng số mức xám đầu vào: $L'=L$)

---

## Slide 34 — Ví dụ 1

**Original Slide Number:** 34

- Xét bức ảnh có kích thước $64\times 64$ ở 8 mức xám $(0,1,...,7)$
- Chuẩn hóa giá trị mức xám như trong bảng sau. Hãy thực hiện quá trình cân bằng mức xám cho ảnh.

| $k$ | $r_{k}$ | $n_{k}$ | $P(r_{k})=n_{k}/n$ |
| :--- | :--- | :--- | :--- |
| 0 | $r_{0}=0$ | 790 | 0.19 |
| 1 | $r_{1}=1$ | 1023 | 0.25 |
| 2 | $r_{2}=2$ | 850 | 0.21 |
| 3 | $r_{3}=3$ | 656 | 0.16 |
| 4 | $r_{4}=4$ | 329 | 0.08 |
| 5 | $r_{5}=5$ | 245 | 0.06 |
| 6 | $r_{6}=6$ | 122 | 0.03 |
| 7 | $r_{7}=7$ | 81 | 0.02 |

---

## Slide 35 — Ví dụ 1

**Original Slide Number:** 35

$$s_{0}=T(r_{0})=7\sum_{j=0}^{0}p_{r}(r_{j})=7p_{r}(r_{0})=1.33$$

$$s_{1}=T(r_{1})=3.08,\ s_{2}=4.55,\ s_{3}=5.67,\ s_{4}=6.23,$$
$$s_{5}=6.65,\ s_{6}=6.86 \text{ and } s_{7}=7.00.$$

$$s_{0}=1.33\rightarrow 1 \quad s_{2}=4.55\rightarrow 5 \quad s_{4}=6.23\rightarrow 6 \quad s_{6}=6.86\rightarrow 7$$
$$s_{1}=3.08\rightarrow 3 \quad s_{3}=5.67\rightarrow 6 \quad s_{5}=6.65\rightarrow 7 \quad s_{7}=7.00\rightarrow 7$$

---

## Slide 36 — Ví dụ 1

**Original Slide Number:** 36

![Original histogram p_r(r_k)](images/slide-36-figure-01.png)

**Embedded Labels in Graph 1:**
- **Y-axis:** $p_{r}(r_{k})$ [Values: $.05, .10, .15, .20, .25$]
- **X-axis:** $r_{k}$ [Values: $0, 1, 2, 3, 4, 5, 6, 7$]

![Transformation function T(r)](images/slide-36-figure-02.png)

**Embedded Labels in Graph 2:**
- **Y-axis:** $s_{k}$ [Values: $1.4, 2.8, 4.2, 5.6, 7.0$]
- **X-axis:** $r_{k}$ [Values: $0, 1, 2, 3, 4, 5, 6, 7$]
- **Curve Label:** $T(r)$

![Equalized histogram p_s(s_k)](images/slide-36-figure-03.png)

**Embedded Labels in Graph 3:**
- **Y-axis:** $p_{s}(s_{k})$ [Values: $.05, .10, .15, .20, .25$]
- **X-axis:** $s_{k}$ [Values: $0, 1, 2, 3, 4, 5, 6, 7$]

---

## Slide 37 — Ví dụ 2

**Original Slide Number:** 37

- Cho bức ảnh $I$. Hãy vẽ lược đồ xám và thực hiện cân bằng lược đồ xám. Tìm ảnh $I'$ sau khi đã cân bằng lược đồ xám.

$$I=\begin{bmatrix}1&2&0&4\\ 1&0&0&7\\ 2&2&1&0\\ 4&1&2&1\\ 2&0&1&1\end{bmatrix}$$

---

## Slide 38 — Ví dụ 3

**Original Slide Number:** 38

- Cho bức ảnh $I$. Hãy vẽ lược đồ xám và thực hiện cân bằng lược đồ xám. Tìm ảnh $I'$ sau khi đã cân bằng lược đồ xám.

$$I=\begin{bmatrix}0&5&0&5&5\\ 1&7&6&6&7\\ 5&5&6&2&7\\ 6&6&7&5&6\\ 0&1&5&0&2\end{bmatrix}$$

---

## Slide 39 — Tách ngưỡng và tách ngưỡng tự động

**Original Slide Number:** 39

- Giả sử ta có: ảnh $I$ có kích thước $m\times n$, hai số $Min$, $Max$ và một giá trị mức xám $\theta$ gọi là ngưỡng.
- Khi đó, kỹ thuật tách ngưỡng được thể hiện (theo dạng biểu thức điều kiện của Ngôn ngữ lập trình C) như sau:
  $$I[i,j]=I[i,j]>=\theta\ ?\ Max : Min;\text{ với mọi }(i,j)$$
- Hay:
  - Nếu $I[i,j]>=\theta$ thì $I[i,j]=max$
  - Nếu $I[i,j]<\theta$ thì $I[i,j]=min$
- Nếu $Min=0$, $Max=1$ thì ảnh thu được sau tách ngưỡng là ảnh đen trắng.

---

## Slide 40 — Tách ngưỡng và tách ngưỡng tự động

**Original Slide Number:** [40]

- Thuật toán tìm ngưỡng tự động của ảnh:
  - $I \sim$ kích thước $m\times n$
  - $G \sim$ là số mức xám của ảnh kể cả khuyết thiếu
  - $t(g) \sim$ số điểm ảnh có mức xám $\le g$
  - $m(g)=\frac{1}{t(g)}\sum_{i=0}^{g}i\cdot h(i) \sim$ mômen quán tính TB có mức xám $\le g$
- Hàm $f: g\mapsto f(g)$ được định nghĩa là:
  $$f(g)=\frac{t(g)}{m\times n-t(g)}[m(g)-m(G-1)]^{2}$$
- Ngưỡng $\theta$ được xác định sao cho:
  $$f(\theta)=\max_{0\le g<G-1}\{f(g)\}$$

---

## Slide 41 — Tách ngưỡng và tách ngưỡng tự động

**Original Slide Number:** 41

- Ví dụ tìm ngưỡng tự động cho ảnh sau:

$$I = \begin{bmatrix} 0 & 1 & 2 & 3 & 4 & 5 \\ 0 & 0 & 1 & 2 & 3 & 4 \\ 0 & 0 & 0 & 1 & 2 & 3 \\ 0 & 0 & 0 & 0 & 1 & 2 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$

---

## Slide 42 — Tách ngưỡng và tách ngưỡng tự động

**Original Slide Number:** [42]

- Kết quả:

$$f(g)=\frac{t(g)}{m\times n-t(g)}[m(g)-m(G-1)]^{2}$$

$$m(g)=\frac{1}{t(g)}\sum_{i=0}^{g}i\cdot h(i) \sim \text{mômen quán tính TB có mức xám }\le g$$

$$I = \begin{bmatrix} 0 & 1 & 2 & 3 & 4 & 5 \\ 0 & 0 & 1 & 2 & 3 & 4 \\ 0 & 0 & 0 & 1 & 2 & 3 \\ 0 & 0 & 0 & 0 & 1 & 2 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$

| $g$ | $h(g)$ | $t(g)$ | $g\cdot h(g)$ | $\sum_{i=0}^{g}i\cdot h(i)$ | $m(g)$ | $f(g)$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 15 | 15 | 0 | 0 | 0 | 1.35 |
| 1 | 5 | 20 | 5 | 5 | 0,25 | 1.66 |
| 2 | 4 | 24 | 8 | 13 | 0,54 | 1.54 |
| 3 | 3 | 27 | 9 | 22 | 0,81 | 1.10 |
| 4 | 2 | 29 | 8 | 30 | 1,03 | 0.49 |
| 5 | 1 | 30 | 5 | 35 | 1,16 | 0 |

Ngưỡng tìm được $\theta=1$ ứng với $f(\theta)=1.66.$

---

## Slide 43 — CẢI THIỆN ẢNH SỬ DỤNG CỬA SỔ

**Original Slide Number:** 43

# CẢI THIỆN ẢNH SỬ DỤNG CỬA SỔ

---

## Slide 44 — Toán tử láng giềng (neighbourhood)

**Original Slide Number:** 44

- Toán tử láng giềng hoạt động trên một tập các pixel xung quanh pixel tâm.
- Neighbourhoods (láng giềng, lân cận) hầu hết có dạng hình vuông hay chữ nhật quanh 1 pixel tâm.

![Neighbourhood around pixel (x, y) in Image f(x, y)](images/slide-44-figure-01.png)

**Embedded Labels in Diagram:**
- Origin
- $x$
- $y$
- Neighbourhood
- $(x, y)$
- Image $f(x, y)$

---

## Slide 45 — Toán tử láng giềng

**Original Slide Number:** 45

- Đối với mỗi pixel trong ảnh gốc, sau xử lý đầu ra là pixel ở cùng vị trí đó trong ảnh đích.

![Mapping a neighbourhood in Image f(x, y) to Target pixel in Image g(x, y)](images/slide-45-figure-01.png)

**Embedded Labels in Diagram:**
- Origin
- $x$
- $y$
- Neighbourhood
- $(x,y)$
- Image $f(x, y)$
- Target
- Image $g(x,y)$

---

## Slide 46 — Một số toán tử láng giềng đơn giản

**Original Slide Number:** 46

- Min: Thiết lập giá trị pixel trong ảnh mới là giá trị min của neighbourhood.
- Max: Thiết lập giá trị pixel trong ảnh mới là max của neighbourhood.

---

## Slide 47 — Quá trình lọc không gian

**Original Slide Number:** 47

![3x3 spatial filter applied to a 3x3 neighbourhood around pixel e](images/slide-47-figure-01.png)

**Embedded Labels in Diagram:**
- Origin
- $x$
- $y$
- Simple $3*3$ Neighbourhood
- $e$
- $3*3$ Filter
- Image $f(x, y)$

**Original Image Pixels:**

| $a$ | $b$ | $c$ |
| :---: | :---: | :---: |
| $d$ | $e$ | $f$ |
| $g$ | $h$ | $i$ |

$*$

**Filter ($w$):**

| $j$ | $k$ | $l$ |
| :---: | :---: | :---: |
| $m$ | $n$ | $o$ |
| $p$ | $q$ | $r$ |

$$e_{processed}=n*e+j*a+k*b+l*c+m*d+o*f+p*g+q*h+r*i$$

Quá trình trên được lặp lại đối với mỗi pixel trong ảnh gốc để tạo ra ảnh mới.

---

## Slide 48 — Lọc không gian: Công thức

**Original Slide Number:** 48

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

![Spatial filtering mechanism with kernel coefficients w(s,t) and image pixels f(x,y)](images/slide-48-figure-01.png)

**Embedded Labels in Diagram:**
- Image origin
- $x$
- $y$
- Kernel origin
- $s$
- $t$
- Image pixels
- Filter kernel
- Image $f$
- Filter kernel, $w(s,t)$
- Magnified view showing filter kernel coefficients and corresponding pixels in the image

**Kernel coefficients:**

| $w(-1,-1)$ | $w(-1,0)$ | $w(-1,1)$ |
| :---: | :---: | :---: |
| $w(0,-1)$ | $w(0,0)$ | $w(0,1)$ |
| $w(1,-1)$ | $w(1,0)$ | $w(1,1)$ |

**Pixel values under kernel when it is centered on $(x,y)$:**

| $f(x-1,y-1)$ | $f(x-1,y)$ | $f(x-1,y+1)$ |
| :---: | :---: | :---: |
| $f(x,y-1)$ | $f(x,y)$ | $f(x,y+1)$ |
| $f(x+1,y-1)$ | $f(x+1,y)$ | $f(x+1,y+1)$ |

$$g(x,y)=\sum_{s=-a}^{a}\sum_{t=-b}^{b}w(s,t)f(x+s,y+t)$$

---

## Slide 49 — Ví dụ

**Original Slide Number:** 49

![Step-by-step example of 3x3 spatial convolution on a 5x5 image patch](images/slide-49-figure-01.png)

**Embedded Matrices & Text in Diagram:**

- **Original image**
- **Image with color values placed over it:**
  $$\begin{matrix} 164 & 188 & 164 & 161 & 195 \\ 178 & 201 & 197 & 150 & 137 \\ 174 & 168 & 181 & 190 & 184 \\ 131 & 179 & 176 & 185 & 198 \\ 90 & 185 & 179 & 133 & 167 \end{matrix}$$
- **Image with 3x3 kernel placed over it:**
  $$\begin{matrix} 0\cdot 164 & 1\cdot 188 & 0\cdot 164 & 161 & 195 \\ 1\cdot 178 & 1\cdot 201 & 1\cdot 197 & 150 & 137 \\ 0\cdot 174 & 1\cdot 168 & 0\cdot 181 & 190 & 184 \\ 131 & 179 & 176 & 185 & 198 \\ 82 & 185 & 179 & 133 & 167 \end{matrix}$$
- **Output image**
- **Color values $\times$ Kernel:**
  $$\begin{bmatrix} 164 & 188 & 164 \\ 178 & 201 & 197 \\ 174 & 168 & 181 \end{bmatrix} \times \begin{bmatrix} 0 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
  $$=164*0+188*1+164*0+178*1+201*1+197*1+174*0+168*1+181*0$$
  $$=932$$
- **Divided by the sum of the kernel:**
  $$932/5 = \text{new pixel color}$$

---

## Slide 50 — Các bộ lọc không gian làm mịn ảnh

**Original Slide Number:** 50

- Một trong số các toán tử lọc không gian đơn giản nhất là toán tử làm mịn ảnh.
  - Đơn giản là lấy trung bình của tất cả các pixel trong vùng lân cận xung quanh giá trị tâm.
  - Đặc biệt hữu ích trong việc loại bỏ nhiễu trong ảnh.

**Bộ lọc trung bình đơn giản:**

| $1/9$ | $1/9$ | $1/9$ |
| :---: | :---: | :---: |
| $1/9$ | $1/9$ | $1/9$ |
| $1/9$ | $1/9$ | $1/9$ |

---

## Slide 51 — Bộ lọc không gian làm mịn ảnh

**Original Slide Number:** 51

![3x3 smoothing filter applied to a 3x3 image patch](images/slide-51-figure-01.png)

**Embedded Labels in Diagram:**
- Origin
- $x$
- $y$
- Simple $3*3$ Neighbourhood
- $3*3$ Smoothing Filter
- Image $f(x, y)$

**Original Image Pixels:**

| 104 | 100 | 108 |
| :---: | :---: | :---: |
| 99 | 106 | 98 |
| 95 | 90 | 85 |

$*$

**Filter:**

| $1/9$ | $1/9$ | $1/9$ |
| :---: | :---: | :---: |
| $1/9$ | $1/9$ | $1/9$ |
| $1/9$ | $1/9$ | $1/9$ |

$$e=\frac{1}{9}*106+\frac{1}{9}*104+\frac{1}{9}*100+\frac{1}{9}*108+\frac{1}{9}*99+\frac{1}{9}*98+\frac{1}{9}*95+\frac{1}{9}*90+\frac{1}{9}*85=98.3333$$

---

## Slide 52 — Ví dụ làm mịn ảnh

**Original Slide Number:** 52

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

- Ảnh phía trên bên trái là ảnh gốc kích thước $500*500$
- Các ảnh sau là ảnh sau khi lọc bằng bộ lọc trung bình với kích thước tăng dần 3, 5, 9, 15 và 35
- Chú ý các chi tiết dần biến mất.

![Test pattern image smoothed with averaging filters of size 3, 5, 9, 15, and 35](images/slide-52-figure-01.png)

**Embedded Text in Test Pattern Images:**
- a
- a a a a a a a

---

## Slide 53 — Bộ lọc trung bình có trọng số

**Original Slide Number:** 53

- Có thể tạo ra bộ lọc làm mịn ảnh hiệu quả hơn bằng cách gán trọng số cho các pixel khác nhau của lân cận trong hàm trung bình.
  - Các pixel gần pixel tâm có vai trò quan trọng hơn.
  - Các bộ lọc này thường được gọi là trung bình có trọng số (weighted averaging)

**Bộ lọc trung bình có trọng số:**

| $1/16$ | $2/16$ | $1/16$ |
| :---: | :---: | :---: |
| $2/16$ | $4/16$ | $2/16$ |
| $1/16$ | $2/16$ | $1/16$ |

---

## Slide 54 — Bộ lọc trung bình k giá trị gần nhất

**Original Slide Number:** 54

- Cho ảnh $I$, pixel $P$, cửa sổ $W(P)$, ngưỡng $\theta$, giá trị $k$.
- Thuật toán:
  - Bước 1: Tìm $k$ giá trị nằm trong cửa sổ $W(P)$ gần với giá trị pixel $P$ nhất.
  - Bước 2: Tính trung bình của $k$ giá trị này:
    $$AV_{k}(P)=\left(\sum_{i=1}^{k}P_{i}\right)/k$$
  - Bước 3: Gán giá trị
    $$I(P)=\begin{cases}I(P) & |I(P)-AV_{k}(P)|\le\theta\\ AV_{k}(P) & |I(P)-AV_{k}(P)|>\theta\end{cases}$$

---

## Slide 55 — Ví dụ

**Original Slide Number:** 55

- Cho ảnh $I$ như sau:
- Cửa sổ $W(3\times 3)$,
- Ngưỡng $\theta=2$, giá trị $k=3$

$$I=\begin{pmatrix}1&2&3&2\\ 4&16&2&1\\ 4&2&1&1\\ 2&1&2&1\end{pmatrix} \quad I_{kq}=\begin{pmatrix}1&2&3&2\\ 4&(8)&2&1\\ 4&2&1&1\\ 2&1&2&1\end{pmatrix}$$

---

## Slide 56 — Bộ lọc trung vị (Median filter)

**Original Slide Number:** 56

- Lọc median (lọc trung vị) được dùng khi nhiễu có các mức xám tách biệt. Ví dụ: nhiễu muối tiêu
- Lọc median là lọc phi tuyến. Mỗi pixel được thay thế bằng giá trị giữa trong các pixel láng giềng
- Giả sử $A=\{a_{1},a_{2},...,a_{k}\}$ là các giá trị pixel láng giềng với $a_{1}\le a_{2}\le\cdots\le a_{k}$, thì
  $$median(A)=\begin{cases}a_{k/2}\text{ với }k\text{ chẵn}\\ a_{(k+1)/2}\text{ với }k\text{ lẻ}\end{cases}$$
- Ví dụ: $A=\{0,1,2,6,8,10,13\}$ thì $median(A)=6$

---

## Slide 57 — Ví dụ về bộ lọc trung bình và bộ lọc trung vị

**Original Slide Number:** 57

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

![Noisy circuit board image](images/slide-57-figure-01.png)

**Caption:** Ảnh gốc bị nhiễu

![Circuit board image after averaging filter](images/slide-57-figure-02.png)

**Caption:** Ảnh sau khi lọc trung bình

![Circuit board image after median filter](images/slide-57-figure-03.png)

**Caption:** Ảnh sau khi lọc trung vị

---

## Slide 58 — Xử lý biên ảnh

**Original Slide Number:** 58

- Tại biên của bức ảnh sẽ gặp trường hợp thiếu pixel ảnh để tạo thành neighbourhood.

![Diagram showing filter windows overhanging the borders of Image f(x, y)](images/slide-58-figure-01.png)

**Embedded Labels in Diagram:**
- Origin
- $x$
- $y$
- $e$
- Image $f(x, y)$

---

## Slide 59 — Xử lý biên ảnh (tiếp)

**Original Slide Number:** 59

- Có một số cách tiếp cận để giải quyết vấn đề thiếu pixel ở biên:
  - Loại bỏ pixel bị thiếu.
    - Chỉ làm việc với một số bộ lọc.
    - Làm thay đổi kích thước ảnh.
  - Chèn thêm pixel vào ảnh
    - Thông thường thêm vào hoặc toàn pixel đen hoặc toàn pixel trắng.
    - Làm thay đổi tính chất ảnh.
  - Lặp lại các pixels ở rìa ảnh.

---

## Slide 60 — Ví dụ

**Original Slide Number:** 60

- Cho bức ảnh bị nhiễu sau. Dùng bộ lọc trung bình và trung vị kích thước $3\times 3$ để loại bỏ nhiễu trong ảnh.

| 2 | 7 | 3 | 0 |
| :---: | :---: | :---: | :---: |
| 1 | 3 | 6 | 3 |
| 1 | 0 | 3 | 5 |
| 3 | 6 | 7 | 1 |

---

## Slide 61 — KHÔI PHỤC ẢNH

**Original Slide Number:** 61

# KHÔI PHỤC ẢNH

---

## Slide 62 — Nội dung

**Original Slide Number:** 62

- Thế nào là khôi phục ảnh?
- Nhiễu và ảnh
- Các mô hình nhiễu
- Loại bỏ nhiễu sử dụng bộ lọc không gian
- Nhiễu tuần hoàn và loại bỏ nhiễu sử dụng bộ lọc miền tần số.

---

## Slide 63 — Thế nào là khôi phục ảnh?

**Original Slide Number:** 63

- Khôi phục ảnh nỗ lực tái tạo lại bức ảnh đã bị xuống cấp về chất lượng.
  - Xác định quá trình làm bức ảnh xuống cấp và nỗ lực làm ngược lại.
  - Tương tự như tăng cường ảnh nhưng có mục đích hơn.

![Lunar surface image corrupted by periodic noise](images/slide-63-figure-01.png)

---

## Slide 64 — Nhiễu và ảnh

**Original Slide Number:** 64

- Nguồn nhiễu trong ảnh số thường bắt nguồn từ quá trình thu nhận ảnh (số hóa) và truyền ảnh.
  - Sensor thu nhận ảnh có thể bị ảnh hưởng bởi các điều kiện xung quanh.
  - Ảnh cũng có thể bị can nhiễu trong quá trình truyền dẫn.

![Illustration of satellite transmitting data to Earth](images/slide-64-figure-01.png)

---

## Slide 65 — Mô hình nhiễu

**Original Slide Number:** 65

- Ta có thể mô hình hóa ảnh nhiễu như sau:
  $$g(x,y)=f(x,y)+\eta(x,y)$$
  trong đó: $(x, y)$ là pixel bị nhiễu, $f(x,y)$ là cường độ sáng ở ảnh gốc, $\eta(x,y)$ là thành phần nhiễu, $g(x,y)$ là cường độ sáng ở pixel bị nhiễu.
- Nếu ta có thể ước đoán mô hình nhiễu của bức ảnh thì có thể đưa ra cách khôi phục bức ảnh.

---

## Slide 66 — Các mô hình nhiễu

**Original Slide Number:** 66

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

- Có nhiều mô hình nhiễu khác nhau sử dụng cho ảnh: $\eta(x,y)$
  - Gaussian
    - Mô hình thông dụng nhất
  - Rayleigh
  - Erlang
  - Exponential (hàm mũ)
  - Uniform (đồng nhất)
  - Impulse (hàm xung)
    - Salt and pepper noise

![Gaussian and Rayleigh PDF plots](images/slide-66-figure-01.png)

**Embedded Labels in Row 1 Plots:**
- **Gaussian:** $p(z)$, $\frac{1}{\sqrt{2\pi}\sigma}$, $\frac{0.607}{\sqrt{2\pi}\sigma}$, $\mu-\sigma$, $\mu$, $\mu+\sigma$, $z$
- **Rayleigh:** $p(z)$, $0.607\sqrt{\frac{2}{b}}$, $a$, $a+\sqrt{\frac{b}{2}}$, $z$

![Erlang and Exponential PDF plots](images/slide-66-figure-02.png)

**Embedded Labels in Row 2 Plots:**
- **Erlang:** $p(z)$, $K$, $K=\frac{a(b-1)^{b-1}}{(b-1)!}e^{-(b-1)}$, $(b-1)/a$, $z$
- **Exponential:** $p(z)$, $a$, $z$

![Uniform and Impulse PDF plots](images/slide-66-figure-03.png)

**Embedded Labels in Row 3 Plots:**
- **Uniform:** $p(z)$, $\frac{1}{b-a}$, $a$, $b$, $z$
- **Impulse:** $p(z)$, $P_{b}$, $P_{a}$, $a$, $b$, $z$

Với $p(z)$ là hàm PDF – Probability Density Function (Mật độ xác suất) $\sim$ Histogram của 1 đại lượng $p$ theo biến $z$

---

## Slide 67 — Ví dụ về nhiễu

**Original Slide Number:** 67

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

- Ảnh bên phải là ảnh gốc
- Các slide tiếp theo mô tả kết quả của quá trình thêm nhiễu vào ảnh với các mô hình khác nhau.

![Original synthetic test image with three gray levels](images/slide-67-figure-01.png)

**Caption:** Ảnh gốc

![Histogram of original synthetic test image](images/slide-67-figure-02.png)

**Caption:** Histogram

---

## Slide 68 — Ví dụ về nhiễu (tiếp)

**Original Slide Number:** 68

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

![Test images and histograms corrupted by Gaussian, Rayleigh, and Erlang noise](images/slide-68-figure-01.png)

**Embedded Labels:**
- Gaussian
- Rayleigh
- Erlang

---

## Slide 69 — Ví dụ về nhiễu (tiếp)

**Original Slide Number:** 69

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

![Test images and histograms corrupted by Exponential, Uniform, and Impulse noise](images/slide-69-figure-01.png)

**Embedded Labels:**
- Exponential
- Uniform
- Impulse

---

## Slide 70 — Lọc để loại bỏ nhiễu

**Original Slide Number:** 70

- Ta có thể sử dụng các loại bộ lọc không gian khác nhau để loại bỏ các loại nhiễu khác nhau: bộ lọc trung bình số học, hình học...
- Bộ lọc trung bình số học (arithmetic mean filter) là bộ lọc rất đơn giản và được tính như sau:
  $$\hat{f}(x,y)=\frac{1}{mn}\sum_{(r,c)\in S_{xy}}g(r,c)$$
  $m, n$ là kích cỡ của cửa sổ
- Bộ lọc này được thực hiện như bộ lọc làm mịn ảnh đơn giản.
- Làm mờ ảnh để loại bỏ nhiễu.

| $1/9$ | $1/9$ | $1/9$ |
| :---: | :---: | :---: |
| $1/9$ | $1/9$ | $1/9$ |
| $1/9$ | $1/9$ | $1/9$ |

---

## Slide 71 — Ví dụ về loại bỏ nhiễu

**Original Slide Number:** 71

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

- Bộ lọc trung bình hình học
  $$\hat{f}(x,y)=\left[\prod_{(r,c)\in S_{xy}}g(r,c)\right]^{\frac{1}{mn}}$$

![Comparison of original image, Gaussian noise image, arithmetic mean filter, and geometric mean filter](images/slide-71-figure-01.png)

**Captions:**
- Ảnh gốc
- Ảnh bị nhiễu Gaussian
- Sau khi lọc bằng bộ lọc trung bình số học $3*3$
- Sau khi lọc bằng bộ lọc trung bình hình học $3*3$

---

## Slide 72 — Một số bộ lọc không gian khác

**Original Slide Number:** 72

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

- Bộ lọc trung vị (Median filter)
- Bộ lọc max và min (Max and min filter)
- Bộ lọc điểm giữa (Midpoint filter)

---

## Slide 73 — Bộ lọc trung vị

**Original Slide Number:** 73

- Median filter: replaces the value of a pixel by the median of the intensity levels in a predefined neighborhood of that pixel:
  $$\hat{f}(x,y)=\text{median}_{(s,t)\in S_{xy}}\{g(s,t)\}$$
  where, as before, is a sub-image (neighborhood) centered on point $(x, y)$. The value of the pixel at $(x, y)$ is included in the computation of the median.

$$10, 15, 20, 20, \underset{\substack{\uparrow \\ \text{median}}}{20}, 20, 20, 25, 100$$

- Đặc trưng:
  - Loại bỏ nhiễu tốt mà không có hiệu ứng làm mờ ảnh như các bộ lọc làm mịn khác.
  - Đặc biệt hiệu quả với nhiễu muối tiêu.

---

## Slide 74 — Bộ lọc Max và Min

**Original Slide Number:** 74

- Bộ lọc Max:
  $$\hat{f}(x,y)=\max_{(s,t)\in S_{xy}}\{g(s,t)\}$$
- Bộ lọc Min:
  $$\hat{f}(x,y)=\min_{(s,t)\in S_{xy}}\{g(s,t)\}$$
- Bộ lọc Max hiệu quả với nhiễu tiêu và bộ lọc min hiệu quả với nhiễu muối.

---

## Slide 75 — Midpoint Filter

**Original Slide Number:** 75

- Midpoint Filter:
  $$\hat{f}(x,y)=\frac{1}{2}\left[\max_{(s,t)\in S_{xy}}\{g(s,t)\}+\min_{(s,t)\in S_{xy}}\{g(s,t)\}\right]$$
  - Hiệu quả với nhiễu Gaussian ngẫu nhiên và nhiễu đồng nhất.
- Alpha-trimmed mean filter:
  $$\hat{f}(x,y)=\frac{1}{mn-d}\sum_{(r,c)\in S_{xy}}g_{R}(r,c)$$
  $$0\le d\le mn-1$$
  *(Note: original slide displays $0\le 0\le mn-1$)*
  - Hiệu quả với nhiễu Gaussian ngẫu nhiên và nhiễu đồng nhất.

---

## Slide 76 — Ví dụ về loại bỏ nhiễu

**Original Slide Number:** 76

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

![Circuit board corrupted by salt and pepper noise and results of 1, 2, and 3 passes of 3x3 median filter](images/slide-76-figure-01.png)

**Captions:**
- Image corrupted by Salt and Pepper noise
- Result of 1 pass with a $3*3$ Median filter
- Result of 2 passes with a $3*3$ Median filter
- Result of 3 passes with a $3*3$ Median filter

---

## Slide 77 — Ví dụ về loại bỏ nhiễu (tiếp)

**Original Slide Number:** 77

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

![Comparison of pepper noise filtered with 3x3 Max filter and salt noise filtered with 3x3 Min filter](images/slide-77-figure-01.png)

**Captions:**
- Image Corrupted By Pepper Noise
- Result Of Filtering Above With A $3*3$ Max Filter
- Image Corrupted By Salt Noise
- Result Of Filtering Above With A $3*3$ Min Filter

---

## Slide 78 — Ví dụ về loại bỏ nhiễu (tiếp)

**Original Slide Number:** 78

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

![Comparison of 5x5 Arithmetic, Geometric, Median, and Alpha-Trimmed Mean filters on combined uniform and salt-and-pepper noise](images/slide-78-figure-01.png)

**Captions:**
- Image Corrupted By Uniform Noise
- Image Further Corrupted By Salt and Pepper Noise
- Filtered By $5*5$ Arithmetic Mean Filter
- Filtered By $5*5$ Geometric Mean Filter
- Filtered By $5*5$ Median Filter
- Filtered By $5*5$ Alpha-Trimmed Mean Filter

---

## Slide 79 — Nhiễu tuần hoàn

**Original Slide Number:** 79

*Images taken from Gonzalez & Woods, Digital Image Processing (2002)*

- Thường xuất hiện do can nhiễu điện hoặc điện từ.
- Thường tạo ra các mẫu nhiễu điển hình trong bức ảnh.
- Để loại bỏ nhiễu này thì sử dụng các kỹ thuật miền tần số là hiệu quả nhất.

![Lunar image with periodic noise and its Fourier spectrum showing impulse spikes](images/slide-79-figure-01.png)

---

## Slide 80 — Kết luận

**Original Slide Number:** 80

- Khôi phục ảnh là quá trình xử lý có mục đích hơn so với tăng cường ảnh.
- Các kỹ thuật miền không gian đặc biệt hữu ích đối với việc loại bỏ nhiễu ngẫu nhiên.
- Các kỹ thuật miền tần số đặc biệt hữu ích đối với nhiễu tuần hoàn.

---

## Slide 81 — Bài tập 1

**Original Slide Number:** 81

a) Tính biểu đồ tần suất $h(g)$ cho bức ảnh $I$ sau:

| 2 | 5 | 3 | 7 | 2 | 4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 6 | 1 | 8 | 3 | 5 | 0 |
| 4 | 9 | 2 | 3 | 6 | 1 |
| 7 | 2 | 0 | 5 | 3 | 8 |
| 5 | 4 | 0 | 5 | 1 | 3 |
| 3 | 6 | 7 | 9 | 4 | 6 |

b) $h'(g)$ là biểu đồ tần suất của ảnh $I'$ biến đổi từ ảnh $I$ bằng hàm $f(g)$ sau. Hãy tính $h'(g)$:
$$f(g)=|2g-7|$$

---

## Slide 82 — Bài tập 2

**Original Slide Number:** 82

- Cho ảnh $I$ như sau:

| 0 | 3 | 1 | 2 | 4 | 1 | 2 | 5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2 | 0 | 3 | 1 | 4 | 2 | 6 |
| 2 | 1 | 3 | 0 | 2 | 1 | 7 | 3 |
| 3 | 4 | 1 | 2 | 0 | 3 | 1 | 8 |
| 1 | 0 | 2 | 1 | 3 | 2 | 0 | 9 |
| 2 | 1 | 0 | 3 | 1 | 2 | 1 | 4 |

a) Sử dụng thuật toán tìm ngưỡng tự động để tách ngưỡng ảnh $I$, được biết ảnh có 10 mức xám.

b) Có lúc nào chúng ta có thể tìm được nhiều ngưỡng không? Trong trường hợp như vậy thì ta nên xử lý như thế nào?

$$f(g)=\frac{t(g)}{m\times n-t(g)}[m(g)-m(G-1)]^{2}$$

$$m(g)=\frac{1}{t(g)}\sum_{i=0}^{g}i\cdot h(i)$$

| $g$ | $h(g)$ | $t(g)$ | $g\cdot h(g)$ | $\sum_{i=0}^{g}i\cdot h(i)$ | $m(g)$ | $f(g)$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | | | | | | |

---

## Slide 83 — Bài tập 3

**Original Slide Number:** 83

a) Thực hiện cân bằng tần suất cho ảnh $I$ thành ảnh $I'$, được biết ảnh $I$ và $I'$ cùng là ảnh 6 mức xám (mức xám từ 0 đến 5):

| 1 | 3 | 4 | 4 | 3 | 1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 3 | 3 | 2 | 5 | 3 |
| 3 | 3 | 3 | 4 | 2 | 3 |
| 4 | 3 | 2 | 3 | 5 | 0 |
| 1 | 3 | 2 | 3 | 2 | 3 |
| 0 | 3 | 4 | 3 | 0 | 1 |

*(Note: Row 3 Column 2 and Row 5 Column 2 in the raw OCR showed doubled digits `33` due to overlapping table grid artifacts; all 6x6 cells are preserved above).*

b) Thế nào là ảnh cân bằng lý tưởng? Sau khi cân bằng, ảnh trên đã là cân bằng lý tưởng hay chưa?

---

## Slide 84 — Bài tập 4

**Original Slide Number:** 84

a) Thực hiện tìm ngưỡng tự động với thuật toán đẳng điệu (isodata) cho bức ảnh $I$ có biểu đồ tần suất sau, được biết ảnh có 10 mức xám. Mô tả từng bước cho đến khi tìm được ngưỡng mong muốn.

| $g$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $h(g)$ | 20 | 38 | 55 | 60 | 52 | 33 | 20 | 12 | 6 | 2 |

b) Thực hiện tìm ngưỡng tự động với thuật toán đối xứng nền cho bức ảnh $I'$ có biểu đồ tần suất sau, được biết độ chính xác cần tính là 92%:

| $g$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $h(g)$ | 5 | 9 | 16 | 28 | 42 | 96 | 35 | 18 | 8 | 3 |

Với hai biểu đồ tần suất như trên thì việc chọn phương pháp đã đúng chưa? Nếu được chọn lại thì bạn chọn như thế nào, tại sao?