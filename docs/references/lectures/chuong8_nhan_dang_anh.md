> [!NOTE]
> **Lưu ý về hình ảnh minh họa:** File markdown này được trích xuất nội dung văn bản từ tài liệu gốc. Do định dạng văn bản chưa nhúng trực tiếp toàn bộ ảnh minh họa, biểu đồ và sơ đồ thị giác, nếu cần xem chi tiết hình ảnh trực quan, vui lòng mở file PDF gốc tại: [8_XLA8_NhanDangAnh.pdf](8_XLA8_NhanDangAnh.pdf)
>

---

# Slide 1 — XỬ LÝ ẢNH SỐ - Chương 8: Nhận dạng ảnh

**XỬ LÝ ẢNH SỐ**  
**Chương 8**  
**Nhận dạng ảnh**

---

# Slide 2 — Nội dung

2 of 89

**Nội dung**

* Nhận dạng ảnh: bài toán, thách thức, hệ thống nhận dạng
* Học có giám sát và không giám sát
* Trích chọn đặc trưng: màu, kết cấu, hình dạng, cục bộ
* Nhận dạng dựa theo không gian
  * K-means, khoảng cách lớn nhất, ISODATA, Mean-Shift
  * k láng giềng (kNN), mô hình túi từ, đánh giá
* Nhận dạng theo cấu trúc
* Một số mô hình mạng nơ ron
  * Mạng Hopfield, mạng Kohonen, hướng tới CNN

---

# Slide 3 — Quá trình nhận dạng

3 of 89

**Quá trình nhận dạng**

* Quá trình nhận dạng dựa vào những mẫu học biết trước:
  * Học có giám sát; ngược lại: Học không giám sát
* Quá trình nhận dạng
  * Lựa chọn mô hình biểu diễn đối tượng
  * Lựa chọn luật ra quyết định (phương pháp nhận dạng) và suy diễn quá trình học
* Học nhận dạng
  * Khi mô hình biểu diễn đã được xác định
    * Mô hình tham số
    * Mô hình cấu trúc
  * Đến quá trình học nhằm cải thiện, điều chỉnh việc phân lớp
  * Nhận dạng là tìm ra quy luật và thuật toán để gán đối tượng vào lớp của nó

---

# Slide 4 — Các bài toán thị giác ngữ nghĩa

4 of 89

**Các bài toán thị giác ngữ nghĩa**

* **Phân lớp:** có đèn giao thông?
* **Phát hiện:** người ở đâu?
* **Định danh:** có phải Potala?
* **Phân lớp khung cảnh:** Outdoor, Marketplace, City
* **Nhận dạng hành động**
* **Phân vùng ngữ nghĩa:** Trees, Mountain, Building, Vendors, People

*Nguồn hình: K. Grauman, CS376, UT Austin; I. Gkioulekas, 16-385, CMU*

---

# Slide 5 — Vì sao nhận dạng ảnh khó?

5 of 89

**Vì sao nhận dạng ảnh khó?**

* Chiếu sáng (Michelangelo 1475-1564)
* Góc nhìn
* Che khuất
* Kích thước
* Biến dạng
* Đa dạng trong lớp
* Nền phức tạp (Kilmeny Niland 1995)
* 10.000-30.000 lớp ("How many object categories are there? ~10,000 to 30,000")

*Nguồn hình: K. Grauman, CS376, UT Austin; I. Gkioulekas, 16-385, CMU*

---

# Slide 6 — Giới hạn của so khớp mẫu

6 of 89

**Giới hạn của so khớp mẫu (Template matching)**

Trượt ảnh mẫu $T$ trên ảnh $I$, tại mỗi vị trí tính tương quan chuẩn hóa (NCC):

$$NCC = \frac{\sum (I - \bar{I})(T - \bar{T})}{\sqrt{[\sum (I - \bar{I})^2 \cdot \sum (T - \bar{T})^2]}}$$

* $NCC \in [-1, 1]$; bất biến với thay đổi độ sáng tuyến tính $a I + b$
* Tốt khi đối tượng giống hệt mẫu: cùng góc nhìn, kích thước, hình dáng
* Thất bại khi đối tượng biến đổi hoặc nền phức tạp $\rightarrow$ cần đặc trưng bất biến + bộ phân lớp học từ dữ liệu
* *Đây là cái ghế* / *Hãy tìm cái ghế trong ảnh* / *Độ tương quan sau khi chuẩn hoá* / *Tìm cái ghế trong bức ảnh này*
* **Quá nhiều nhiễu**: Kỹ thuật template matching đơn giản không thể giải quyết được.

*Nguồn hình: K. Grauman, CS376, UT Austin; I. Gkioulekas, 16-385, CMU*

---

# Slide 7 — Học có giám sát (supervised learning)

7 of 89

**Học có giám sát (supervised learning)**

* Học có giám sát - supervised learning: dựa vào kiến thức biết trước.
* Sử dụng một thư viện mẫu chuẩn.
* Mẫu cần nhận dạng sẽ được so sánh với mẫu chuẩn để xác định mẫu đang đánh giá thuộc loại nào.
* Vấn đề chủ yếu là thiết kế hệ thống có thể so sánh đối tượng với mẫu nên tương đối rõ ràng về kết quả và dễ đánh giá hiệu quả

---

# Slide 8 — Học không giám sát (unsupervised learning)

8 of 89

**Học không giám sát (unsupervised learning)**

* Học không giám sát phải tự định ra các lớp khác nhau và xác định đặc trưng của từng lớp
* Khó khăn để xác định tính chính xác: số lớp, những đặc trưng của các lớp đối tượng trong kiểu học này vì không có thư viện mẫu để so sánh
* Học không giám sát được phát triển nhằm gộp các nhóm theo một cách tối ưu.

---

# Slide 9 — Cấu trúc 1 hệ thống nhận dạng

9 of 89

**Cấu trúc 1 hệ thống nhận dạng**

```
[ Khối nhận dạng: Trích chọn đặc tính biểu diễn đối tượng ] 
                      │
                      ▼
             [ Phân lớp ra quyết định ] 
                      │
                      ▼
                   [ Đánh giá ]
```

---

# Slide 10 — Huấn luyện (Tiếp cận dựa trên dữ liệu)

10 of 89

**Huấn luyện và kiểm tra bộ phân lớp**

**Huấn luyện**
```
[ Ảnh huấn luyện (có nhãn) ] ────► [ Trích chọn đặc trưng ] ────► [ Huấn luyện ] ────► [ Bộ phân lớp đã học ]
                                                                       ▲
                                                       [ Nhãn ] ───────┘
```

**Kiểm tra**
```
[ Ảnh kiểm tra (chưa từng thấy) ] ──► [ Trích chọn đặc trưng ] ──► [ Bộ phân lớp đã học ] ──► [ Dự đoán "mèo", "chó"... ]
```

* Thu thập bộ dữ liệu ảnh kèm nhãn lớp
* **Huấn luyện:** dùng một thuật toán học để xây bộ phân lớp từ (đặc trưng, nhãn)
* **Kiểm tra:** đánh giá trên ảnh không dùng khi huấn luyện; trích đặc trưng giống hệt khi huấn luyện

---

# Slide 11 — Nhận dạng dựa theo không gian

11 of 89

**Nhận dạng dựa theo không gian**

* Trong kỹ thuật này thì đối tượng được biểu diễn bằng một vector nhiều chiều.
* Mỗi chiều là một tham số thể hiện một đặc điểm của đối tượng đó
* Một số khái niệm
  * Phân hoạch không gian
  * Hàm phân lớp hoặc hàm ra quyết định

---

# Slide 12 — Trích chọn đặc trưng

12 of 89

**Trích chọn đặc trưng: Đặc trưng toàn cục và đặc trưng cục bộ**

| Tiêu chí | Đặc trưng toàn cục | Đặc trưng cục bộ |
| :--- | :--- | :--- |
| **Mô tả** | Cả ảnh / cả đối tượng như một khối | Từng vùng nhỏ quanh các điểm đặc trưng |
| **Ví dụ** | Màu (lược đồ), kết cấu (GLCM), hình dạng (mômen Hu, độ tròn), HOG | SIFT, SURF, LBP, ORB, BRISK, FREAK |
| **Số vector** | Một vector cố định cho mỗi ảnh | Hàng trăm – hàng nghìn vector, số lượng thay đổi theo ảnh |
| **Ưu điểm** | Gọn, tính nhanh, dùng ngay với bộ phân lớp | Bền với che khuất, nền phức tạp; bất biến tỉ lệ, phép quay |
| **Nhược điểm** | Nhạy với nền, che khuất, biến dạng | Phải gom lại thành một vector (VD: mô hình túi từ) |

*Đặc trưng tốt: phân biệt được các lớp, bất biến với biến đổi không quan trọng, gọn, tính nhanh.*

---

# Slide 13 — Đặc trưng màu: lược đồ

13 of 89

**Đặc trưng màu: lược đồ**

* Lược đồ mức xám hoặc lược đồ màu (VD $8 \times 8 \times 8$ bin cho R, G, B)
* Chuẩn hóa: chia cho tổng số điểm ảnh
* **Ưu điểm:**
  * Đơn giản, tính nhanh
  * Bất biến với phép dịch, quay, co giãn
  * Bền với biến dạng nhẹ
* **Nhược điểm:**
  * Bỏ qua vị trí không gian
  * Không tính tương đồng giữa các bin gần nhau
  * Bị ảnh hưởng bởi nền, ánh sáng
* **So sánh hai lược đồ:** khoảng cách $L_1$, giao lược đồ $\sum \min(h_1, h_2)$, $\chi^2$

---

# Slide 14 — Đặc trưng kết cấu: GLCM

14 of 89

**Đặc trưng kết cấu: GLCM**

* Ma trận đồng hiện mức xám (Haralick, 1973)
* Đếm số cặp điểm ảnh $(p_1, p_2)$, $p_2$ cách $p_1$ một khoảng $d$ theo hướng $\theta$ có mức xám $(i, j)$:

$$C(i,j) = \#\{(p_1, p_2) : I(p_1)=i, I(p_2)=j\}$$

* Kích thước $N_g \times N_g$ ($N_g$: số mức xám, thường giảm xuống 8, 16, 32)
* Mỗi cặp $(d, \theta)$ cho một ma trận: $\theta = 0^\circ, 45^\circ, 90^\circ, 135^\circ$; $d = 1, 2, 3...$
* Chuẩn hóa $P(i,j) = C(i,j) / \text{tổng}$

**Ví dụ:**
* Ảnh $4 \times 4$ (3 mức xám):
$$\begin{bmatrix} 0 & 0 & 1 & 1 \\ 0 & 0 & 1 & 1 \\ 0 & 2 & 2 & 2 \\ 2 & 2 & 1 & 1 \end{bmatrix}$$

* GLCM, $d=1$, $\theta=0^\circ$ (sang phải):

| $i \backslash j$ | 0 | 1 | 2 |
| :---: | :---: | :---: | :---: |
| **0** | 2 | 2 | 1 |
| **1** | 0 | 3 | 0 |
| **2** | 0 | 2 | 1 |

*Tổng 12 cặp. Kết cấu mịn, đồng nhất $\rightarrow$ giá trị lớn dồn về đường chéo; kết cấu thô, tương phản cao $\rightarrow$ giá trị trải xa đường chéo.*

---

# Slide 15 — Các đặc trưng Haralick

15 of 89

**Các đặc trưng Haralick (Rút gọn GLCM thành vài con số)**

| Đặc trưng | Công thức | Ý nghĩa | Ví dụ |
| :--- | :--- | :--- | :--- |
| **Năng lượng (energy)** | $\sum P(i,j)^2$ | Lớn khi kết cấu đều, lặp lại | 0,194 |
| **Entropy** | $-\sum P(i,j) \cdot \log_2 P(i,j)$ | Lớn khi kết cấu hỗn loạn | 2,459 |
| **Tương phản (contrast)** | $\sum (i-j)^2 \cdot P(i,j)$ | Lớn khi mức xám lân cận chênh lệch nhiều | 0,583 |
| **Đồng nhất (homogeneity)** | $\sum \frac{P(i,j)}{1+(i-j)^2}$ | Lớn khi giá trị dồn về đường chéo | 0,808 |

*Cột ví dụ: tính từ GLCM ở phần trước, $P=C/12$. Haralick đề xuất 14 đặc trưng; ghép nhiều $(d, \theta) \rightarrow$ vector kết cấu, lấy trung bình 4 hướng để gần bất biến với phép quay.*

---

# Slide 16 — Đặc trưng hình dạng

16 of 89

**Đặc trưng hình dạng (Hình học và mômen ảnh)**

* **Diện tích** $A$, **chu vi** $P$, **tỉ số dài/rộng**, **độ tròn** $C = 4\pi A / P^2$ (hình tròn: $C=1$, hình vuông: $C \approx 0,785$)
* **Mômen ảnh bậc $(p+q)$:**

$$M(p,q) = \sum_x \sum_y x^p \cdot y^q \cdot I(x,y)$$

* $M(0,0) =$ diện tích; trọng tâm $\bar{x} = M(1,0)/M(0,0)$, $\bar{y} = M(0,1)/M(0,0)$
* **Mômen trung tâm $\rightarrow$ bất biến với phép dịch:**

$$\mu(p,q) = \sum_x \sum_y (x - \bar{x})^p \cdot (y - \bar{y})^q \cdot I(x,y)$$

* **Chuẩn hóa $\rightarrow$ bất biến tỉ lệ:**

$$\eta(p,q) = \frac{\mu(p,q)}{\mu(0,0)^{1 + \frac{p+q}{2}}}$$

* **7 mômen Hu (1962) $\rightarrow$ bất biến dịch, tỉ lệ, quay:**
  * VD: $\phi_1 = \eta(2,0) + \eta(0,2)$
* **Hướng trục chính:** $\theta = \frac{1}{2} \text{atan2}(2\mu(1,1), \mu(2,0) - \mu(0,2))$

---

# Slide 17 — Đặc trưng cục bộ

17 of 89

**Đặc trưng cục bộ: Điểm đặc trưng và bộ mô tả SIFT**

* **"flat" region:** no change in all directions
* **"edge":** no change along the edge direction
* **"corner":** significant change in all directions
* **Phát hiện điểm đặc trưng:** góc Harris (dịch cửa sổ theo mọi hướng, cường độ đều thay đổi mạnh), DoG / SIFT, FAST...
* **Mô tả vùng quanh điểm:** SIFT chia vùng $16 \times 16$ thành $4 \times 4$ ô, mỗi ô lược đồ hướng gradient 8 bin $\rightarrow$ vector $4 \times 4 \times 8 = 128$ chiều
* **Bất biến tỉ lệ và phép quay** (chuẩn theo hướng chính); **bền với thay đổi chiếu sáng** (chuẩn hóa vector)
* **Ứng dụng:** so khớp ảnh, ghép ảnh toàn cảnh, nhận dạng đối tượng, mô hình túi từ

*Nguồn hình: A. Efros; D. Lowe, IJCV 2004*

---

# Slide 18 — Phân hoạch không gian

18 of 89

**Phân hoạch không gian**

* Giả sử không gian đối tượng $X$ được biểu diễn bởi $m$ vector $X_i$:

$$X = \{X_i, i = 1, 2, ..., m\}$$

* $P$ được gọi là 1 phân hoạch không gian $X$ thành các lớp $C_i, C_i \subset X$ sao cho:

$$C_i \cap C_j = \emptyset, \quad i \neq j, \quad \bigcup_{i=1}^m C_i = X$$

* Trong trường hợp thông thường thì không gian chỉ có thể được phân tách từng phần.

---

# Slide 19 — Hàm phân lớp

19 of 89

**Hàm phân lớp**

* Để phân đối tượng ra các lớp ta cần xác định số lớp và ranh giới giữa các lớp đó
* Hàm phân lớp sẽ giúp phân biệt các lớp
* Với $k$ lớp cần $k$ hàm phân lớp (tương đương $k-1$ hàm độc lập; 2 lớp chỉ cần 1 hàm)

---

# Slide 20 — Hàm phân lớp (tiếp)

20 of 89

**Hàm phân lớp**

Gọi $\{g\}$ là lớp các hàm phân lớp. Lớp hàm này được định nghĩa như sau:

Nếu $\forall i \neq k, g_k(X) > g_i(X)$ thì ta quyết định: $\Rightarrow X \in \text{lớp } k.$

Như vậy để phân biệt $k$ lớp, dùng $k$ hàm phân biệt ($k-1$ hàm độc lập).

Hàm phân lớp $g$ thường là hàm tuyến tính:

$$g(X) = W_0 + W_1 X_1 + ... + W_k X_k$$

Trong đó:
* $W_i$ là các trọng số gán cho các thành phần $X_i$.
* $W_0$ là trọng số hiệu chỉnh.

---

# Slide 21 — Hàm phân lớp (tiếp)

21 of 89

**Hàm phân lớp**

* Nếu $g$ là tuyến tính, ta nói việc phân lớp là tuyến tính hay siêu phẳng (Hyperplane).
* Các hàm phân biệt thường được xây dựng dựa trên khái niệm khoảng cách hay dựa vào xác suất có điều kiện.
* **Theo khoảng cách:** Nếu khoảng cách giữa 2 đối tượng nhỏ hơn một ngưỡng $\theta$ nào đấy thì coi 2 đối tượng là giống nhau và gộp chúng vào một lớp. Ngược lại, nếu khoảng cách lớn hơn ngưỡng, có nghĩa là chúng khác nhau và tách thành 2 lớp.

---

# Slide 22 — Thuật toán nhận dạng không gian

22 of 89

**Thuật toán nhận dạng không gian (Phân cụm và phân lớp)**

Một số thuật toán tiêu biểu:
* Thuật toán $k$ trung bình (K-means)
* Thuật toán dựa vào khoảng cách lớn nhất
* Thuật toán ISODATA
* Thuật toán Mean-Shift
* Thuật toán $k$ láng giềng (k-nearest neighbour) – học có giám sát

---

# Slide 23 — Thuật toán K-means

23 of 89

**Thuật toán nhận dạng không gian (Tự học – Unsupervised learning)**

**Thuật toán $k$ trung bình ($k$-means)**
* Khởi tạo $K$ tâm: $K$ phần tử đầu tiên hoặc chọn ngẫu nhiên ($K$ cho trước)
* Tìm $K$ cụm sao cho tổng bình phương khoảng cách từ các phần tử tới tâm cụm của chúng là nhỏ nhất
* Công thức tâm cụm ($N$: số phần tử của cụm $C_k$):

$$Z_k = \frac{1}{N} \sum_{j=1}^N X_j, \quad X_j \in C_k$$

---

# Slide 24 — Thuật toán K-means: các bước

24 of 89

**Thuật toán K-means: các bước**

1. Chọn ngẫu nhiên $K$ tâm (centroid) cho $K$ cụm (cluster). Mỗi cụm được đại diện bằng các tâm của cụm.
2. Tính khoảng cách giữa các đối tượng đến $K$ tâm (thường dùng khoảng cách Euclid).
3. Nhóm các đối tượng vào nhóm gần nhất.
4. Xác định lại tâm mới cho các nhóm.
5. Thực hiện lại bước 2 cho đến khi không có sự thay đổi nhóm nào của các đối tượng.

---

# Slide 25 — Ví dụ: K-means trên mức xám

25 of 89

**Ví dụ: K-means trên mức xám**

Dữ liệu: 6 mức xám $\{12, 18, 25, 140, 160, 200\}$; $K=2$; tâm ban đầu $=2$ phần tử đầu: 12 và 18

| Vòng | Tâm hiện tại | Cụm 1 | Cụm 2 | Tâm mới |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 12; 18 | {12} | {18, 25, 140, 160, 200} | 12; 108,6 |
| **2** | 12; 108,6 | {12, 18, 25} | {140, 160, 200} | 18,33; 166,67 |
| **3** | 18,33; 166,67 | {12, 18, 25} | {140, 160, 200} | Không đổi $\rightarrow$ dừng |

* Vòng 2: 18 cách tâm 1 là 6, cách tâm 2 là 90,6 $\rightarrow$ chuyển sang cụm 1; 25 cũng chuyển
* $J_{\text{cuối}} = (12-18,33)^2 + ... + (200-166,67)^2 \approx 1951,3$
* Ranh giới hai cụm = trung điểm hai tâm $\approx 92,5$ – K-means với $K=2$ trên mức xám chính là một cách tự tìm ngưỡng nhị phân, gần với tư tưởng Otsu.

---

# Slide 26 — K-means trong phân vùng ảnh

26 of 89

**K-means trong phân vùng ảnh**

* Mỗi điểm ảnh là một vector: mức xám (1 chiều), màu RGB (3 chiều), màu + vị trí $(x, y)$ (5 chiều)
* Phân cụm các vector $\rightarrow$ mỗi cụm là một vùng; thay mỗi điểm ảnh bằng tâm cụm của nó $\rightarrow$ ảnh còn $K$ màu
* **Các bước:**
  1. Khởi tạo tâm cụm
  2. Gán điểm vào cụm
  3. Cập nhật tâm cụm
  4. Lặp lại bước (2) và (3)
* *Minh họa kết quả trên ảnh:* Original image, $k=2, \text{obj\_func}=1634$, $k=3, \text{obj\_func}=1072$, $k=4, \text{obj\_func}=728$, $k=5, \text{obj\_func}=592$, $k=6, \text{obj\_func}=454$

---

# Slide 27 — Chọn số cụm K

27 of 89

**Chọn số cụm K - Phương pháp khuỷu tay (elbow)**

**Ví dụ 6 mức xám:**

| K | J |
| :---: | :---: |
| 1 | 34 955,5 |
| 2 | 1 951,3 |
| 3 | 284,7 |
| 6 | 0 |

* $J$ luôn giảm khi $K$ tăng ($K=N$ thì $J=0$) $\rightarrow$ không chọn $K$ theo $J$ nhỏ nhất
* Chọn $K$ tại “khuỷu tay”: sau điểm đó $J$ giảm chậm hẳn $\rightarrow$ ví dụ trên: $K=2$
* Cách khác: thử nhiều $K$ và đánh giá trên tập kiểm định; hoặc chọn $K$ theo yêu cầu ứng dụng (VD số màu cần giữ lại)

---

# Slide 28 — K-means: ưu điểm và hạn chế

28 of 89

**K-means: ưu điểm và hạn chế**

| Ưu điểm | Hạn chế |
| :--- | :--- |
| Đơn giản, dễ cài đặt | Phải chọn trước $K$ |
| Nhanh: mỗi vòng $O(K \cdot N \cdot d)$ | Hội tụ về cực tiểu cục bộ, phụ thuộc khởi tạo |
| Luôn hội tụ ($J$ không tăng) | Nhạy với điểm ngoại lai |
| Dùng làm bước trung gian: phân vùng, học từ điển (BoW) | Chỉ hợp cụm dạng cầu, kích thước tương đương |
| | Phụ thuộc thang đo của đặc trưng |

**Cải tiến thường dùng:**
* Chạy nhiều lần với khởi tạo khác nhau, giữ lần có $J$ nhỏ nhất
* **K-means++:** chọn tâm tiếp theo với xác suất tỉ lệ bình phương khoảng cách tới tâm gần nhất
* Chuẩn hóa đặc trưng trước khi phân cụm

---

# Slide 29 — Thuật toán Khoảng cách lớn nhất

29 of 89

**Thuật toán nhận dạng không gian (Tự học – Unsupervised learning)**

**Thuật toán Khoảng cách lớn nhất:** Với một tập gồm $m$ đối tượng:
* Khoảng cách giữa các đối tượng có thể được xác định như là đại lượng để phân lớp
* Khoảng cách lớn nhất ứng với phần tử xa nhất tạo nên lớp mới
* Sự phân lớp sẽ hình thành dần dần dựa vào việc tính khoảng cách giữa các đối tượng và các lớp

---

# Slide 30 — Thuật toán Khoảng cách lớn nhất: các bước

30 of 89

**Thuật toán Khoảng cách lớn nhất: các bước**

**Bước 1:**
* Chọn hạt nhân ban đầu: $X_i \in C_1$ là lớp ứng với hàm $g_1$ với $Z_1$ là phần tử trung tâm (phần tử có khoảng cách trung bình tới các phần tử khác trong $g_1$)
* Tính tất cả các khoảng cách $D_{j1} = D(X_j, Z_1)$ với $j = 1..m$
* Tìm $D_{k1}$ lớn nhất. $X_k$ là phần tử xa nhất của nhóm $g_1$, $X_k$ trở thành phần tử trung tâm $Z_2$ của lớp mới $g_2$.
* Tính $d_1 = D_{12} = D(Z_1, Z_2)$

---

# Slide 31 — Thuật toán Khoảng cách lớn nhất: các bước (tiếp)

31 of 89

**Thuật toán Khoảng cách lớn nhất: các bước**

**Bước 2:**
* Tính các khoảng cách $D_{j1}, D_{j2}$ với $j = 1..m$
* Đặt $D_k = \max_j \min(D_{j1}, D_{j2})$: khoảng cách lớn nhất từ một phần tử tới tâm gần nhất của nó
* **Điều kiện kết thúc:**
  * Nếu $D_k < \theta d_1$ kết thúc thuật toán (phân lớp xong).
  * Nếu không sẽ tạo nên lớp $g_3$ có $X_k$ là phần tử trung tâm $Z_3$.
  * Tính $d_3 = (D_{12} + D_{13} + D_{23})/3$
* Lặp đến khi thỏa đk kết thúc; cuối cùng gán mỗi phần tử vào lớp có tâm gần nhất.

---

# Slide 32 — Ví dụ: thuật toán khoảng cách lớn nhất

32 of 89

**Ví dụ: thuật toán khoảng cách lớn nhất**

$A(0;0)$, $B(1;0)$, $C(0;1)$, $D(8;8)$, $E(9;8)$, $F(0;9)$; $Z_1 = A$, $\theta = 0,5$

| Điểm | Vòng 1: $d$ tới A | Vòng 2: $\min d$ tới {A, E} | Vòng 3: $\min d$ tới {A, E, F} |
| :---: | :---: | :---: | :---: |
| **A** | 0 | 0 | 0 |
| **B** | 1 | 1 | 1 |
| **C** | 1 | 1 | 1 |
| **D** | 11,31 | 1 | 1 |
| **E** | 12,04 | 0 | 0 |
| **F** | 9 | 9 | 0 |
| **Quyết định** | $\max = 12,04 (E)$<br>$\rightarrow Z_2 = E; d_1 = 12,04$ | $9 > 0,5 \cdot 12,04 = 6,02$<br>$\rightarrow Z_3 = F; d = 10,03$ | $1 < 0,5 \cdot 10,03 = 5,02$<br>$\rightarrow$ dừng |

* **Kết quả:** $\{A, B, C\}, \{D, E\}, \{F\}$.
* Với $\theta = 0,95$: $9 < 11,44 \rightarrow$ dừng ngay, chỉ 2 lớp.
* $\theta$ là “núm vặn” số lớp; $F$ đứng riêng $\rightarrow$ thuật toán nhạy với điểm nhiễu.

---

# Slide 33 — Thuật toán ISODATA

33 of 89

**Thuật toán nhận dạng không gian (Tự học – Unsupervised learning)**

**Thuật toán ISODATA:**
* ISODATA: Iterative Self Organizing Data Analysis
* Thuật toán này tương đối mềm dẻo và không cố định các lớp

---

# Slide 34 — Thuật toán ISODATA: các bước

34 of 89

**Thuật toán ISODATA: các bước**

* Lựa chọn phân hoạch dựa vào các tâm bất kỳ (ít phụ thuộc vào tâm ban đầu hơn K-means)
* Phân vùng bằng cách đặt các điểm vào lớp có tâm gần nhất dựa vào khoảng cách
* Tách đôi lớp có độ lệch chuẩn lớn hơn ngưỡng $t_1$
* Xác định phân hoạch mới dựa vào các tâm vừa xác định cho đến khi không có tâm mới
* Gộp hai lớp có khoảng cách giữa hai tâm nhỏ hơn ngưỡng $t_2$
* Lặp các bước trên cho đến khi thỏa mãn tiêu chuẩn phân hoạch.

---

# Slide 35 — Mean-Shift

35 of 89

**Mean-Shift**

* Phân cụm không cần biết trước $K$
* **Ý tưởng:** cụm = vùng quanh một đỉnh (mode) của mật độ dữ liệu
* **Các bước:**
  * B1: đặt cửa sổ bán kính $h$ tại một điểm
  * B2: tính trọng tâm (mean) các điểm trong cửa sổ
  * B3: dịch cửa sổ tới trọng tâm (vector mean-shift)
  * B4: lặp tới khi hội tụ; các điểm hội tụ về cùng đỉnh thuộc một cụm

**Ví dụ 1 chiều:** dữ liệu $\{1, 2, 3, 7, 8\}$, $h=2$
* Bắt đầu tại 4: cửa sổ $[2; 6]$ chứa $\{2, 3\} \rightarrow 2,5 \rightarrow [0,5; 4,5]$ chứa $\{1, 2, 3\} \rightarrow 2 \rightarrow$ không đổi
* Bắt đầu tại 6: cửa sổ $[4; 8]$ chứa $\{7, 8\} \rightarrow 7,5 \rightarrow$ không đổi
* $\rightarrow 2$ đỉnh (2 và 7,5) $\Rightarrow 2$ cụm, không cần cho trước $K$

*Nguồn hình: Thị giác máy tính – Bài 10: Phân đoạn ảnh (theo Y. Ukrainitz & B. Sarel)*

---

# Slide 36 — Mean-Shift: ưu điểm và nhược điểm

36 of 89

**Mean-Shift: vùng hút và ứng dụng**

| Ưu điểm | Nhược điểm |
| :--- | :--- |
| Không cần biết trước số cụm | Kết quả phụ thuộc bán kính $h$; chọn $h$ không dễ |
| Cụm có hình dạng bất kỳ | Tính toán chậm (mỗi bước duyệt nhiều điểm) |
| Ít bị ảnh hưởng bởi điểm ngoại lai | Khó mở rộng khi số chiều đặc trưng lớn |
| Chỉ một tham số $h$ | |

---

# Slide 37 — Thuật toán k láng giềng (kNN)

37 of 89

**Thuật toán nhận dạng không gian (Học có giám sát - Supervised learning)**

**Thuật toán $k$ láng giềng:**
* Thuật toán này dựa vào vị trí của đối tượng cần xét để xác định lớp mà đối tượng đó thuộc về là lớp nào
* Các mẫu sẽ được biểu diễn trên không gian đối tượng
* Mỗi đối tượng cần xác định lớp sẽ tìm $k$ mẫu gần nhất xung quanh
* Đối tượng đầu vào này sẽ thuộc lớp có số lượng lớn nhất trong số $k$ láng giềng

---

# Slide 38 — Thuật toán k láng giềng (tiếp)

38 of 89

**Thuật toán $k$ láng giềng:**

* Vì thuật toán chọn số đông để quyết định việc phân lớp nên thông thường lớp lớn nhất trong tập mẫu thường có xu hướng thống trị
* Để làm giảm việc này thì trọng số liên quan đến khoảng cách có thể được thêm vào, và mẫu càng xa sẽ có ảnh hưởng ít hơn so với mẫu ở gần

---

# Slide 39 — Ví dụ: kNN phân loại hạt gạo

39 of 89

**Ví dụ: kNN phân loại hạt gạo**

Đặc trưng: (chiều dài, chiều rộng) mm; hạt cần nhận dạng $Q = (5,0; 2,4)$

| Thứ hạng | Mẫu | (dài; rộng) | Lớp | Khoảng cách tới Q |
| :---: | :---: | :---: | :---: | :---: |
| **1** | G2 | (4,0; 2,0) | gãy | 1,08 |
| **2** | N3 | (6,2; 3,0) | nguyên | 1,34 |
| **3** | G1 | (3,5; 1,8) | gãy | 1,62 |
| **4** | N1 | (6,5; 3,2) | nguyên | 1,70 |
| **5** | G3 | (3,2; 1,5) | gãy | 2,01 |
| **6** | N2 | (6,8; 3,4) | nguyên | 2,06 |

* $k=1$: G2 $\rightarrow$ gãy
* $k=3$: 2 gãy, 1 nguyên $\rightarrow$ gãy
* $k=2$: 1-1 hòa với 2 lớp nên chọn $k$ lẻ
* $k=5$: 3 gãy, 2 nguyên $\rightarrow$ gãy

*Chuẩn hóa đặc trưng nếu các chiều có đơn vị / thang đo khác nhau.*

---

# Slide 40 — KNN: độ đo khoảng cách

40 of 89

**KNN: độ đo khoảng cách**

| Độ đo | Công thức | Thường dùng khi |
| :--- | :--- | :--- |
| **Euclid ($L_2$)** | $d = \sqrt{\sum (x_i - y_i)^2}$ | Đặc trưng liên tục, cùng thang đo |
| **Manhattan ($L_1$)** | $d = \sum \|x_i - y_i\|$ | Ít nhạy với chênh lệch lớn ở một chiều |
| **Minkowski ($L_p$)** | $d = (\sum \|x_i - y_i\|^p)^{1/p}$ | Tổng quát: $p=1 \rightarrow L_1, p=2 \rightarrow L_2$ |
| **Cosine** | $d = 1 - \frac{x \cdot y}{\|x\| \cdot \|y\|}$ | So sánh hướng |
| **Chi bình phương ($\chi^2$)** | $d = \frac{1}{2} \sum \frac{(x_i - y_i)^2}{x_i + y_i}$ | Vector tần suất (BoW), so sánh lược đồ |

* **Chuẩn hóa trước khi tính khoảng cách:**
  * **Min-max:** $x' = (x - \min) / (\max - \min) \rightarrow [0; 1]$
  * **Z-score:** $x' = (x - \mu) / \sigma \rightarrow$ trung bình 0, độ lệch chuẩn 1

---

# Slide 41 — Chọn k: kiểm định chéo

41 of 89

**Chọn k: kiểm định chéo**

* Siêu tham số ($k$, độ đo khoảng cách) không học được từ tập huấn luyện
* **Chia dữ liệu:** huấn luyện / kiểm định / kiểm tra (VD 70/15/15); tập kiểm tra chỉ dùng một lần ở cuối
* **Kiểm định chéo 5 phần (5-fold Cross-validation):** lần lượt lấy 1 phần làm kiểm định, 4 phần huấn luyện; lấy trung bình 5 lần
  * Fold 1, Fold 2, Fold 3, Fold 4, Fold 5 $\rightarrow$ Test data
* **Biên quyết định:** 1-NN (nhạy nhiễu) vs 5-NN (trơn hơn)

---

# Slide 42 — kNN: chi phí, ưu và nhược điểm

42 of 89

**kNN: chi phí, ưu và nhược điểm**

$N$ mẫu huấn luyện, $M$ mẫu kiểm tra, $d$ chiều:
* **Huấn luyện:** $O(1)$ (chỉ lưu dữ liệu)
* **Nhận dạng:** $O(M \cdot N \cdot d)$

**So sánh kết quả trên MNIST (chữ số viết tay $28 \times 28$):**

| Phương pháp | Lỗi (%) |
| :--- | :---: |
| Phân lớp tuyến tính | 12,0 |
| KNN, Euclid | 5,0 |
| KNN, Euclid, ảnh chỉnh nghiêng | 2,4 |
| KNN, khoảng cách tiếp tuyến | 1,1 |
| KNN, so khớp shape context | 0,67 |
| SVM đa thức bậc 4 | 1,1 |
| CNN LeNet-5 | 0,8 |

| Ưu điểm | Nhược điểm |
| :--- | :--- |
| Đơn giản, không cần huấn luyện | Nhận dạng chậm, tốn bộ nhớ |
| Hiệu quả tốt nếu đặc trưng tốt | Kém khi số chiều lớn |

*Tăng tốc: cây k-d, tìm láng giềng xấp xỉ.*  
*Nguồn: Y. LeCun, MNIST database (60 000 ảnh huấn luyện, 10 000 ảnh kiểm tra); trích từ Thị giác máy tính – Bài 8*

---

# Slide 43 — Mô hình túi từ (Bag-of-Words)

43 of 89

**Mô hình túi từ (Bag-of-Words)**

**Ý tưởng từ xử lý văn bản:**
* Văn bản = vector đếm số lần xuất hiện của mỗi từ trong từ điển (bỏ qua thứ tự từ)
* **Độ tương đồng hai văn bản:** cosine $\cos(h_1, h_2) = \frac{h_1 \cdot h_2}{\|h_1\| \cdot \|h_2\|}$
* Từ phổ biến (“và”, “của”) ít mang thông tin $\rightarrow$ **trọng số TF-IDF:**

$$w(t,d) = tf(t,d) \cdot \log(N / n_t)$$

| Từ | $tf$ trong văn bản $d$ | Số văn bản chứa từ ($N=1000$) | $idf = \log_{10}(N/n_t)$ | $tf \cdot idf$ |
| :--- | :---: | :---: | :---: | :---: |
| "và" | 20 | 1000 | 0 | 0 |
| "robot" | 5 | 10 | 2 | 10 |

---

# Slide 44 — Túi từ trực quan cho ảnh

44 of 89

**Túi từ trực quan cho ảnh**

* Một đối tượng (ví dụ: xe DUCATI) $\rightarrow$ Tập hợp các đặc trưng cục bộ (bag-of-features)
  * Giải quyết tốt vấn đề che khuất
  * Bất biến với kích thước
  * Bất biến với phép xoay

| Văn bản | Ảnh |
| :--- | :--- |
| **Từ** | Đặc trưng cục bộ đã lượng tử hóa = “từ trực quan” (visual word) |
| **Từ điển** | Bộ từ điển trực quan (codebook): $K$ tâm cụm học bằng K-means |
| **Vector đếm từ** | Lược đồ $K$ chiều: số đặc trưng rơi vào mỗi visual word |

* **Ưu:** bền với che khuất, bất biến tỉ lệ & phép quay (nhờ đặc trưng cục bộ)
* **Nhược:** mất bố cục không gian $\rightarrow$ khắc phục: chia ảnh thành lưới (spatial pyramid)

*Nguồn hình: Thị giác máy tính – Bài 8: Phân loại ảnh (theo K. Grauman, CS376, UT Austin; I. Gkioulekas, 16-385, CMU)*

---

# Slide 45 — Quy trình mô hình túi từ trực quan

45 of 89

**Quy trình mô hình túi từ trực quan**

1. **Học từ điển:** Trích SIFT từ tập ảnh huấn luyện; K-means $\rightarrow K$ visual word. Từ điển trực quan (mỗi ô: một tâm cụm).
2. **Mã hóa:** Gán mỗi đặc trưng vào tâm gần nhất; đếm $\rightarrow$ lược đồ $K$ chiều.
3. **Phân lớp:** KNN, SVM, Naive Bayes trên vector BoW.

*$K$ thường từ vài trăm đến vài nghìn; truy vấn CSDL lớn: hàng trăm nghìn – hàng triệu từ.*

---

# Slide 46 — Ví dụ: mã hóa ảnh thành vector BoW

46 of 89

**Ví dụ: mã hóa ảnh thành vector BoW**

Từ điển 4 từ (đặc trưng 2 chiều): $w_1(0;0), w_2(4;0), w_3(0;4), w_4(4;4)$

| Đặc trưng của ảnh 1 | (0,5; 0,2) | (3,6; 0,4) | (0,3; 3,9) | (4,2; 3,7) | (3,8; 4,1) | (0,1; 0,6) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Từ gần nhất** | $w_1$ | $w_2$ | $w_3$ | $w_4$ | $w_4$ | $w_1$ |

| Ảnh | Lược đồ $h$ | Cosine với ảnh 1 |
| :--- | :---: | :---: |
| **Ảnh 1** | (2; 1; 1; 2) | 1 |
| **Ảnh 2** | (0; 1; 1; 4) | $10 / (\sqrt{10} \cdot \sqrt{18}) \approx 0,745$ |
| **Ảnh 3** | (3; 1; 1; 1) | $10 / (\sqrt{10} \cdot \sqrt{12}) \approx 0,913$ |

*Ảnh 3 giống ảnh 1 hơn ảnh 2 $\rightarrow$ nếu ảnh 3 có nhãn “mô tô”, 1-NN gán ảnh 1 là “mô tô”. Số đặc trưng mỗi ảnh khác nhau nhưng lược đồ luôn có $K=4$ chiều.*

---

# Slide 47 — Cây từ vựng

47 of 89

**Cây từ vựng (Vocabulary tree - Nistér & Stewénius, CVPR 2006)**

* K-means phân cấp: chia dữ liệu thành $b$ cụm, mỗi cụm lại chia $b$ cụm... sâu $L$ mức
* Số visual word $= b^L$; VD $b=10, L=6 \rightarrow 1\,000\,000$ từ
* Gán một đặc trưng: đi từ gốc xuống lá, mỗi mức so với $b$ tâm $\rightarrow$ chỉ $b \cdot L = 60$ phép so sánh thay vì $1\,000\,000$
* Kết hợp chỉ mục ngược (inverted index): mỗi từ lưu danh sách ảnh chứa nó $\rightarrow$ truy vấn CSDL hàng triệu ảnh

---

# Slide 48 — Đánh giá: ma trận nhầm lẫn

48 of 89

**Đánh giá: ma trận nhầm lẫn (Confusion matrix)**

| | Dự đoán: dương | Dự đoán: âm |
| :--- | :---: | :---: |
| **Thực tế: dương** | TP (dương thật) | FN (âm giả – bỏ sót) |
| **Thực tế: âm** | FP (dương giả – báo nhầm) | TN (âm thật) |

* Lớp “dương” = lớp ta quan tâm phát hiện (VD hạt gãy, khối u, khuôn mặt)
* Nhiều lớp: ma trận $C \times C$, ô $(i,j) =$ số mẫu lớp $i$ bị dự đoán là lớp $j$; đường chéo = dự đoán đúng
* Ô ngoài đường chéo cho biết lớp nào hay bị nhầm với lớp nào (VD chữ số $4 \leftrightarrow 9$)

---

# Slide 49 — Accuracy, Precision, Recall, F1

49 of 89

**Accuracy, Precision, Recall, F1**

| Độ đo | Công thức | Trả lời câu hỏi |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | Đoán đúng bao nhiêu phần trăm? |
| **Precision** | $\frac{TP}{TP+FP}$ | Trong các mẫu báo dương, bao nhiêu là đúng? |
| **Recall** | $\frac{TP}{TP+FN}$ | Trong các mẫu dương thật, tìm được bao nhiêu? |
| **F1** | $\frac{2 \cdot P \cdot R}{P+R}$ | Trung bình điều hòa của precision và recall |

* Dữ liệu lệch lớp $\rightarrow$ accuracy dễ đánh lừa: dùng precision, recall, F1
* Tăng ngưỡng báo dương $\rightarrow$ precision thường tăng, recall giảm (và ngược lại)
* Recall còn gọi là độ nhạy (sensitivity, TPR)

---

# Slide 50 — Thuật toán nhận dạng theo cấu trúc

50 of 89

**Thuật toán nhận dạng theo cấu trúc**

* Ngoài cách biểu diễn định lượng thì còn tồn tại kiểu đối tượng định tính
* Ví dụ mối quan hệ giữa các đối tượng hoặc dạng của đối tượng
* Giả sử đối tượng có thể được biểu diễn bởi một chuỗi ký tự và mỗi ký tự sẽ thể hiện một đặc tính
* Hàm phân biệt là hàm logic kiểm tra chuỗi có được sinh ra bởi tập quy tắc (văn phạm) của lớp hay không

---

# Slide 51 — Thuật toán nhận dạng theo cấu trúc (tiếp)

51 of 89

**Thuật toán nhận dạng theo cấu trúc**

Có hai giai đoạn trong quy trình nhận diện:
* Xác định quy tắc xây dựng
* Xác định các dạng dựa vào các quy tắc đó

In hai giai đoạn thì xác định quy tắc xây dựng là rất khó khăn và là vấn đề chính cần giải quyết trong quy trình nhận diện dạng này.

---

# Slide 52 — Nhận diện dựa trên mạng Nơron

52 of 89

**Nhận diện dựa trên mạng Nơron**

* Mạng nơron bao gồm nhiều phần tử xử lý đơn giản hoạt động song song.
* Tính năng của hệ thống phụ thuộc vào cấu trúc liên kết giữa các nơron và trọng số của các liên kết này.
* Mạng nơron có thể học từ dữ liệu mẫu và tổng quát hóa dựa trên các dữ liệu mẫu học. Bản chất quá trình học là việc hiệu chỉnh các tham số của các nơron để đầu mạng dự đoán đúng dữ liệu học.
* Trong mạng nơron, các nơron đón nhận tín hiệu vào gọi là nơron vào và các nơron đưa thông tin ra gọi là nơron ra.

---

# Slide 53 — Mạng Hopfield

53 of 89

**Mạng Nơron: Mạng Hopfield**

* Mạng Hopfield là mạng nơron 1 lớp
* Ánh xạ dữ liệu tín hiệu vào sang tín hiệu ra theo kiểu tự kết hợp
* Nếu tín hiệu vào là $X$ thuộc miền $D$ thì tín hiệu ra $Y$ cũng thuộc miền $D$
* Mạng Hopfield mô phỏng khả năng hồi tưởng của não người (nhận ra người quen khi nhận ra các nét quen trên khuôn mặt)

---

# Slide 54 — Mạng Hopfield (tiếp)

54 of 89

**Mạng Nơron: Mạng Hopfield**

* Mạng Hopfield có một lớp ra có kích thước bằng kích thước tín hiệu vào, liên kết nơron là đầy đủ.
* Mạng Hopfield yêu cầu tín hiệu vào có giá trị lưỡng cực -1 và 1.

```
Tín hiệu Vào ──────► [ Nơ ron thứ i ] ──────► Tín hiệu Ra
       ▲                                            │
       └────────────────────────────────────────────┘ (Liên kết phản hồi đầy đủ)
```

---

# Slide 55 — Mạng Hopfield (tiếp)

55 of 89

**Mạng Nơron: Mạng Hopfield**

Hàm kích hoạt tại các nơron là hàm dấu:

$$out_j = \text{sign}(Net_j) = \text{sign}\left(\sum_{i=1}^m w_{ji} x_i\right)$$

$w_{ij}$ là trọng số của liên kết giữa nơron thứ $i$ với nơron thứ $j$.

---

# Slide 56 — Mạng Hopfield (tiếp)

56 of 89

**Mạng Nơron: Mạng Hopfield**

**Học theo luật Hebb từ $p$ mẫu cho trước:**
* Giả sử có $p$ mẫu học tương ứng với các vector tín hiệu vào $X_s, s=1..p$
* Mạng cần xác định bộ trọng số $W$ sao cho $X_s = f(X_s, W)$ với mọi $s=1..p$
* Ta xây dựng ma trận trọng số $W$ như sau:

$$w_{ij} = \begin{cases} \sum_{s=1}^p x_{sj} x_{si}, & i \neq j \\ 0, & i = j \end{cases}$$

---

# Slide 57 — Mạng Hopfield: Hồi tưởng (nhận dạng)

57 of 89

**Mạng Nơron: Mạng Hopfield (Hồi tưởng - nhận dạng)**

* Giả sử đưa vào mạng vector tín hiệu $X$
* Việc tính toán đầu ra $Y$ cho tín hiệu $X$ là quá trình lặp lại:
  * Đặt $X_0 = X$
  * Tính $Y_t$ là tín hiệu đầu ra tương ứng với $X_t$ lan truyền trong mạng một lần
  * Nếu $Y_t$ khác $X_t$ thì tiếp tục với $t = t+1$ và $X_{t+1} = Y_t$
  * Nếu $Y_t = X_t$ thì dừng lại và kết quả của mạng là $Y_t$
* $W$ không thay đổi trong quá trình tính $Y$.

---

# Slide 58 — Ví dụ: hồi tưởng với mạng Hopfield

58 of 89

**Ví dụ: hồi tưởng với mạng Hopfield**

Mẫu đã lưu $X^1 = (1, -1, 1, -1)$; $w_{ij} = x^1_i \cdot x^1_j$ ($i \neq j$), $w_{ii} = 0$

**Ma trận $W$:**

$$\begin{bmatrix} 0 & -1 & 1 & -1 \\ -1 & 0 & -1 & 1 \\ 1 & -1 & 0 & -1 \\ -1 & 1 & -1 & 0 \end{bmatrix}$$

**Cập nhật đồng bộ:**

| $t$ | $X_t$ | $Net = W \cdot X_t$ | $Y_t = \text{sign}(Net)$ |
| :---: | :---: | :---: | :---: |
| **0** | $(1, 1, 1, -1)$ | $(1, -3, 1, -1)$ | $(1, -1, 1, -1)$ |
| **1** | $(1, -1, 1, -1)$ | $(3, -3, 3, -3)$ | $(1, -1, 1, -1)$ |

* $Y_1 = X_1 \rightarrow$ dừng; điểm ảnh thứ 2 bị nhiễu đã được sửa. Nơron 2 nhận lại dứt khoát $Net = -3$: cả 3 nơron còn lại cùng “bỏ phiếu” $-1 \rightarrow$ lật.
* Cập nhật không đồng bộ (mỗi lần một nơron) luôn hội tụ khi $W$ đối xứng, $w_{ii} = 0$; cập nhật đồng bộ có thể dao động.

---

# Slide 59 — Hàm năng lượng và sức chứa mạng Hopfield

59 of 89

**Mạng Hopfield: Hàm năng lượng và sức chứa**

* **Hàm năng lượng:**

$$E(X) = -\frac{1}{2} \sum_i \sum_j w_{ij} \cdot x_i \cdot x_j$$

* Cập nhật không đồng bộ: mỗi lần lật một nơron theo hàm dấu $\rightarrow E$ không tăng $\rightarrow$ mạng luôn dừng ở một cực tiểu.
* Ví dụ trên: $E(1,1,1,-1) = 0 \rightarrow E(1,-1,1,-1) = -6$
* Mẫu đã học = cực tiểu (điểm hút); trạng thái giả = cực tiểu không mong muốn; $-X$ cũng là cực tiểu.
* **Sức chứa (luật Hebb, mẫu ngẫu nhiên):**
  * Khoảng $0,138 N$ mẫu; $N=100$ nơron $\rightarrow$ khoảng 13 mẫu
  * Mẫu càng giống nhau (tương quan cao) $\rightarrow$ sức chứa càng giảm
  * Số trọng số tăng theo $N^2$: ảnh $10 \times 10 \rightarrow 4\,950$ trọng số độc lập

---

# Slide 60 — Tình huống nảy sinh trong huấn luyện mạng Hopfield

60 of 89

**Mạng Nơron: Mạng Hopfield**  
*Một số tình huống nảy sinh trong huấn luyện mạng*

* **Mạng không dừng:** Mạng có thể đưa ra luân phiên một vài mẫu học hoặc ảnh ngược của chúng.
* **Mạng dừng và $X_t = X$:** $X$ đã được đoán đúng dựa trên mẫu học, $X$ có thể là một trong các mẫu đã học.

---

# Slide 61 — Tình huống nảy sinh (tiếp)

61 of 89

**Mạng Nơron: Mạng Hopfield**  
*Một số tình huống nảy sinh trong huấn luyện mạng*

* **Mạng dừng với $X_t$ là $X_s$ nào đó trong mẫu đã học:** Mạng đã phục hồi nguyên dạng của $X$ chính là $X_s$.
* **Mạng dừng với $X_t$ không thuộc mẫu đã học:** Chỉ ra một vector mới có thể dùng để cập nhật trọng số.
* **Mạng dừng với $X_t$ là ảnh ngược ở trên.**

---

# Slide 62 — Mạng Kohonen

62 of 89

**Mạng Nơron: Mạng Kohonen**  
*(Bản đồ Kohonen hoặc bản đồ tự tổ chức - SOM)*

* Mạng Kohonen tập trung vào mối liên hệ có tính cấu trúc trong các vùng lân cận hoặc trong toàn thể không gian mẫu.
* Trong mạng Kohonen các vector tín hiệu vào gần nhau sẽ được ánh xạ sang các nơron trong mạng lân cận nhau.

---

# Slide 63 — Mạng Kohonen (tiếp)

63 of 89

**Mạng Nơron: Mạng Kohonen**

* Mạng Kohonen rất gần với mạng sinh học về cấu tạo lẫn cơ chế học.
* Mạng Kohonen có một lớp kích hoạt là các nơron được phân bố trong mặt phẳng hai chiều kiểu lưới vuông hoặc lục giác.
* Mỗi nơron bên trong lưới có cùng số láng giềng; các đầu vào tương tự sẽ kích hoạt các nơron gần nhau (có thể cuộn lưới thành hình xuyến để bỏ hiệu ứng biên).

---

# Slide 64 — Cấu trúc lưới nơron Kohonen

64 of 89

**Mạng Nơron: Mạng Kohonen**  
*Bản đồ Kohonen hoặc bản đồ tự tổ chức*

Lưới các nơron trong mặt phẳng 2 chiều (Lưới vuông 4 láng giềng vs. Lưới lục giác 6 láng giềng).

---

# Slide 65 — Mạng Kohonen (tiếp)

65 of 89

**Mạng Nơron: Mạng Kohonen**

* Tất cả các nơron ở lớp kích hoạt được nối đầy đủ với lớp vào và mỗi kết nối này sẽ có một trọng số.
* Các nơron trên lớp kích hoạt chỉ liên hệ với các nơron lân cận nên khi có tín hiệu đầu vào sẽ chỉ tạo ra kích hoạt địa phương.
* Vùng được kích hoạt sẽ chỉ ra vị trí của tín hiệu đầu vào trong không gian đối tượng.

---

# Slide 66 — Học cạnh tranh trong mạng Kohonen

66 of 89

**Mạng Nơron: Mạng Kohonen**

* Việc học của mạng Kohonen dựa trên kỹ thuật cạnh tranh không cần mẫu học.
* Trọng số của các nơron được khởi tạo bằng một số nhỏ, bất kỳ.
* Mạng được học qua một lượng lớn mẫu gần nhau và thường nhiều lần theo các vòng lặp.
* Tại một thời điểm chỉ có một nơron duy nhất $C$ trong lớp kích hoạt được lựa chọn với nguyên tắc là nơron $C$ có vector trọng số gần với tín hiệu vào nhất.

---

# Slide 67 — Cập nhật trọng số mạng Kohonen

67 of 89

**Mạng Nơron: Mạng Kohonen**

* Sau khi có nơron $C$, các trọng số $w_{Ci}$ được hiệu chỉnh nhằm cho gần với trọng số mong muốn (đầu vào).
* Trọng số được kéo về phía tín hiệu vào: thành phần $w_{Ci}$ lớn hơn $x_{Si}$ thì giảm, nhỏ hơn thì tăng, tỉ lệ với độ chênh lệch.

---

# Slide 68 — Công thức cập nhật trọng số SOM

68 of 89

**Mạng Nơron: Mạng Kohonen**

Tỉ lệ thay đổi trọng số nhỏ dần theo thời gian ($\alpha(s)$) và theo khoảng cách trên lưới giữa nơron $v$ và nơron thắng $u$ ($\theta(u, v, s)$):

$$w_v(s+1) = w_v(s) + \theta(u, v, s) \alpha(s) (D(t) - w_v(s))$$

Với:
* $s$ là thứ tự bước
* $D(t)$ là vector đầu vào $t$
* $u$ là số thứ tự của nơron $C$ được chọn
* $\theta(u, v, s)$ là hàm tỷ lệ khoảng cách giữa nơron $v$ và $u$ (có thể là hàm ngưỡng hoặc hàm Gaussian)

---

# Slide 69 — Ví dụ: một bước cập nhật SOM

69 of 89

**Ví dụ: một bước cập nhật SOM**

Đầu vào $x = (0,8; 0,2)$; $\alpha = 0,5$; nơron thắng A; B là láng giềng của A trên lưới.

| Nơron | $w$ cũ | $\|x - w\|^2$ | $\theta(u, v)$ | $\alpha \cdot \theta$ | $w$ mới $= w + \alpha \cdot \theta \cdot (x - w)$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A (thắng)** | $(0,5; 0,5)$ | 0,18 | 1 | 0,5 | $(0,65; 0,35)$ |
| **B (láng giềng)** | $(0,4; 0,6)$ | 0,32 | 0,5 | 0,25 | $(0,50; 0,50)$ |
| **C (ở xa)** | $(0,1; 0,9)$ | 0,98 | 0 | 0 | $(0,1; 0,9)$ |

* A tiến về $x$ một nửa quãng đường; B tiến theo một phần tư; C đứng yên.
* Sau cập nhật: $\|x - A\|^2$ giảm từ 0,18 xuống 0,045 $\rightarrow$ lần sau A thắng dứt khoát hơn.
* B “kế thừa” vị trí cũ của A $\rightarrow$ các nơron cạnh nhau mang trọng số giống nhau $\rightarrow$ bản đồ hình thành.

---

# Slide 70 — Thuật toán huấn luyện SOM

70 of 89

**Thuật toán huấn luyện SOM**

1. Khởi tạo ngẫu nhiên (nhỏ) các vector trọng số $w_v$
2. Lấy một mẫu $D(t)$ từ tập dữ liệu
3. Tìm nơron thắng: $u = \arg\min_v \|D(t) - w_v\|$
4. Cập nhật mọi nơron: $w_v \leftarrow w_v + \theta(u, v, s) \cdot \alpha(s) \cdot (D(t) - w_v)$
5. Giảm $\alpha(s)$ và bán kính $\sigma(s)$; lặp lại từ bước 2

| Thành phần | Dạng thường dùng |
| :--- | :--- |
| **Hàm lân cận** | $\theta(u, v, s) = \exp\left(-\frac{\|r_u - r_v\|^2}{2\sigma(s)^2}\right)$, $r$: tọa độ nơron trên lưới |
| **Tốc độ học** | $\alpha(s) = \alpha_0 \cdot \exp(-s/T)$, VD $\alpha_0 = 0,5$ |
| **Bán kính** | $\sigma(s) = \sigma_0 \cdot \exp(-s/T)$, $\sigma_0 \approx$ nửa kích thước lưới |

---

# Slide 71 — Mạng Kohonen (tiếp)

71 of 89

**Mạng Nơron: Mạng Kohonen**

* Mạng Kohonen được huấn luyện dựa vào sự kích hoạt địa phương và các nơron lân cận nên các đối tượng gần nhau sẽ kích hoạt các nơron gần nhau.
* Khi chuyển một tín hiệu đầu vào vào mạng thì một vùng nơron sẽ được kích hoạt cho chúng ta thấy tín hiệu đầu vào này thuộc vùng dữ liệu nào.

---

# Slide 72 — Từ mạng nơ ron cổ điển đến CNN

72 of 89

**Từ mạng nơ ron cổ điển đến CNN**

* **Đặc trưng thủ công** (GLCM, Hu, SIFT, BoW) $\rightarrow$ **đặc trưng học tự động** từ dữ liệu.
* **Lớp tích chập:**
  * Bộ lọc $F \times F \times C$ trượt trên ảnh (như lọc không gian ở chương 3) + hàm kích hoạt ReLU
  * Kích thước ra: $(N - F + 2P)/S + 1$
  * VD ảnh $32 \times 32 \times 3$, 6 bộ lọc $5 \times 5$, $P=0, S=1 \rightarrow 28 \times 28 \times 6$; số tham số $6 \times (5 \cdot 5 \cdot 3 + 1) = 456$
* **Lớp pooling:**
  * Max / average trên cửa sổ $2 \times 2$, bước $2 \rightarrow$ giảm một nửa kích thước ($28 \rightarrow 14$)
* **Huấn luyện:**
  * Lan truyền ngược trên hàng chục nghìn – hàng triệu ảnh có nhãn
  * Học chuyển giao: tinh chỉnh mạng đã huấn luyện sẵn cho bài toán mới, ít dữ liệu

---

# Slide 73 — So sánh các phương pháp

73 of 89

**So sánh các phương pháp**

| Phương pháp | Kiểu học | Ý tưởng chính | Lưu ý |
| :--- | :--- | :--- | :--- |
| **K-means** | Không giám sát | Gán điểm – tính lại tâm (Cho trước $K$) | Nhạy khởi tạo, ngoại lai |
| **Khoảng cách lớn nhất** | Không giám sát | Phần tử xa nhất $\rightarrow$ tâm mới (Theo ngưỡng $\theta$) | Nhạy điểm nhiễu |
| **ISODATA** | Không giám sát | Tách / gộp ($K\text{-means} + \text{tách} + \text{gộp}$) | Nhiều tham số |
| **Mean-Shift** | Không giám sát | Dịch cửa sổ tới đỉnh mật độ (Theo bán kính $h$) | Chậm, phụ thuộc $h$ |
| **KNN** | Có giám sát | Theo nhãn mẫu (Bỏ phiếu $k$ láng giềng) | Nhận dạng chậm |
| **Cấu trúc** | Theo quy tắc | Chuỗi ký hiệu + phân tích cú pháp (Theo văn phạm) | Khó xây quy tắc |
| **Hopfield** | Luật Hebb | Bộ nhớ liên kết, lặp tới ổn định (Số mẫu lưu) | Sức chứa $\approx 0,138 N$ |
| **Kohonen (SOM)** | Không giám sát | Học cạnh tranh + láng giềng (Kích thước lưới) | Bảo toàn quan hệ lân cận |

---

# Slide 74 — Tóm tắt

74 of 89

**Tóm tắt**

* Nhận dạng ảnh: bài toán, thách thức, hệ thống nhận dạng
* Học có giám sát và không giám sát
* Trích chọn đặc trưng: màu, kết cấu, hình dạng, cục bộ
* Nhận dạng dựa theo không gian
  * K-means, khoảng cách lớn nhất, ISODATA, Mean-Shift
  * k láng giềng (kNN), mô hình túi từ, đánh giá
* Nhận dạng theo cấu trúc
* Một số mô hình mạng nơ ron
  * Mạng Hopfield, mạng Kohonen, hướng tới CNN