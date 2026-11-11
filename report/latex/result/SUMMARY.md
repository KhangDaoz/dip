# Kết quả cho báo cáo

Sinh bởi `code/tools/export_report_results.py` lúc 11/11/2026 17:03, commit `08a8d4b`. Mọi số liệu trên **176 ảnh kiểm tra** trừ khi ghi khác. Chạy lại script để cập nhật.

## Dữ liệu (đáp án khoản thuốc, từ điển)

- Tập kiểm tra: 176 ảnh (176 ảnh đã soát tay), 443 khoản thuốc, 98 tên thuốc khác nhau.
- Từ điển tên thuốc (tập huấn luyện): 138 tên.
- Ngoài từ điển (OOV): 2 khoản (0.5 %), 1 tên (1.0 %).

## E-A2 — Sai số chỉnh nghiêng (khối A)

| abs_angle | n | median | p90 | max | pct_le_0_5 | sec_per_img |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 176 | 0.0723 | 0.1357 | 0.2611 | 100.0 | 0.0775 |
| 1.0 | 176 | 0.0564 | 0.1169 | 0.1501 | 100.0 | 0.0771 |
| 2.0 | 176 | 0.0614 | 0.1101 | 0.1351 | 100.0 | 0.0764 |
| 3.0 | 176 | 0.0582 | 0.1112 | 0.1531 | 100.0 | 0.0765 |
| 5.0 | 176 | 0.0660 | 0.1124 | 0.1471 | 100.0 | 0.0768 |
| 7.0 | 176 | 0.0654 | 0.1113 | 0.1329 | 100.0 | 0.0769 |
| 10.0 | 176 | 0.0595 | 0.1140 | 0.1316 | 100.0 | 0.0768 |
| all | 1232 | 0.0629 | 0.1157 | 0.2611 | 100.0 | 0.0769 |

Góc thật = ±(a + u), u ~ U[0; 0,5)°. Đơn vị sai số: độ. Hình: `error_by_angle.png`.


File: `result/e_a2_deskew/`

## E-A3 — Thời gian khối A (giây/ảnh, 50 ảnh mỗi mức)

| level | t_rectify_mean | t_deskew_mean | t_shadow_mean | t_denoise_mean | t_clahe_mean | t_total_mean |
| --- | --- | --- | --- | --- | --- | --- |
| L0 | 0.0289 | 0.0543 | 0.0336 | 0.0011 | 0.0039 | 0.1221 |
| L2 | 0.0272 | 0.0595 | 0.0319 | 0.0012 | 0.0038 | 0.1240 |
| L4 | 0.0080 | 0.0586 | 0.0310 | 0.0011 | 0.0038 | 0.1028 |

Phương pháp tìm góc giấy của A1:

| level | contour | full_image | hough | minrect |
| --- | --- | --- | --- | --- |
| L0 | 0 | 50 | 0 | 0 |
| L2 | 0 | 49 | 0 | 1 |
| L4 | 47 | 0 | 2 | 1 |


File: `result/e_a3_runtime/`

## Hình minh họa khối A

- `a_steps.png`: ảnh L4 qua 5 bước A1–A5
- `a_levels.png`: một ảnh ở L0 / L2 / L4
- `a_shadow.png`: ảnh có bóng, nền ước lượng bằng phép đóng, kết quả chia nền


File: `result/fig_enhance/`

## E-B — Phát hiện vùng chữ (khối B)

**E-B1 — không xét lớp, IoU ≥ 0,5**

| method | P | R | F1 | TP | FP | FN | s_per_img |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cổ điển (Sauvola + giãn) | 0.8639 | 0.7759 | 0.8175 | 4722 | 744 | 1364 | 0.0608 |
| YOLOv8s | 0.9915 | 0.9951 | 0.9933 | 6056 | 52 | 30 | 0.0306 |

**E-B2 — YOLOv8s theo lớp** (dòng `all` = cộng dồn mọi vùng)

| class | P | R | F1 | n_gt | mAP50 |
| --- | --- | --- | --- | --- | --- |
| drugname | 0.9977 | 1.0 | 0.9989 | 443 | 0.9950 |
| quantity | 1.0 | 1.0 | 1.0 | 443 | 0.9950 |
| usage | 1.0 | 0.9983 | 0.9992 | 602 | 0.9950 |
| diagnose | 0.9960 | 0.9921 | 0.9941 | 253 | 0.9934 |
| date | 1.0 | 1.0 | 1.0 | 176 | 0.9950 |
| other | 0.9874 | 0.9928 | 0.9901 | 4169 | 0.9947 |
| all | 0.9910 | 0.9946 | 0.9928 | 6086 | 0.9947 |

mAP50 = 0.9947, mAP50-95 = 0.9271. Ngưỡng YOLO 0.443; cổ điển chọn trên valid: {'a': 2.5, 'v': 0.2, 'lines': False}. Ma trận nhầm lẫn: `b2_confusion.png`.


File: `result/e_b_detection/`

## E-C1 — Nhận dạng chữ (khối C)

| engine | CER | WER | exact_match | CER_drugname | CER_quantity | CER_usage | CER_diagnose | CER_date | ms_per_crop | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_tesseract | 0.0689 | 0.1961 | 0.6359 | 0.0363 | 0.0016 | 0.1039 | 0.0811 | 0.1065 | 35.6782 | 6086 |
| C1_tesseract_sauvola | 0.0684 | 0.2002 | 0.6195 | 0.0340 | 0.0168 | 0.1029 | 0.0700 | 0.1034 | 37.8487 | 6086 |
| C2_pretrained | 0.0864 | 0.1097 | 0.8156 | 0.0275 | 0.0014 | 0.0123 | 0.2325 | 0.0 | 17.3476 | 6086 |
| C3_finetune | 0.0329 | 0.0437 | 0.9229 | 0.0003 | 0.0004 | 0.0038 | 0.0625 | 0.0 | 18.8942 | 6086 |

Vùng cắt theo khung chuẩn. `exact_match` = tỉ lệ vùng đọc đúng nguyên chuỗi. Tesseract 8 luồng CPU; VietOCR GPU RTX 3050, batch 64. `c3_errors.png`: 3 vùng C3 đọc khác nhãn (2 ca do nhãn sai).


File: `result/e_c1_ocr/`

## E-D1 — Luật ghép khoản trên khung + chữ chuẩn (khối D)

| mode | tp | fp | fn | f1 | n_bad_images |
| --- | --- | --- | --- | --- | --- |
| labels | 443 | 0 | 0 | 1.0 | 0 |
| text_only | 443 | 0 | 0 | 1.0 | 0 |


File: `result/e_d1_rules/`

## E-D2 — Bảng chính đầu-cuối

_chưa có — chưa chạy thí nghiệm_


## E-X1 — Ablation: lỗi đến từ khối nào

_chưa có — chưa chạy thí nghiệm_


## E-A1 — Độ bền khi ảnh suy giảm

_chưa có — chưa chạy thí nghiệm_


## Thời gian khối B + C theo pipeline

| preset | n | B_s_per_img | C_s_per_img |
| --- | --- | --- | --- |
| dl_c3 | 176 | 0.0494 | 3.3196 |

Khối B + C trên 176 ảnh test (GPU RTX 3050 cho YOLO / VietOCR, Tesseract CPU). Khối A xem E-A3; khối D ≈ 0,01–0,03 s/ảnh.


File: `result/runtime/`
