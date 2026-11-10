"""D6 — Tiền xử lý giúp OCR bao nhiêu? CLAHE hay cân bằng lược đồ toàn cục? Người làm: Hiếu. Kế hoạch: KeHoach.md mục 5.2 (D6), 5.3.

Dùng chung bộ chạy của T8 (experiments/t8_ablation.py): 450 mẫu test bộ 5, chữ chuẩn = các dòng trong vùng thuốc, đầu ra =
OCR từng khoản thuốc do ④ tìm tự động (Tesseract 5 vie). So sánh:
  full    pipeline như đang dùng (① nắn + chỉnh nghiêng + khử lóa, KHÔNG tăng tương phản trên đường OCR)
  clahe   + CLAHE (clipLimit 2, ô 8×8) trên kênh L (Lab) của ảnh đã nắn, trước ②
  he      + cân bằng lược đồ toàn cục s_k = (L−1) Σ n_j / n (C3 · S32–33) trên kênh L, trước ②
  no_geom − nắn phối cảnh + chỉnh nghiêng (đóng góp của ① trên đường OCR)
Độ đo: CER toàn tập, F1 trên từ. Kết quả: results/d6/{per_sample.csv, summary.csv, by_recipe.csv, config.json}

Chạy (từ code/): python experiments/d6_enhance.py
"""
import t8_ablation as T8

if __name__ == "__main__":
    T8.run("d6", ["full", "clahe", "he", "no_geom"])
