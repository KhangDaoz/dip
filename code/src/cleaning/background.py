"""R1 ước lượng nền, R2 khử bóng (KeHoach.md mục 6.1)."""
import cv2
import numpy as np


def background_closing(gray: np.ndarray, kernel: int = 81, median_ksize: int = 0) -> np.ndarray:
    """Phương án chính: Closing ảnh xám = lọc Max rồi Min (C3 · S74; C7 · S35, S38), sau đó trung vị nếu median_ksize > 0.

    Nét chữ tối và hẹp hơn SE nên bị giá trị nền sáng xung quanh lấp; phần còn lại xấp xỉ ánh sáng chiếu lên giấy.
    """
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    if median_ksize and median_ksize > 1:
        bg = cv2.medianBlur(bg, median_ksize | 1)
    return bg


def background_median(gray: np.ndarray, kernel: int = 81) -> np.ndarray:
    """Đối chứng: trung vị cỡ lớn trực tiếp (C3 · S73). Làm trên ảnh co 1/4 cho nhanh rồi phóng về."""
    h, w = gray.shape
    small = cv2.resize(gray, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
    k = max(3, (kernel // 4) | 1)
    return cv2.resize(cv2.medianBlur(small, k), (w, h), interpolation=cv2.INTER_LINEAR)


def stretch_contrast(gray: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """R2b: kéo giãn độ tương phản tuyến tính từng đoạn (C3 · S21): mức xám ≤ r₁ → 0, ≥ r₂ → 255, giữa nội suy
    tuyến tính; r₁, r₂ = phân vị low_pct / high_pct. Ảnh chụp điện thoại có chữ nhạt (≈ 200–230 trên giấy 250)."""
    r1, r2 = np.percentile(gray, [low_pct, high_pct])
    if r2 - r1 < 1:
        return gray.copy()
    return np.clip((gray.astype(np.float32) - r1) * 255.0 / (r2 - r1), 0, 255).astype(np.uint8)


def remove_shadow(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """R2: toán tử điểm I_clean = min(255, I / B̂ · 255) (C3 · S10–11). Ảnh ≈ chiếu sáng × phản xạ (➕ C2 · S22)."""
    out = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)
