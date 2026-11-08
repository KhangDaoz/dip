"""Khử nhiễu (Chương 3 — bộ lọc không gian; bài giảng slide 56, 73–78; bài thực hành Chương 3 mục 5). Thí nghiệm D11.

Ảnh bộ 2, 3 (Roboflow) có nhiễu muối tiêu do tăng cường dữ liệu; nhiễu này làm hỏng các bước sau:
điểm trắng bị tính là lóa, điểm đen bị tính là mực khi chỉnh nghiêng / nhị phân hóa.

Phương án (đều trả ảnh cùng shape, dtype uint8, BGR hoặc xám):
    median3, median5        lọc trung vị toàn ảnh (slide 56, 73) — đơn giản nhưng mài mòn nét chữ mảnh
    switching               trung vị chọn lọc: chỉ thay điểm bị phát hiện là nhiễu xung, giữ nguyên phần còn lại
    adaptive                trung vị thích nghi (Gonzalez & Woods): cửa sổ nở 3 → max_size đến khi trung vị
                            không phải điểm cực trị; chỉ áp cho điểm nhiễu
    gaussian, bilateral     đối chứng (dành cho nhiễu Gauss, không hợp nhiễu xung)
"""
import cv2
import numpy as np

from common.io import to_gray

SALT_MARGIN = 40          # muối: sáng hơn MỌI láng giềng 8-liên thông > 40 mức xám
PEPPER_MARGIN = 80        # tiêu: tối hơn MỌI láng giềng > 80 (lớn hơn muối: đầu / góc nét chữ khử răng cưa
                          # tối hơn mọi láng giềng ~50 mức nhưng láng giềng vẫn là nét, không phải nền)
CLUSTER_SALT_MARGIN = 40     # cụm nhỏ (≤ 3×3): cực trị 3×3 và sáng / tối hơn mọi điểm vòng 5×5 một biên độ
CLUSTER_PEPPER_MARGIN = 100  # — bắt các cụm 2–3 điểm dính nhau khi nhiễu dày (mật độ ≥ 5%)
SWITCHING_PASSES = 3          # D11: nhiễu 10% F-measure nét chữ 0,80 (1 lượt) → 0,88 (3 lượt); ảnh sạch giữ 0,996
NOISE_RATIO_MIN = 0.0005     # > 0,05% điểm ảnh là nhiễu xung → ảnh có nhiễu, cần khử (ảnh sạch bộ 1, 6: ≤ 0,02%)


RING3 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
RING5 = np.ones((5, 5), np.uint8)
RING5[1:4, 1:4] = 0


def impulse_mask(img: np.ndarray, salt_margin: int = SALT_MARGIN, pepper_margin: int = PEPPER_MARGIN,
                 cluster_salt_margin: int = CLUSTER_SALT_MARGIN,
                 cluster_pepper_margin: int = CLUSTER_PEPPER_MARGIN) -> np.ndarray:
    """Mask điểm nhiễu xung (255).

    1. Điểm đơn: cực trị cục bộ chặt — sáng (tối) hơn MỌI láng giềng 8-liên thông một biên độ. Khác tiêu chí
       "lệch trung vị 3×3", cách này không bắt nhầm nét chữ rộng 1 px (điểm trên nét có láng giềng cùng màu).
    2. Cụm nhỏ: điểm là cực đại (cực tiểu) của cửa sổ 3×3 và sáng (tối) hơn mọi điểm của vòng 5×5 bao quanh —
       nhiễu dày tạo cụm 2–3 điểm dính nhau mà tiêu chí 1 bỏ sót.
    Hiệu chỉnh 02/11 (D11): ảnh sạch bộ 6 bắt nhầm ≤ 0,02%; nhiễu nhân tạo + JPEG: độ chính xác ≈ 1,0
    (bỏ sót chủ yếu điểm muối trên nền trắng / tiêu trên nét đen — vốn vô hại).
    """
    g = to_gray(img)
    gi = g.astype(np.int16)
    m = (gi - cv2.dilate(g, RING3).astype(np.int16) > salt_margin) | \
        (cv2.erode(g, RING3).astype(np.int16) - gi > pepper_margin)
    box = np.ones((3, 3), np.uint8)
    m |= (gi - cv2.dilate(g, RING5).astype(np.int16) > cluster_salt_margin) & (g == cv2.dilate(g, box))
    m |= (cv2.erode(g, RING5).astype(np.int16) - gi > cluster_pepper_margin) & (g == cv2.erode(g, box))
    return np.where(m, 255, 0).astype(np.uint8)


def noise_ratio(img: np.ndarray) -> float:
    """Tỉ lệ điểm nhiễu xung trên ảnh."""
    return float((impulse_mask(img) > 0).mean())


def _per_channel(img: np.ndarray, fn) -> np.ndarray:
    if img.ndim == 2:
        return fn(img)
    return cv2.merge([fn(c) for c in cv2.split(img)])


def median(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(img, ksize)


def switching_median(img: np.ndarray, ksize: int = 3, mask: np.ndarray = None, passes: int = SWITCHING_PASSES) -> np.ndarray:
    """Chỉ điểm trong mask nhiễu (mặc định impulse_mask) được thay bằng trung vị ksize×ksize.

    Lặp tối đa `passes` lượt (như lọc trung vị nhiều lượt, slide 76), mỗi lượt phát hiện lại trên ảnh vừa
    khử: nhiễu dày tạo cụm mà vòng 5×5 quanh cụm còn chứa điểm nhiễu khác, gỡ các điểm đó xong lượt sau
    mới bắt được cụm. Dừng sớm khi không còn điểm nhiễu. mask truyền vào chỉ dùng cho lượt đầu.
    """
    out = img.copy()
    for k in range(passes):
        m = (impulse_mask(out) if (mask is None or k > 0) else mask) > 0
        if not m.any():
            break
        med = cv2.medianBlur(out, ksize)
        out[m] = med[m]
    return out


def adaptive_median(img: np.ndarray, max_size: int = 7, mask: np.ndarray = None) -> np.ndarray:
    """Trung vị thích nghi trên các điểm nhiễu: với mỗi điểm, dùng cửa sổ nhỏ nhất (3, 5, …, max_size)
    có trung vị nằm giữa min và max của cửa sổ (tức trung vị không phải là chính nhiễu)."""
    mask = impulse_mask(img) if mask is None else mask
    todo = mask > 0

    def one(ch: np.ndarray) -> np.ndarray:
        out, left = ch.copy(), todo.copy()
        for k in range(3, max_size + 1, 2):
            if not left.any():
                break
            se = np.ones((k, k), np.uint8)
            med = cv2.medianBlur(ch, k)
            lo, hi = cv2.erode(ch, se), cv2.dilate(ch, se)
            ok = left & (med > lo) & (med < hi)
            out[ok] = med[ok]
            left &= ~ok
        if left.any():                      # cửa sổ lớn nhất vẫn bão hòa → dùng trung vị lớn nhất
            out[left] = cv2.medianBlur(ch, max_size)[left]
        return out

    return _per_channel(img, one)


def gaussian(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def bilateral(img: np.ndarray, d: int = 5, sigma_color: float = 50, sigma_space: float = 5) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


METHODS = {
    "median3": lambda im: median(im, 3),
    "median5": lambda im: median(im, 5),
    "switching": switching_median,
    "adaptive": adaptive_median,
    "gaussian": gaussian,
    "bilateral": bilateral,
}
DEFAULT_METHOD = "switching"  # D11: F-measure nét chữ bằng trung vị thích nghi, nhanh hơn ~14 lần; trung vị toàn ảnh làm mòn chữ nhỏ


def denoise(img: np.ndarray, method: str = DEFAULT_METHOD, min_ratio: float = NOISE_RATIO_MIN):
    """Khử nhiễu nếu ảnh có nhiễu xung đáng kể. Trả về (ảnh, noise_ratio, đã_khử)."""
    r = noise_ratio(img)
    if r <= min_ratio:
        return img, r, False
    return METHODS[method](img), r, True
