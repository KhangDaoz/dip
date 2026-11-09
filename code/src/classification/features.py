"""H10 trích đặc trưng bố cục (KeHoach.md mục 5.1). Đầu vào: ảnh xám đã tiền xử lý (CLAHE)."""
import cv2
import numpy as np

WORK = 256   # mọi đặc trưng tính trên ảnh co về 256×256 để vector có độ dài cố định


def _resize(gray: np.ndarray, size: int = WORK) -> np.ndarray:
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def _binary(gray: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # Otsu, mực = 255 (C5 · S65)
    return b


def hog_features(gray: np.ndarray, size: int = 128) -> np.ndarray:
    """HOG: lược đồ hướng gradient theo ô (C8 · S12, S17). 9 hướng, ô 16×16, khối 2×2 → 1.764 chiều với size 128."""
    from skimage.feature import hog
    return hog(_resize(gray, size), orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def glcm_haralick(gray: np.ndarray, distances=(1,), angles=(0, 45, 90, 135), levels: int = 32) -> np.ndarray:
    """GLCM C(i,j) theo (d, θ), chuẩn hóa P = C / tổng; 4 đặc trưng Haralick lấy trung bình 4 hướng (C8 · S14–15):
    năng lượng ΣP², entropy −ΣP log₂P, tương phản Σ(i−j)²P, đồng nhất ΣP/(1+(i−j)²)."""
    from skimage.feature import graycomatrix
    q = (_resize(gray, 128).astype(np.int32) * levels // 256).astype(np.uint8)
    P = graycomatrix(q, distances=list(distances), angles=[np.radians(a) for a in angles],
                     levels=levels, symmetric=True, normed=True)
    i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    feats = []
    for d in range(P.shape[2]):
        for a in range(P.shape[3]):
            p = P[:, :, d, a]
            nz = p[p > 0]
            feats.append([(p ** 2).sum(), -(nz * np.log2(nz)).sum(),
                          ((i - j) ** 2 * p).sum(), (p / (1 + (i - j) ** 2)).sum()])
    return np.asarray(feats, dtype=np.float32).mean(axis=0)


def ink_density_grid(binary: np.ndarray, grids=(3, 6)) -> np.ndarray:
    """Mật độ mực trên lưới 3×3 và 6×6 (45 chiều): đơn thuốc và phiếu XN phân bố chữ khác nhau."""
    b = (_resize(binary) > 127).astype(np.float32)
    out = []
    for g in grids:
        for rows in np.array_split(b, g, axis=0):
            for cell in np.array_split(rows, g, axis=1):
                out.append(cell.mean())
    return np.asarray(out, dtype=np.float32)


def line_ratio(binary: np.ndarray, length: int = 40) -> np.ndarray:
    """Tỉ lệ đường kẻ ngang / dọc: Opening SE 1×length và length×1 chỉ giữ đoạn thẳng đủ dài "fit" SE (C7 · S6, S28)."""
    b = (_resize(binary) > 127).astype(np.uint8) * 255
    ink = max(int((b > 0).sum()), 1)
    horiz = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vert = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    return np.array([(horiz > 0).sum() / ink, (vert > 0).sum() / ink], dtype=np.float32)


def text_block_stats(binary: np.ndarray) -> np.ndarray:
    """Số dòng (đỉnh projection ngang, ➕ C3 · S23), độ rộng dòng trung bình, tỉ lệ vùng trắng."""
    b = (_resize(binary) > 127).astype(np.uint8)
    prof = b.sum(axis=1).astype(np.float64)
    on = prof > 0.05 * max(prof.max(), 1)
    n_lines = int(np.sum(on[1:] & ~on[:-1]) + on[0])
    widths = [np.ptp(np.nonzero(r)[0]) for r in b if r.any()]
    return np.array([n_lines / 50.0, (np.mean(widths) / WORK) if widths else 0.0, 1 - b.mean()], dtype=np.float32)


def color_histogram(img_bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    """Đối chứng: lược đồ màu chuẩn hóa 8×8×8 bin cho B, G, R, chia cho tổng số điểm ảnh (C8 · S13)."""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    h = cv2.calcHist([_resize(img_bgr)], [0, 1, 2], None, [bins] * 3, [0, 256] * 3).ravel()
    return (h / max(h.sum(), 1)).astype(np.float32)


def normalize_page(img_bgr: np.ndarray) -> np.ndarray:
    """Chuẩn hóa ảnh trước khi trích đặc trưng để giảm "học thuộc nguồn" (KeHoach.md mục 3.2):
    khử bóng (Closing ảnh xám, C7 · S35) + nhị phân Sauvola → ảnh mực đen trên nền trắng, mất màu giấy / độ sáng / nhiễu
    riêng của từng nguồn; cắt bỏ viền trắng quanh nội dung."""
    from cleaning.pipeline import PARAMS, binarize_clean, shadow_free
    b, _ = binarize_clean(shadow_free(img_bgr, PARAMS), PARAMS)
    ys, xs = np.nonzero(b)
    if len(ys):
        b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return 255 - b    # chữ đen, nền trắng


def extract_normalized(img_bgr: np.ndarray) -> np.ndarray:
    """Đặc trưng bố cục trên ảnh đã chuẩn hóa: HOG + mật độ mực + đường kẻ + thống kê dòng (GLCM bỏ vì ảnh nhị phân)."""
    return extract_all(normalize_page(img_bgr), groups=("hog", "density", "lines", "text"))


def extract_all(gray: np.ndarray, groups=("hog", "glcm", "density", "lines", "text")) -> np.ndarray:
    """Ghép các nhóm đặc trưng chính thành một vector (groups dùng cho ablation đặc trưng)."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    b = _binary(gray)
    parts = {"hog": lambda: hog_features(gray), "glcm": lambda: glcm_haralick(gray),
             "density": lambda: ink_density_grid(b), "lines": lambda: line_ratio(b),
             "text": lambda: text_block_stats(b)}
    return np.concatenate([parts[g]() for g in groups]).astype(np.float32)
