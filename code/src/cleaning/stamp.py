"""R4 phân biệt dấu / chữ đỏ, R5 mask dấu đặc (KeHoach.md mục 6.1)."""
import cv2
import numpy as np

CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)   # SE chữ thập (C7 · S5, bài tập 5 · S49)


def circularity(area: float, perimeter: float) -> float:
    """C = 4πA / P² (C8 · S16): tròn 1, vuông 0,785."""
    return float(4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0


def region_fill(A: np.ndarray, max_iter: int = 2000) -> np.ndarray:
    """Làm đầy vùng (C7 · S43–45): X_k = (X_{k−1} ⊕ B) ∩ Aᶜ, lặp tới X_k = X_{k−1}; kết quả = lỗ ∪ A.

    X₀ là các điểm biên khung ngoài A (chắc chắn nằm ngoài đối tượng) nên X_∞ = phần NỀN ngoài nối với biên;
    lỗ bên trong = Aᶜ trừ X_∞. Cách lấy hạt giống này tương đương đặt X₀ trong từng lỗ nhưng không cần biết trước lỗ.
    """
    a = (A > 0).astype(np.uint8)
    a = np.pad(a, 1)                       # thêm viền để biên khung chắc chắn là nền
    comp = 1 - a
    X = np.zeros_like(a)
    X[0, :] = X[-1, :] = X[:, 0] = X[:, -1] = 1
    X &= comp
    for _ in range(max_iter):
        Xn = cv2.dilate(X, CROSS) & comp
        if np.array_equal(Xn, X):
            break
        X = Xn
    filled = (a | (comp & (1 - X)))[1:-1, 1:-1]
    return filled.astype(np.uint8) * 255


def boundary(A: np.ndarray) -> np.ndarray:
    """Tách biên β(A) = A − (A ⊖ B) (C7 · S40)."""
    a = (A > 0).astype(np.uint8)
    return (a - cv2.erode(a, np.ones((3, 3), np.uint8))).astype(np.uint8) * 255


def fill_stamp(ring_mask: np.ndarray) -> np.ndarray:
    """R5: tách vành β(A) = A − (A ⊖ B) (C7 · S40), làm đầy vùng X_k = (X_{k−1} ⊕ B) ∩ Aᶜ (C7 · S43–45).

    Vành dấu bị chữ / nhiễu làm đứt → Closing nhỏ nối vành trước (C7 · S38).
    """
    ring = cv2.morphologyEx((ring_mask > 0).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return region_fill(ring)


def split_stamp_vs_text(red_mask: np.ndarray, min_area_ratio: float = 0.002, min_circularity: float = 0.6,
                        min_size_px: int = 40) -> tuple:
    """R4: theo từng thành phần liên thông của mask đỏ (C5 · S84): dấu = to, độ tròn C cao (sau khi làm đầy), có vành kín.
    Trả về (stamp_mask, red_text_mask)."""
    h, w = red_mask.shape
    joined = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))   # nối nét dấu đứt
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    stamp = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if min(bw, bh) < min_size_px or bw * bh < min_area_ratio * h * w:
            continue
        crop = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8) * 255
        filled = fill_stamp(crop)
        cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        circ = circularity(cv2.contourArea(c), cv2.arcLength(c, True))
        aspect = min(bw, bh) / max(bw, bh)
        has_hole = (filled > 0).sum() > 1.5 * (crop > 0).sum()   # làm đầy thêm nhiều → có vành kín
        if circ >= min_circularity and aspect > 0.5 and has_hole:
            stamp[y:y + bh, x:x + bw] = np.maximum(stamp[y:y + bh, x:x + bw], filled)
    red_text = np.where(stamp > 0, 0, red_mask).astype(np.uint8)
    return stamp, red_text
