"""R6 xóa dấu / chữ ký — CHỈ chạy cho đơn thuốc (KeHoach.md mục 6.1). Inpainting ⚠️ ngoài bài giảng."""
import cv2
import numpy as np


def remove_marks(img_bgr: np.ndarray, mask: np.ndarray, method: str = "telea", radius: int = 3) -> np.ndarray:
    """method ∈ {"telea", "ns", "fill"}; "fill" = tô màu nền (toán tử điểm, C3 · S10). Không đổi pixel ngoài mask."""
    m = mask > 0
    if not m.any():
        return img_bgr.copy()
    if method == "fill":
        out = img_bgr.copy()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        paper = img_bgr[(~m) & (gray >= np.percentile(gray[~m], 60))]   # pixel sáng ngoài mask ≈ màu giấy
        color = np.median(paper, axis=0) if len(paper) else np.array([255, 255, 255])
        out[m] = color.astype(np.uint8)
        return out
    flag = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}[method]
    out = cv2.inpaint(img_bgr, m.astype(np.uint8) * 255, radius, flag)
    out[~m] = img_bgr[~m]
    return out
