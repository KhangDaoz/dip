"""H5 khử lóa nhẹ (KeHoach.md mục 5.1). Inpainting ⚠️ không có trong bài giảng — ghi rõ trong báo cáo."""
import cv2
import numpy as np


def remove_glare(img_bgr: np.ndarray, mask: np.ndarray, method: str = "telea", radius: int = 5) -> np.ndarray:
    """Inpainting vùng lóa; chỉ gọi khi tỉ lệ lóa 1–8 %. method ∈ {"telea", "ns"}."""
    flag = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}[method]
    return cv2.inpaint(img_bgr, (mask > 0).astype(np.uint8) * 255, radius, flag)
