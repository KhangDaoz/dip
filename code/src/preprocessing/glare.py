"""Khử lóa bằng inpainting. Thí nghiệm D5."""
import cv2
import numpy as np

METHODS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}
MASK_DILATE = 5   # nới mask vài pixel: rìa vùng lóa còn sáng nhòe, không nới thì để lại quầng


def remove_glare(img: np.ndarray, mask: np.ndarray, method: str = "telea", radius: int = 5,
                 dilate: int = MASK_DILATE) -> np.ndarray:
    """Inpaint vùng mask (255) trên ảnh BGR. method: "telea" | "ns" (Navier–Stokes).

    mask phải cùng kích thước ảnh (ví dụ quality.glare_mask(img)). Mask rỗng → trả về bản sao.
    """
    if method not in METHODS:
        raise ValueError(f"method phải thuộc {tuple(METHODS)}")
    if mask.shape[:2] != img.shape[:2]:
        raise ValueError("mask phải cùng kích thước với ảnh")
    mask = (mask > 0).astype(np.uint8) * 255
    if not mask.any():
        return img.copy()
    if dilate > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    return cv2.inpaint(img, mask, radius, METHODS[method])
