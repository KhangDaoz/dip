"""R3 tách lớp mực theo HSV (KeHoach.md mục 6.1; C4 · S37–41)."""
import cv2
import numpy as np


def ink_masks(img_bgr: np.ndarray, red_h=((0, 10), (170, 180)), blue_h=(100, 130), s_min: int = 60,
              v_min: int = 50, dark_v: int = 140) -> dict:
    """Trả về {"red", "blue", "black"}: mask uint8, mực = 255.

    H (độ màu, C4 · S38) chọn đỏ / xanh; S (độ bão hòa, C4 · S39) loại vùng gần xám; V (C4 · S40) loại pixel quá tối
    vì H không đáng tin ở đó. HSV tách màu khỏi độ sáng nên ít bị bóng ảnh hưởng (C4 · S41).
    OpenCV lưu H trong [0, 180) (= độ / 2).
    """
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    colored = (S >= s_min) & (V >= v_min)
    red = np.zeros(H.shape, bool)
    for lo, hi in red_h:
        red |= (H >= lo) & (H <= hi)
    red &= colored
    blue = (H >= blue_h[0]) & (H <= blue_h[1]) & colored
    black = (V < dark_v) & ~red & ~blue
    to_u8 = lambda m: m.astype(np.uint8) * 255
    return {"red": to_u8(red), "blue": to_u8(blue), "black": to_u8(black)}
