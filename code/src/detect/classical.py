"""Đối chứng cổ điển của khối B: tìm vùng chữ bằng xử lý ảnh, KHÔNG phân lớp (plan.md mục 5.2).

1. Xám → Sauvola cửa sổ 31, k 0,2: T = m·[1 + k(s/R − 1)]                                   (ch. 5, ngưỡng cục bộ)
2. h = trung vị chiều cao thành phần liên thông                                             (ch. 3, đo đối tượng)
3. Xóa đường kẻ: phép mở với SE đường thẳng dài L·h (ngang và dọc), trừ khỏi ảnh nhị phân   (ch. 7)
4. Giãn SE chữ nhật (a·h cao × b·h rộng) nối ký tự thành cụm, không nối sang cột bên cạnh     (ch. 7)
5. Thành phần liên thông → box; lọc box quá thấp, quá lớn, quá dẹt                          (hậu xử lý)

Viết độc lập (không import cleaning/ hay table_extraction/) vì các module v1 đang chuyển sang legacy/.
Sauvola giống cleaning/binarize.py::sauvola; giãn + CC giống table_extraction/geometry.py::words_from_binary.
Tham số a, b, remove_lines chọn trên valid bằng experiments/e_b1.py --grid (results/e_b1/classical_grid.csv).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .textbox import TextBox


@dataclass
class ClassicalParams:
    window: int = 31          # cửa sổ Sauvola (plan.md)
    k: float = 0.2            # hệ số Sauvola (plan.md)
    r: float = 128.0          # dải động độ lệch chuẩn của Sauvola
    a: float = 0.5            # chiều cao SE giãn, × h
    b: float = 1.5            # chiều rộng SE giãn, × h
    remove_lines: bool = True
    line_len: float = 3.0     # độ dài SE lấy đường kẻ, × h (dài hơn mọi nét trong một ký tự)
    min_h: float = 0.5        # bỏ box thấp hơn min_h·h
    max_area: float = 0.5     # bỏ box lớn hơn max_area × diện tích trang

    def to_dict(self) -> dict:
        return asdict(self)


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """Ảnh nhị phân uint8, mực = 255. m, s tính bằng lọc trung bình (box filter) trên I và I²."""
    g = gray.astype(np.float64)
    m = cv2.boxFilter(g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(g * g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    s = np.sqrt(np.maximum(m2 - m * m, 0))
    return np.where(g < m * (1 + k * (s / r - 1)), 255, 0).astype(np.uint8)


def char_height(binary: np.ndarray, min_area: int = 10, default: float = 20.0) -> float:
    """Trung vị chiều cao các thành phần liên thông có diện tích > min_area (bỏ nhiễu), bỏ thành phần cao > 20 % trang."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT][(stats[1:, cv2.CC_STAT_AREA] > min_area)]
    hs = hs[hs <= 0.2 * binary.shape[0]]
    return float(np.median(hs)) if len(hs) else default


def remove_rules(binary: np.ndarray, length: int) -> np.ndarray:
    """Trừ đường kẻ ngang/dọc: phép mở với SE 1×L và L×1 giữ lại đúng các đoạn thẳng dài ≥ L."""
    length = max(3, int(length))
    hor = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    ver = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    return cv2.subtract(binary, cv2.bitwise_or(hor, ver))


class ClassicalDetector:
    def __init__(self, params: ClassicalParams | None = None, **overrides):
        p = params or ClassicalParams()
        self.p = ClassicalParams(**{**p.to_dict(), **overrides})

    def binarize(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Bước 1–3, tách riêng để quét lưới a, b không phải nhị phân hóa lại."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        p = self.p
        binary = sauvola(gray, p.window, p.k, p.r)
        h = char_height(binary)
        if p.remove_lines:
            binary = remove_rules(binary, round(p.line_len * h))
        return binary, h

    def boxes_from_binary(self, binary: np.ndarray, h: float, a: float | None = None,
                          b: float | None = None) -> list[list[int]]:
        """Bước 4–5: giãn + CC + lọc → list [x0, y0, x1, y1]."""
        p = self.p
        a = p.a if a is None else a
        b = p.b if b is None else b
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, round(b * h)), max(1, round(a * h))))
        smear = cv2.dilate(binary, se)
        _, _, stats, _ = cv2.connectedComponentsWithStats(smear, connectivity=8)
        page = binary.shape[0] * binary.shape[1]
        out = []
        for x, y, w, hh, _ in stats[1:]:
            if hh < p.min_h * h or w * hh > p.max_area * page:
                continue
            # SE giãn làm box phình thêm khoảng nửa SE mỗi phía: co lại về sát mực
            ys, xs = np.nonzero(binary[y:y + hh, x:x + w])
            if len(xs) == 0:
                continue
            out.append([int(x + xs.min()), int(y + ys.min()), int(x + xs.max() + 1), int(y + ys.max() + 1)])
        return sorted(out, key=lambda bb: (bb[1], bb[0]))

    def detect(self, img_bgr: np.ndarray) -> list[TextBox]:
        binary, h = self.binarize(img_bgr)
        return [TextBox(box=bb, label=None, score=1.0) for bb in self.boxes_from_binary(binary, h)]
