"""Tách ứng viên mực theo màu; chưa phân biệt dấu/chữ hoặc in/viết tay."""
import cv2
import numpy as np

from .binarize import sauvola


# Starting values from the team plan, not parameters selected on a benchmark.
_MIN_COLOR_VALUE = 40
# Leave a saturation gap before blue (>60) / red (>80): a pale colored stroke
# must not become black merely because it falls just below a color threshold.
_BLACK_MAX_SATURATION = 40
_BLACK_WINDOW = 31
_BLACK_K = 0.2
_WEAK_MIN_CONTRAST = 8.0
_BLACK_MAX_LAB_CHROMA = 12.0
_BLACK_MAX_TINT_SATURATION = 80
_BLACK_ABSOLUTE_CHROMA = 8.0
_BLACK_FRINGE_CHROMA = 20.0
_SUPPORTED_MIN_CONTRAST = 5.0
_RECOVERY_MIN_PAPER = 80


def _nearby(mask: np.ndarray) -> np.ndarray:
    """3x3 support lookup only; never use the enlarged support as output ink."""
    return cv2.boxFilter(mask.astype(np.uint8), -1, (3, 3), normalize=False,
                         borderType=cv2.BORDER_CONSTANT) > 0


def _black_strokes(img: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    """Dark/weak local strokes, with neutrality relative to local paper color.

    Closing estimates paper per channel; normalizing by it compensates a mild
    paper/lighting tint without modifying the BGR input or the red/blue masks.
    Original low-saturation pixels remain eligible. Absolute Lab neutrality
    handles saturation instability near black; supported, moderately tinted
    edges can be retained without allowing pale blue/red neighbors to spread.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_BLACK_WINDOW, _BLACK_WINDOW))
    paper = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE)
    ratios = img.astype(np.float32) / np.maximum(paper.astype(np.float32), 1.0)
    maximum = ratios.max(axis=2)
    minimum = ratios.min(axis=2)
    # HSV-like saturation of channel ratios, with defined neutrality at black.
    relative_saturation = np.divide(maximum - minimum, maximum,
                                    out=np.zeros_like(maximum), where=maximum > 0) * 255.0
    # HSV saturation divides by brightness: minor channel differences on a dark
    # stroke can have high S. A second, perceptual near-neutral check retains
    # such strokes without admitting the deliberately pale-blue test sample.
    lab = cv2.cvtColor(np.clip(ratios, 0, 1), cv2.COLOR_BGR2LAB)
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
    # Only compensate a moderate tint. A saturated purple graphic on purple
    # paper is neutral after normalization too, but is still not black ink.
    mild_tint = saturation <= _BLACK_MAX_TINT_SATURATION
    neutral = ((saturation <= _BLACK_MAX_SATURATION)
               | (mild_tint & ((relative_saturation <= _BLACK_MAX_SATURATION)
                               | (chroma <= _BLACK_MAX_LAB_CHROMA))))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    strong = sauvola(gray, window=_BLACK_WINDOW, k=_BLACK_K) > 0
    values = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(values, (_BLACK_WINDOW, _BLACK_WINDOW), 0,
                              borderType=cv2.BORDER_REFLECT_101)
    # Closing on gray separately avoids artificial contrast from mixing the
    # channel-wise maxima of two different-colored, equally bright columns.
    paper_gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel,
                                 borderType=cv2.BORDER_REPLICATE).astype(np.float32)
    weak = ((smooth - values >= _WEAK_MIN_CONTRAST)
            & (paper_gray - values >= _WEAK_MIN_CONTRAST))
    candidate = strong | weak
    raw_lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    raw_chroma = np.hypot(raw_lab[:, :, 1], raw_lab[:, :, 2])
    # HSV saturation is unstable near black: tiny absolute channel differences
    # can yield S=255. Perceptually neutral dark cores should not be discarded.
    hue, sat, value = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    color_hint = (((hue <= 10) | (hue >= 170) | ((hue >= 100) & (hue <= 130)))
                  & (sat > 40) & (value >= 40))
    recovery_paper = paper.min(axis=2) >= _RECOVERY_MIN_PAPER
    neutral = neutral | ((raw_chroma <= _BLACK_ABSOLUTE_CHROMA)
                         & ~color_hint & recovery_paper)
    colored = (((hue <= 10) | (hue >= 170)) & (sat > 80)
               | ((hue >= 100) & (hue <= 130) & (sat > 60))) & (value >= 40)
    core = candidate & neutral & ~colored
    # Accept weaker intensity evidence only beside an existing neutral stroke.
    # Unlike dilation, a selected pixel still has to be darker than BOTH local
    # references. One support pass avoids recursively growing into the shadow.
    supported = (_nearby(core) & (smooth - values >= _SUPPORTED_MIN_CONTRAST)
                 & (paper_gray - values >= _SUPPORTED_MIN_CONTRAST))
    candidate = candidate | supported
    # Restore moderately tinted edges next to a known neutral core, only when
    # every channel is darker than the local paper. No recursive propagation.
    fringe = (_nearby(core) & (maximum <= .8) & ~color_hint & recovery_paper
              & (raw_chroma <= _BLACK_FRINGE_CHROMA))
    neutral = neutral | fringe
    return candidate & neutral


def ink_masks(img: np.ndarray) -> dict:
    """Nhận BGR uint8 HxWx3, trả red/blue/black uint8 HxW, mực=255/nền=0.

    Không sửa đầu vào, resize hoặc thay hệ tọa độ. HSV OpenCV uint8:
    H=0..179, S/V=0..255. Đỏ: H<=10 hoặc H>=170, S>80; xanh:
    100<=H<=130, S>60. Cả hai yêu cầu V>=40 để bỏ màu quá tối.
    Nguồn thang HSV: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

    Đen gồm Sauvola (window=31, k=0.2, R=128) và nét tối hơn cả nền
    Closing xám lẫn Gaussian31 ít nhất 8 mức. Có thể nhận thêm nét yếu
    với chênh lệch >=5 ở cả hai nền, chỉ trong lân cận 3x3 của nét trung
    tính đã có; một lượt, không lan truyền đệ quy, không dilation đầu ra.
    Trung tính: S<=40, hoặc S<=80 và sau chia nền màu Closing có S<=40
    hay chroma Lab<=12. Bổ sung chroma Lab màu gốc<=8; hoặc chroma<=20,
    mọi kênh <=0.8 nền màu và sát lõi trung tính. Nền ước lượng mỗi kênh
    phải >=80 cho hai nhánh phục hồi màu, tránh mảng tối ngoài giấy. Chúng không
    nhận sắc đỏ/xanh có S>40,V>=40. Lab float, a/b quanh 0.
    Không phục hồi đồ họa bão hòa chỉ vì cùng màu nền; loại vùng đã nhận
    đỏ/xanh. Không lấy phần bù hai mask màu làm đen. Ngưỡng đang phát triển,
    chưa được chọn trên benchmark có GT màu độc lập.
    Ba mask không chồng nhau và không nhất thiết phủ hết các nét/màu.
    Không lọc thành phần nhỏ hay morphology để giữ dấu chấm/nét mảnh.

    Đây là ứng viên màu, không phải mask dấu/chữ ký đã xác nhận. Nền màu,
    logo hoặc quảng cáo đủ bão hòa có thể vào mask màu; màu nhạt/quá tối,
    nét đen dày hoặc tương phản quá thấp có thể bị bỏ sót; nhiễu tối,
    nền có vân hoặc màu gần nền có thể bị lẫn vào đen. Các màu không
    đạt điều kiện được để trống có chủ ý. Ảnh hằng trung tính
    không được suy thành mực đen chỉ vì tối. Chưa khử bóng hoặc đổi màu
    ảnh BGR trong hàm này; không tự động xóa bất kỳ vùng nào.

    Raises ValueError nếu đầu vào không phải ảnh BGR uint8 không rỗng.
    """
    if (not isinstance(img, np.ndarray) or img.dtype != np.uint8
            or img.ndim != 3 or img.shape[2] != 3 or img.size == 0):
        raise ValueError("Expected a nonempty HxWx3 uint8 BGR image")

    hue, saturation, value = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    visible = value >= _MIN_COLOR_VALUE
    red = ((hue <= 10) | (hue >= 170)) & (saturation > 80) & visible
    blue = (hue >= 100) & (hue <= 130) & (saturation > 60) & visible
    black = _black_strokes(img, saturation) & ~red & ~blue
    return {"red": red.astype(np.uint8) * 255,
            "blue": blue.astype(np.uint8) * 255,
            "black": black.astype(np.uint8) * 255}


def _is_closed_stamp_rim(component: np.ndarray, page_short_side: int) -> bool:
    """Recognize an isolated thin elliptical rim, not its interior contents.

    Provisional geometric gates, not thresholds selected on real stamp GT.
    Requiring the *whole* component to follow the rim rejects attached text
    instead of cutting that text at an ellipse boundary. Broken rims and
    ornate/connected seal interiors are deliberately left for later work.
    """
    contours, hierarchy = cv2.findContours(
        component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return False
    outer_ids = [i for i, item in enumerate(hierarchy[0]) if item[3] == -1]
    if len(outer_ids) != 1:
        return False
    index = outer_ids[0]
    outer = contours[index]
    area = cv2.contourArea(outer)
    if len(outer) < 5 or area <= 0:
        return False
    child = hierarchy[0, index, 2]
    holes = []
    while child != -1:
        holes.append(cv2.contourArea(contours[child]))
        child = hierarchy[0, child, 0]
    # A large central opening excludes solid blobs and dense graphics.
    if not holes or max(holes) / area < .65:
        return False
    (cx, cy), (diameter_a, diameter_b), angle = cv2.fitEllipse(outer)
    minor, major = sorted((diameter_a, diameter_b))
    if minor < max(16, .10 * page_short_side) or minor / major < .55:
        return False
    if not .90 <= area / (np.pi * diameter_a * diameter_b / 4) <= 1.05:
        return False

    theta = np.deg2rad(angle)

    def radii(xs, ys):
        dx, dy = xs - cx, ys - cy
        u = dx * np.cos(theta) + dy * np.sin(theta)
        v = -dx * np.sin(theta) + dy * np.cos(theta)
        return np.hypot(u / (diameter_a / 2), v / (diameter_b / 2))

    edge = outer[:, 0, :].astype(np.float64)
    if np.max(np.abs(radii(edge[:, 0], edge[:, 1]) - 1)) > .05:
        return False
    ys, xs = np.nonzero(component)
    radius = radii(xs, ys)
    # Reject the entire component when branches depart from the thin rim.
    # Do not swallow attached red text or emit a filled ellipse mask.
    return bool(np.all((radius >= .85) & (radius <= 1.05)))


def split_stamp_vs_text(red_mask: np.ndarray):
    """Partition red candidates into (stamp_mask, red_text_mask), conservatively.

    Contract: input is a nonempty HxW numpy uint8 array containing only 0/255;
    otherwise raise ValueError. Outputs are a tuple of two independent uint8
    arrays of the same shape, containing only 0/255 (foreground=255). Neither
    output aliases the input or the other output. Do not mutate the input;
    read-only and noncontiguous arrays are supported. No coordinate changes.

    Outputs are disjoint and their union equals the input exactly: candidate
    grouping may use morphology, but must never add pixels to the output.
    A zero input returns two zero masks. stamp_mask contains only confident
    geometric stamp candidates. red_text_mask is the remainder to KEEP,
    including uncertain graphics/noise, not a semantic assertion of text.
    Do not assign all pixels inside a candidate ellipse to the stamp merely
    because they lie inside it. Ambiguous/touching marks may remain unassigned
    to stamp; stamp recall is secondary to protecting independent red text.

    This mask-only API cannot prove whether a ring is a seal, logo or letter,
    nor recover strokes absent from red_mask. It does not authorize erasure,
    perform OCR/inpainting, or choose policy based on document type.
    Baseline recognizes only closed, isolated thin circular/elliptical rims:
    minor diameter >= max(16 px, 10% of page short side), axis ratio >= .55,
    largest hole/outer area >= .65, contour/ellipse area .90..1.05, maximum
    outer radial residual <= .05, all component radii .85..1.05. Thresholds
    are provisional. Components touching the image boundary are retained.
    Broken rims, attached text and internal seal lettering are not promoted
    to stamp. A large ring-shaped logo/letter remains geometrically ambiguous.
    """
    if (not isinstance(red_mask, np.ndarray) or red_mask.dtype != np.uint8
            or red_mask.ndim != 2 or red_mask.size == 0
            or np.any((red_mask != 0) & (red_mask != 255))):
        raise ValueError("Expected a nonempty HxW uint8 binary mask containing only 0/255")
    stamp = np.zeros_like(red_mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(red_mask), connectivity=8)
    height, width = red_mask.shape
    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        # Crop per component to avoid allocating a full page per candidate.
        if min(w, h) < max(16, .10 * min(height, width)):
            continue
        selected = labels[y:y+h, x:x+w] == label
        component = selected.astype(np.uint8) * 255
        if _is_closed_stamp_rim(component, min(height, width)):
            stamp[y:y+h, x:x+w][selected] = 255
    keep = red_mask.copy()
    keep[stamp > 0] = 0
    return stamp, keep
