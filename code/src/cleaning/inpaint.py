"""Xóa dấu / nét bút bằng inpainting."""
from numbers import Integral

import cv2
import numpy as np


def remove_marks(img: np.ndarray, mask: np.ndarray, method: str = "telea", radius: int = 3) -> np.ndarray:
    """Process only masked pixels; return an independent image copy.

    Contract:
    * img: nonempty numpy uint8 HxW gray or HxWx3 BGR image.
    * mask: nonempty numpy uint8 HxW array matching img's spatial shape,
      containing only 0 and 255. 255 selects pixels to process; 0 protects them.
    * method: exactly 'telea', 'ns', or 'fill_white' (case-sensitive).
    * radius: positive integer (Python/numpy integer, never bool), in pixels,
      representable as a finite float for OpenCV. Validated for ALL methods.
      fill_white ignores the valid radius.
    * Invalid input/parameters raise ValueError, even for an all-zero mask.
    * Output preserves shape, uint8 dtype, gray/BGR channels and coordinates.
      Do not mutate or alias either input. Support read-only/noncontiguous
      inputs. Every pixel outside mask must remain EXACTLY unchanged.

    telea/ns use OpenCV INPAINT_TELEA/INPAINT_NS with the supplied radius.
    fill_white sets selected pixels to 255 in every channel, not local paper
    color. An all-zero mask returns a copy. An all-255 mask returns white for
    fill_white but raises ValueError for telea/ns (no reference pixels).

    No mask detection/dilation, resizing, color conversion, document routing,
    or automatic protection of red text happens here. The caller selects the
    mask; masked text may be destroyed. Inpainting does not promise to recover
    original hidden letters. This function does not select the best method or
    authorize use of a geometric stamp candidate as an erasure mask.
    """
    if (not isinstance(img, np.ndarray) or img.dtype != np.uint8
            or img.size == 0 or not (img.ndim == 2
                or (img.ndim == 3 and img.shape[2] == 3))):
        raise ValueError("img must be a nonempty uint8 HxW gray or HxWx3 BGR image")
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8
            or mask.ndim != 2 or mask.shape != img.shape[:2]
            or np.any((mask != 0) & (mask != 255))):
        raise ValueError("mask must be uint8 HxW matching img and contain only 0/255")
    if not isinstance(method, str) or method not in ('telea', 'ns', 'fill_white'):
        raise ValueError("method must be 'telea', 'ns', or 'fill_white'")
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, Integral) or radius <= 0:
        raise ValueError("radius must be a positive integer, not bool")
    try:
        backend_radius = float(radius)
    except OverflowError as exc:
        raise ValueError("radius must be representable as a finite float") from exc
    if not np.isfinite(backend_radius):
        raise ValueError("radius must be representable as a finite float")

    selected = mask == 255
    result = img.copy()
    if not selected.any():
        return result
    if method == 'fill_white':
        result[selected] = 255
        return result
    if selected.all():
        raise ValueError("telea/ns require at least one unmasked reference pixel")

    flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    restored = cv2.inpaint(np.ascontiguousarray(img), np.ascontiguousarray(mask),
                           backend_radius, flag)
    # Copy only selected pixels: preservation outside mask is an explicit API
    # guarantee, independent of backend behavior and image memory layout.
    result[selected] = restored[selected]
    return result
