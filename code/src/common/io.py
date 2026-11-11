import json
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_image(path, gray: bool = False) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return img


def write_image(path, img: np.ndarray, jpeg_quality: int = 95) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if path.suffix.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(path.suffix, img, params)
    if not ok:
        raise ValueError(f"Không mã hóa được ảnh sang {path.suffix}")
    buf.tofile(str(path))
    return path


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return path
