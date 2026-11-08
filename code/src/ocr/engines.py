"""Bọc các engine OCR về cùng một giao diện.

Phần phụ trách: Bộ OCR dùng chung (Khang) — dùng cho cả nhóm đo CER/WER và nhận diện ký tự.
Cài: pip install -r requirements-ocr.txt (+ Tesseract 5 có gói vie). Import engine bên trong hàm
để ai không cài OCR vẫn chạy được phần còn lại.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

ENGINES = ("tesseract", "paddle", "vietocr")

logger = logging.getLogger(__name__)

# Cache các engine / predictor để tránh tải lại mô hình nặng mỗi lần gọi hàm
_PADDLE_INSTANCES: Dict[str, Any] = {}
_VIETOCR_INSTANCES: Dict[str, Any] = {}


def _to_pil_image(img: Any):
    """Chuyển numpy array (BGR hoặc Grayscale) hoặc đường dẫn sang PIL Image RGB."""
    from PIL import Image

    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if np is not None and isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img).convert("RGB")
        elif img.ndim == 3:
            if img.shape[2] == 3:
                # OpenCV BGR -> RGB
                return Image.fromarray(img[:, :, ::-1])
            elif img.shape[2] == 4:
                return Image.fromarray(img[:, :, [2, 1, 0, 3]]).convert("RGB")
        return Image.fromarray(img)

    if isinstance(img, (str, bytes)):
        return Image.open(img).convert("RGB")

    raise TypeError(f"Không hỗ trợ chuyển đổi kiểu dữ liệu {type(img)} sang PIL Image")


def _get_paddle_ocr(lang: str = "en"):
    """Lazy load và cache instance PaddleOCR."""
    normalized_lang = "ch" if lang in ("chinese", "ch") else ("en" if lang in ("eng", "en") else lang)
    if normalized_lang not in _PADDLE_INSTANCES:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR chưa được cài đặt. Vui lòng cài: pip install paddlepaddle paddleocr"
            ) from exc
        _PADDLE_INSTANCES[normalized_lang] = PaddleOCR(use_angle_cls=False, lang=normalized_lang, show_log=False)
    return _PADDLE_INSTANCES[normalized_lang]


def _get_vietocr_predictor(config_name: str = "vgg_transformer"):
    """Lazy load và cache instance VietOCR Predictor."""
    if config_name not in _VIETOCR_INSTANCES:
        try:
            from vietocr.tool.config import CConfig
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise ImportError(
                "VietOCR chưa được cài đặt. Vui lòng cài: pip install vietocr"
            ) from exc
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        cfg = CConfig.load_config_default(config_name)
        cfg["device"] = device
        cfg["predictor"]["beamsearch"] = False
        _VIETOCR_INSTANCES[config_name] = Predictor(cfg)
    return _VIETOCR_INSTANCES[config_name]


def ocr_word(img: Any, engine: str = "tesseract", lang: str = "eng") -> Tuple[str, float]:
    """OCR một ảnh chứa một từ (hoặc cụm từ). Trả về (text, confidence 0..1)."""
    if engine not in ENGINES:
        raise ValueError(f"Engine '{engine}' không được hỗ trợ. Chọn từ: {ENGINES}")

    if img is None:
        return "", 0.0

    if engine == "tesseract":
        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError("pytesseract chưa được cài đặt. Vui lòng cài: pip install pytesseract") from exc

        pil_img = _to_pil_image(img)
        # PSM 8: treat the image as a single word
        data = pytesseract.image_to_data(
            pil_img, lang=lang, config="--psm 8", output_type=pytesseract.Output.DICT
        )
        texts: List[str] = []
        confs: List[float] = []
        for t, c in zip(data.get("text", []), data.get("conf", [])):
            clean_t = str(t).strip()
            if clean_t:
                texts.append(clean_t)
                try:
                    score = float(c)
                    if score >= 0:
                        confs.append(score / 100.0)
                except (ValueError, TypeError):
                    pass

        # Nếu PSM 8 không bắt được (ví dụ cụm từ ngắn gồm nhiều token), thử PSM 7 (single line)
        if not texts:
            data = pytesseract.image_to_data(
                pil_img, lang=lang, config="--psm 7", output_type=pytesseract.Output.DICT
            )
            for t, c in zip(data.get("text", []), data.get("conf", [])):
                clean_t = str(t).strip()
                if clean_t:
                    texts.append(clean_t)
                    try:
                        score = float(c)
                        if score >= 0:
                            confs.append(score / 100.0)
                    except (ValueError, TypeError):
                        pass

        text = " ".join(texts)
        conf = float(sum(confs) / len(confs)) if confs else 0.0
        return text, min(1.0, max(0.0, conf))

    elif engine == "paddle":
        ocr = _get_paddle_ocr(lang)
        if np is not None and isinstance(img, np.ndarray):
            input_img = img
        else:
            pil_img = _to_pil_image(img)
            input_img = np.array(pil_img) if np is not None else pil_img

        result = ocr.ocr(input_img, cls=False)
        if not result or not result[0]:
            return "", 0.0

        texts = []
        scores = []
        for item in result[0]:
            if len(item) >= 2 and len(item[1]) >= 2:
                t, s = item[1][0], item[1][1]
                t_str = str(t).strip()
                if t_str:
                    texts.append(t_str)
                    scores.append(float(s))

        text = " ".join(texts)
        conf = float(sum(scores) / len(scores)) if scores else 0.0
        return text, min(1.0, max(0.0, conf))

    elif engine == "vietocr":
        predictor = _get_vietocr_predictor()
        pil_img = _to_pil_image(img)
        text, prob = predictor.predict(pil_img, return_prob=True)
        return str(text).strip(), min(1.0, max(0.0, float(prob)))

    return "", 0.0


def ocr_line(img: Any, engine: str = "tesseract", lang: str = "eng") -> Tuple[str, float]:
    """OCR một ảnh chứa một dòng. Tiếng Việt dùng engine="vietocr". Trả về (text, confidence 0..1)."""
    if engine not in ENGINES:
        raise ValueError(f"Engine '{engine}' không được hỗ trợ. Chọn từ: {ENGINES}")

    if img is None:
        return "", 0.0

    if engine == "vietocr":
        predictor = _get_vietocr_predictor()
        pil_img = _to_pil_image(img)
        text, prob = predictor.predict(pil_img, return_prob=True)
        return str(text).strip(), min(1.0, max(0.0, float(prob)))

    elif engine == "tesseract":
        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError("pytesseract chưa được cài đặt. Vui lòng cài: pip install pytesseract") from exc

        pil_img = _to_pil_image(img)
        # PSM 7: treat the image as a single text line
        data = pytesseract.image_to_data(
            pil_img, lang=lang, config="--psm 7", output_type=pytesseract.Output.DICT
        )
        texts: List[str] = []
        confs: List[float] = []
        for t, c in zip(data.get("text", []), data.get("conf", [])):
            clean_t = str(t).strip()
            if clean_t:
                texts.append(clean_t)
                try:
                    score = float(c)
                    if score >= 0:
                        confs.append(score / 100.0)
                except (ValueError, TypeError):
                    pass

        text = " ".join(texts)
        conf = float(sum(confs) / len(confs)) if confs else 0.0
        return text, min(1.0, max(0.0, conf))

    elif engine == "paddle":
        ocr = _get_paddle_ocr(lang)
        if np is not None and isinstance(img, np.ndarray):
            input_img = img
        else:
            pil_img = _to_pil_image(img)
            input_img = np.array(pil_img) if np is not None else pil_img

        result = ocr.ocr(input_img, cls=False)
        if not result or not result[0]:
            return "", 0.0

        texts = []
        scores = []
        for item in result[0]:
            if len(item) >= 2 and len(item[1]) >= 2:
                t, s = item[1][0], item[1][1]
                t_str = str(t).strip()
                if t_str:
                    texts.append(t_str)
                    scores.append(float(s))

        text = " ".join(texts)
        conf = float(sum(scores) / len(scores)) if scores else 0.0
        return text, min(1.0, max(0.0, conf))

    return "", 0.0


def ocr_page(img: Any, engine: str = "tesseract", lang: str = "eng") -> List[Dict[str, Any]]:
    """OCR cả trang. Trả về list dict {"bbox": [x, y, w, h], "text": str, "confidence": float} theo từ."""
    if engine not in ENGINES:
        raise ValueError(f"Engine '{engine}' không được hỗ trợ. Chọn từ: {ENGINES}")

    if img is None:
        return []

    if engine == "tesseract":
        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError("pytesseract chưa được cài đặt. Vui lòng cài: pip install pytesseract") from exc

        pil_img = _to_pil_image(img)
        # PSM 3: Fully automatic page segmentation, but no OSD
        data = pytesseract.image_to_data(
            pil_img, lang=lang, config="--psm 3", output_type=pytesseract.Output.DICT
        )
        results: List[Dict[str, Any]] = []
        n_boxes = len(data.get("text", []))
        for i in range(n_boxes):
            text = str(data["text"][i]).strip()
            if not text:
                continue
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            conf_val = data["conf"][i]
            try:
                c = float(conf_val)
                confidence = max(0.0, min(1.0, c / 100.0)) if c >= 0 else 0.0
            except (ValueError, TypeError):
                confidence = 0.0

            results.append({
                "bbox": [x, y, w, h],
                "text": text,
                "confidence": confidence,
            })
        return results

    elif engine == "paddle":
        ocr = _get_paddle_ocr(lang)
        if np is not None and isinstance(img, np.ndarray):
            input_img = img
        else:
            pil_img = _to_pil_image(img)
            input_img = np.array(pil_img) if np is not None else pil_img

        result = ocr.ocr(input_img, cls=False)
        results: List[Dict[str, Any]] = []
        if result and result[0]:
            for item in result[0]:
                pts = item[0]
                text = str(item[1][0]).strip()
                score = float(item[1][1])
                if not text:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x = int(min(xs))
                y = int(min(ys))
                w = int(max(xs) - x)
                h = int(max(ys) - y)
                results.append({
                    "bbox": [x, y, w, h],
                    "text": text,
                    "confidence": min(1.0, max(0.0, score)),
                })
        return results

    elif engine == "vietocr":
        raise NotImplementedError(
            "VietOCR là mô hình nhận diện dòng (text line recognition), không hỗ trợ phân đoạn cả trang độc lập. "
            "Vui lòng dùng engine='tesseract' hoặc 'paddle' cho ocr_page, hoặc dùng detector trích dòng "
            "rồi gọi ocr_line(..., engine='vietocr')."
        )

    return []
