"""D4 — đối chứng VLM zero-shot đầu-cuối qua API OpenAI-compatible (FPT AI; fallback OpenRouter) — plan §5.4.

Biến môi trường:
    VLM_API_BASE   (mặc định https://api.fpt.ai/morse/v1)
    VLM_API_KEY    (bắt buộc; hoặc OPENROUTER_API_KEY để dùng OpenRouter)
    VLM_MODEL      (mặc định qwen2.5-vl-32b; OpenRouter: qwen/qwen2.5-vl-32b-instruct)

Ảnh VAIPE đã được nguồn che thông tin cá nhân nên gửi API được. KHÔNG gửi ảnh bộ 4 (ảnh thật có thông tin BN).
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common.contracts import Item, PrescriptionResult

from .text_utils import norm, parse_date, strip_drug_prefix

PROMPT = """Bạn là hệ thống trích xuất thông tin từ ảnh đơn thuốc tiếng Việt.
Đọc ảnh và trả về DUY NHẤT một đối tượng JSON (không giải thích, không markdown) theo đúng schema:
{"items": [{"name": "<tên thuốc kèm hàm lượng, đúng như in trên đơn, bỏ số thứ tự đầu dòng>",
            "quantity": <số nguyên tổng số lượng, null nếu không có>,
            "unit": "<đơn vị, ví dụ Viên>",
            "usage": ["<từng dòng cách dùng, đúng như in>"]}],
 "diagnose": "<nội dung chẩn đoán, bỏ chữ 'Chẩn đoán:'>",
 "date": "<ngày kê đơn dạng dd/mm/yyyy>"}
Giữ nguyên chính tả, dấu tiếng Việt và chữ hoa/thường như trên ảnh. Không bịa thuốc không có trên ảnh."""


class VLMUnavailable(RuntimeError):
    pass


def config() -> dict:
    key = os.environ.get("VLM_API_KEY")
    if key:
        return {"base_url": os.environ.get("VLM_API_BASE", "https://api.fpt.ai/morse/v1"), "api_key": key,
                "model": os.environ.get("VLM_MODEL", "qwen2.5-vl-32b"), "provider": "fpt"}
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return {"base_url": "https://openrouter.ai/api/v1", "api_key": key,
                "model": os.environ.get("VLM_MODEL", "qwen/qwen2.5-vl-32b-instruct"), "provider": "openrouter"}
    raise VLMUnavailable("chưa đặt VLM_API_KEY (hoặc OPENROUTER_API_KEY)")


def _client(cfg):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise VLMUnavailable("chưa cài openai (pip install -r requirements-dl.txt)") from e
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=120)


def encode_image(img: np.ndarray, long_side: int = 1600, quality: int = 90) -> str:
    """BGR → data URL JPEG base64, thu nhỏ cạnh dài ≤ long_side để giảm token."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("không mã hóa được ảnh")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def parse_json(text: str):
    """Lấy đối tượng JSON đầu tiên trong câu trả lời (chịu được ```json ... ```). Hỏng → None."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _to_int(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


def to_result(image_id: str, d: dict | None, lexicon=None) -> PrescriptionResult:
    """JSON của VLM → PrescriptionResult (JSON hỏng → đơn rỗng)."""
    d = d or {}
    items = []
    for it in d.get("items") or []:
        if not isinstance(it, dict):
            continue
        name_raw = norm(str(it.get("name") or ""))
        usage = it.get("usage") or []
        usage = [norm(str(u)) for u in (usage if isinstance(usage, list) else [usage])]
        item = Item(name=strip_drug_prefix(name_raw), name_raw=name_raw, quantity=_to_int(it.get("quantity")),
                    unit=(norm(str(it["unit"])) if it.get("unit") else None), usage=usage)
        if lexicon is not None:
            item.lexicon_match = lexicon.match(item.name)[0]
        items.append(item)
    date = d.get("date")
    return PrescriptionResult(image_id=image_id, items=items, diagnose=(norm(str(d["diagnose"])) if d.get("diagnose") else None),
                              date=(parse_date(f"ngay {date}") or norm(str(date))) if date else None)


def extract(img: np.ndarray, image_id: str = "", prompt: str = PROMPT, lexicon=None, retries: int = 2) -> tuple:
    """Gọi VLM một ảnh. Trả về (PrescriptionResult, raw_text). temperature=0; JSON hỏng thì gọi lại một lần."""
    cfg = config()
    client = _client(cfg)
    url = encode_image(img)
    t0 = time.perf_counter()
    raw, parsed = "", None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"], temperature=0, max_tokens=2048,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt},
                                                       {"type": "image_url", "image_url": {"url": url}}]}])
            raw = resp.choices[0].message.content or ""
            parsed = parse_json(raw)
            if parsed is not None:
                break
        except Exception as e:   # rate-limit / mạng: chờ rồi thử lại
            raw = f"ERROR: {e}"
            time.sleep(2 * (attempt + 1))
    res = to_result(image_id, parsed, lexicon)
    res.timing = {"vlm": time.perf_counter() - t0}
    return res, raw


def extract_many(paths: list, workers: int = 8, out_dir: Path | None = None, lexicon=None) -> list:
    """Chạy song song nhiều ảnh (đường dẫn). Lưu <out_dir>/<id>.json + <id>.raw.txt nếu có out_dir; bỏ qua ảnh đã có."""
    from common.io import read_image, save_json

    def one(p):
        p = Path(p)
        if out_dir is not None and (Path(out_dir) / f"{p.stem}.json").exists():
            from common.io import load_json
            return PrescriptionResult.from_dict(load_json(Path(out_dir) / f"{p.stem}.json"))
        res, raw = extract(read_image(p), p.stem, lexicon=lexicon)
        if out_dir is not None:
            save_json(Path(out_dir) / f"{p.stem}.json", res.to_dict())
            (Path(out_dir) / f"{p.stem}.raw.txt").write_text(raw, encoding="utf-8")
        return res

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, paths))
