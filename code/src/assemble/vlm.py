"""Khối D5 — đối chứng VLM zero-shot qua API OpenAI-compatible (FPT AI / OpenRouter) (plan_Thang.md mục 4.5).

Biến môi trường (hoặc file code/.env, không commit):
    VLM_API_BASE   ví dụ https://mkp-api.fptcloud.com/v1  hoặc  https://openrouter.ai/api/v1
    VLM_API_KEY
    VLM_MODEL      ví dụ Qwen2.5-VL-32B-Instruct  hoặc  qwen/qwen2.5-vl-32b-instruct

Câu trả lời thô được lưu vào cache_dir/<image_id>.json; đã có file thì không gọi lại API.
Đầu ra VLM đi qua CÙNG bước chuẩn hóa (D3) và từ điển (D4) như pipeline: xem vlm_to_items.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2

from common.contracts import Item
from common.io import read_image
from common.paths import CODE_DIR

from .rules import clean_diagnose, clean_name, clean_usage, normalize_unit, parse_date, parse_quantity

PROMPT = """Bạn là hệ thống trích xuất thông tin từ ảnh đơn thuốc Việt Nam.
Trả về DUY NHẤT một đối tượng JSON, không giải thích, theo đúng schema:
{"items":[{"name":str,"quantity":int|null,"unit":str|null,"usage":[str]}],"diagnose":str|null,"date":"dd/mm/yyyy"|null}
Quy tắc: "name" chép nguyên văn tên thuốc, bỏ số thứ tự đầu dòng như "1)". "quantity" là số sau "SL:".
"usage" là các dòng cách dùng của thuốc đó. "diagnose" chép nguyên văn sau "Chẩn đoán:".
Không bịa; trường nào không thấy thì để null."""

MAX_SIDE = 1600
JPEG_Q = 90


def load_env(path: Optional[Path] = None) -> dict:
    """Đọc VLM_* từ biến môi trường, bổ sung từ code/.env nếu có."""
    cfg = {k: os.environ.get(k) for k in ("VLM_API_BASE", "VLM_API_KEY", "VLM_MODEL")}
    env_file = path or (CODE_DIR / ".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                k = k.strip()
                if k in cfg and not cfg[k]:
                    cfg[k] = v.strip().strip('"').strip("'")
    return cfg


def encode_image(image_path) -> str:
    """Ảnh → JPEG base64, cạnh dài ≤ MAX_SIDE."""
    img = read_image(image_path)
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    if not ok:
        raise ValueError(f"Không mã hóa được ảnh {image_path}")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def parse_vlm_json(content: str) -> Optional[dict]:
    """Lấy object JSON từ câu trả lời. Thử nguyên văn, rồi cắt khối {...} ngoài cùng. Hỏng → None."""
    if not content:
        return None
    for cand in (content, content[content.find("{"): content.rfind("}") + 1] if "{" in content else ""):
        cand = re.sub(r"^```(json)?|```$", "", cand.strip()).strip()
        try:
            data = json.loads(cand)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def call_vlm(image_path, cfg: dict, retries: int = 3, timeout: float = 120) -> str:
    from openai import OpenAI   # import muộn: chỉ cần khi thật sự gọi API
    if not cfg.get("VLM_API_KEY") or not cfg.get("VLM_MODEL"):
        raise RuntimeError("Thiếu VLM_API_KEY / VLM_MODEL (đặt biến môi trường hoặc code/.env)")
    client = OpenAI(base_url=cfg.get("VLM_API_BASE") or None, api_key=cfg["VLM_API_KEY"], timeout=timeout)
    b64 = encode_image(image_path)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}]
    last = None
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(model=cfg["VLM_MODEL"], messages=messages,
                                               temperature=0, max_tokens=1500)
            return r.choices[0].message.content or ""
        except Exception as e:      # lỗi mạng, 429, 5xx
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Gọi VLM thất bại sau {retries} lần: {last}")


def vlm_extract(image_path, image_id: str, cache_dir, cfg: Optional[dict] = None) -> dict:
    """→ {"image_id", "raw", "parsed" (dict | None), "error"?}. Có cache."""
    cache = Path(cache_dir) / f"{image_id}.json"
    if cache.exists():
        rec = json.loads(cache.read_text(encoding="utf-8"))
    else:
        cfg = cfg or load_env()
        t0 = time.time()
        try:
            content, err = call_vlm(image_path, cfg), None
        except Exception as e:
            content, err = "", str(e)
        rec = {"image_id": image_id, "model": cfg.get("VLM_MODEL"), "content": content,
               "seconds": round(time.time() - t0, 2), "error": err}
        if err is None:              # lỗi mạng thì không cache, lần sau gọi lại
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    rec["parsed"] = parse_vlm_json(rec.get("content", ""))
    return rec


def run_many(jobs: list[tuple[str, str]], cache_dir, workers: int = 4, cfg: Optional[dict] = None) -> dict:
    """jobs = [(image_id, image_path)] → {image_id: rec}. Chạy song song nhiều luồng."""
    cfg = cfg or load_env()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(lambda j: vlm_extract(j[1], j[0], cache_dir, cfg), jobs))
    return {r["image_id"]: r for r in recs}


def _to_int(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


def vlm_to_items(parsed: Optional[dict], lexicon=None) -> tuple[list[Item], Optional[str], Optional[str]]:
    """Đưa đầu ra VLM qua cùng D3 + D4 như pipeline. parsed=None (JSON hỏng) → đơn rỗng."""
    if not parsed:
        return [], None, None
    items = []
    for it in parsed.get("items") or []:
        if not isinstance(it, dict):
            continue
        raw = str(it.get("name") or "")
        name, match = clean_name(raw), None
        if lexicon is not None and name:
            _, best, _, accepted = lexicon.match(name)
            match = best if accepted else None     # chỉ khi từ điển chấp nhận
        qty = _to_int(it.get("quantity"))
        unit = normalize_unit(it.get("unit"))
        if qty is None and isinstance(it.get("quantity"), str):
            qty, unit = parse_quantity("SL: " + it["quantity"])
        usage = [clean_usage(str(u)) for u in (it.get("usage") or []) if u]
        items.append(Item(name=name, name_raw=raw, quantity=qty, unit=unit, usage=usage, lexicon_match=match))
    diag = parsed.get("diagnose")
    date = parsed.get("date")
    return items, (clean_diagnose(str(diag)) or None) if diag else None, parse_date(str(date)) if date else None
