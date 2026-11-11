from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from common.paths import VIETOCR_FINETUNED, YOLO_WEIGHTS

WEIGHTS = [
    {"path": YOLO_WEIGHTS,
     "drive_id": "1oejjzfMGchx6bek_U1iVAw9AbBbrBIho",
     "size": 22_567_786,
     "sha256": "89fd654a18aa803be014a99960b8890af790d4567c23da2303ad51b6f1ffaedc"},
    {"path": VIETOCR_FINETUNED,
     "drive_id": "1IUq2B0TNLtvLEFcALJ3LjCsjFm9lkuTN",
     "size": 151_811_471,
     "sha256": "702405e6fa5e618f704763b21805875b114b2b06434e99ca3d9c132186bb2da0"},
]
URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_ok(w: dict) -> bool:
    p = w["path"]
    return p.exists() and p.stat().st_size == w["size"] and sha256(p) == w["sha256"]


def download(w: dict):
    dst: Path = w["path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    url = URL.format(id=w["drive_id"])
    print(f"[tải] {dst.name} ({w['size'] / 1e6:.1f} MB) ← drive id {w['drive_id']}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:7.1f} / {w['size'] / 1e6:.1f} MB", end="", flush=True)
    print()
    if tmp.stat().st_size != w["size"] or sha256(tmp) != w["sha256"]:
        head = tmp.read_bytes()[:200]
        tmp.unlink()
        hint = " (Drive trả về trang HTML — link chưa chia sẻ công khai hoặc bị giới hạn lượt tải)" \
            if b"<html" in head.lower() else ""
        raise SystemExit(f"Sai kích thước/mã băm khi tải {dst.name}{hint}")
    shutil.move(tmp, dst)


def main():
    ap = argparse.ArgumentParser(description="Tải trọng số YOLO + VietOCR từ Drive nhóm")
    ap.add_argument("--force", action="store_true", help="tải lại kể cả khi file đã đúng")
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra mã băm")
    args = ap.parse_args()
    bad = []
    for w in WEIGHTS:
        ok = is_ok(w)
        if args.check:
            print(f"[{'OK ' if ok else 'SAI'}] {w['path']}")
            bad += [] if ok else [w]
            continue
        if ok and not args.force:
            print(f"[có sẵn] {w['path']}")
            continue
        download(w)
        print(f"[OK] {w['path']}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
