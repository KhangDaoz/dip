"""Chạy nốt bảng chính E-D2 (kể cả VLM) rồi sinh lại report/latex/result/ — một lệnh.

    python tools/finalize_results.py              # từ thư mục code/
    python tools/finalize_results.py --no-vlm     # bỏ dòng VLM (chưa có API key)
    python tools/finalize_results.py --workers 2  # giảm luồng nếu API báo quá tải (429)

Cần: results/handoff/{classical,hybrid,dl_c2,dl_c3}.jsonl (python tools/make_handoff.py --preset dl_c3 dl_c2 hybrid classical),
data/1_vaipe/test/items_gt.json, và code/.env (VLM_API_BASE, VLM_API_KEY, VLM_MODEL) nếu chạy VLM.
VLM có cache theo ảnh (results/E-D2/vlm_raw/): chạy lại chỉ gọi API cho ảnh còn thiếu hoặc lỗi.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
HANDOFF = CODE.parent / "results" / "handoff"
PRESETS = ["classical", "hybrid", "dl_c2", "dl_c3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    missing = [p for p in PRESETS if not (HANDOFF / f"{p}.jsonl").exists()]
    if missing:
        sys.exit(f"Thiếu file bàn giao {missing} — chạy: python tools/make_handoff.py --preset {' '.join(missing)}")

    cmd = [sys.executable, "experiments/e_d2.py"]
    for p in PRESETS:
        cmd += ["--variant", f"{p}={HANDOFF / f'{p}.jsonl'}"]
    cmd += ["--check", "dl_c3"]
    if not args.no_vlm:
        cmd += ["--vlm", "--workers", str(args.workers)]
    print("[1/2] E-D2:", " ".join(cmd[1:3]), "...", flush=True)
    if subprocess.run(cmd, cwd=CODE).returncode != 0:
        sys.exit("E-D2 lỗi — xem thông báo phía trên")

    print("\n[2/2] Sinh lại report/latex/result/", flush=True)
    subprocess.run([sys.executable, "tools/export_report_results.py"], cwd=CODE, check=True)
    print("\nXong. Nếu E-D2 báo 'VLM lỗi API ở N ảnh', chạy lại lệnh này để gọi tiếp các ảnh đó.")


if __name__ == "__main__":
    main()
