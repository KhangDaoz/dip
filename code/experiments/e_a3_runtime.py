import argparse
import platform
import time

import pandas as pd

from common.experiment import results_dir, save_config, save_table
from common.io import read_image
from common.paths import VAIPE_DEGRADED
from enhance.pipeline import STEPS, enhance

EXP_ID = "e_a3"


def cpu_name() -> str:
    try:
        import subprocess
        if platform.system() == "Windows":
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                           "(Get-CimInstance Win32_Processor).Name"], text=True)
            return out.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--levels", nargs="+", default=["L0", "L2", "L4"])
    args = ap.parse_args()
    results_dir(EXP_ID)
    save_config({**vars(args), "cpu": cpu_name(), "python": platform.python_version()}, EXP_ID)

    rows = []
    for lv in args.levels:
        paths = sorted((VAIPE_DEGRADED / lv).glob("*.png"))[:args.limit]
        if not paths:
            raise SystemExit(f"Thiếu {VAIPE_DEGRADED / lv} — chạy tools/make_degraded.py")
        enhance(read_image(paths[0]), "full")
        for p in paths:
            img = read_image(p)
            t = time.perf_counter()
            _, info = enhance(img, "full")
            rows.append({"level": lv, "image_id": p.stem, "h": img.shape[0], "w": img.shape[1],
                         "rectify_method": info.get("rectify_method"), "skew_deg": info.get("skew_deg"),
                         **{f"t_{s}": info["timing"][s] for s in STEPS}, "t_total": time.perf_counter() - t})
        print(f"  {lv}: {len(paths)} ảnh")
    df = pd.DataFrame(rows)
    save_table(df, EXP_ID, "raw")

    cols = [f"t_{s}" for s in STEPS] + ["t_total"]
    table = df.groupby("level")[cols].agg(["mean", "median"])
    table.columns = [f"{c}_{a}" for c, a in table.columns]
    table = table.reset_index()
    save_table(table, EXP_ID, "table")
    methods = df.groupby(["level", "rectify_method"]).size().unstack(fill_value=0).reset_index()
    save_table(methods, EXP_ID, "rectify_methods")
    pd.set_option("display.width", 200)
    print(df.groupby("level")[cols].mean().round(3).to_string())
    print(methods.to_string(index=False))


if __name__ == "__main__":
    main()
