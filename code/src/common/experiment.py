import time
from contextlib import contextmanager
from pathlib import Path

from .io import ensure_dir, save_json
from .paths import RESULTS_DIR


def results_dir(exp_id: str) -> Path:
    return ensure_dir(RESULTS_DIR / exp_id)


def save_table(df, exp_id: str, name: str = "table") -> Path:
    path = results_dir(exp_id) / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_figure(fig, exp_id: str, name: str = "figure", dpi: int = 200) -> Path:
    path = results_dir(exp_id) / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_config(config: dict, exp_id: str) -> Path:
    return save_json(results_dir(exp_id) / "config.json", config)


@contextmanager
def timer(store: dict, key: str):
    start = time.perf_counter()
    yield
    store[key] = (time.perf_counter() - start) * 1000
