"""Khung chung cho script thí nghiệm: mỗi thí nghiệm ghi vào results/<mã thí nghiệm>/."""
import time
from contextlib import contextmanager
from pathlib import Path

from .io import ensure_dir, save_json
from .paths import RESULTS_DIR


def results_dir(exp_id: str) -> Path:
    """Ví dụ results_dir("d1") -> <repo>/results/d1/"""
    return ensure_dir(RESULTS_DIR / exp_id)


def save_table(df, exp_id: str, name: str = "table") -> Path:
    """Lưu bảng kết quả (pandas DataFrame) ra CSV UTF-8 có BOM để mở bằng Excel không lỗi dấu."""
    path = results_dir(exp_id) / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_figure(fig, exp_id: str, name: str = "figure", dpi: int = 200) -> Path:
    """Lưu hình matplotlib ra PNG (đưa thẳng vào báo cáo)."""
    path = results_dir(exp_id) / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_config(config: dict, exp_id: str) -> Path:
    """Lưu tham số đã dùng, để bảng "tham số & lý do chọn" truy ngược được."""
    return save_json(results_dir(exp_id) / "config.json", config)


@contextmanager
def timer(store: dict, key: str):
    """with timer(times, "deskew"): ...  -> times["deskew"] = số ms."""
    start = time.perf_counter()
    yield
    store[key] = (time.perf_counter() - start) * 1000
