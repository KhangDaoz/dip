"""Sinh bộ 5 → data/5_synthetic/. Chạy: python -m synthetic.generate"""
from pathlib import Path


def generate(config_path: Path = Path(__file__).with_name("config.yaml"), out_dir: Path = None) -> None:
    """Đọc config, chia ảnh gốc 60 / 10 / 30 (biến thể cùng ảnh gốc chung một tập), ghi ảnh hỏng + gt.json + mask."""
    raise NotImplementedError("generate")


if __name__ == "__main__":
    generate()
