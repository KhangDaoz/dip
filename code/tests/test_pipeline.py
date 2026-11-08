"""Mốc tích hợp 05/11: chạy đầu-cuối trên ảnh tổng hợp. Bước nào chưa làm thì báo, không FAIL."""
import pytest

from common.io import write_image
from run_pipeline import run_one


def test_end_to_end_smoke(photo, tmp_path):
    path = write_image(tmp_path / "rx_test.png", photo["img"])
    status = run_one(path, tmp_path / "out", doc_type="prescription")
    errors = {k: v for k, v in status.items() if str(v).startswith("LỖI")}
    assert not errors, errors
    pending = {k: v for k, v in status.items() if str(v).startswith("chưa làm")}
    if pending:
        pytest.skip(f"pipeline dừng ở: {pending}")
