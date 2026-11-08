"""Hỗ trợ test: hàm còn `raise NotImplementedError` thì test tự SKIP thay vì FAIL.

Khi bạn viết xong hàm, test sẽ tự chạy thật — không cần sửa file test.
"""
import functools

import pytest


def todo(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as e:
            pytest.skip(f"chưa làm: {e}")
    return wrapper
