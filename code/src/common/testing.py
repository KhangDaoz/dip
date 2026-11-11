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
