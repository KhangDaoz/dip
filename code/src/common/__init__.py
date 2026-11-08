"""Tiện ích dùng chung cho cả nhóm: đường dẫn, đọc/ghi ảnh, hợp đồng dữ liệu, seed, thí nghiệm."""
import sys

# Console Windows mặc định cp1252 / cp1258 → print tiếng Việt bị UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
