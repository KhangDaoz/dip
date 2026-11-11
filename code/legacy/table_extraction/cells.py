"""K5 ghép ô (KeHoach.md mục 7.1; hợp vùng C5 · S87)."""


def column_of(word: list, columns: list) -> int:
    """Cột có đoạn chồng theo x lớn nhất với cụm từ; không chồng thì cột gần tâm nhất."""
    x, _, w, _ = word
    ov = [max(0, min(x + w, e) - max(x, s)) for s, e in columns]
    if max(ov, default=0) > 0:
        return int(max(range(len(columns)), key=lambda j: ov[j]))
    cx = x + w / 2
    return int(min(range(len(columns)), key=lambda j: abs(cx - (columns[j][0] + columns[j][1]) / 2)))


def assign_words_to_cells(words: list, rows: list, columns: list, min_iou: float = 0.5) -> list:
    """Giao dòng × cột; mỗi cụm từ vào ô có diện tích giao lớn nhất. Trả về list dict {"row", "col", "word_ids", "bbox"}.

    Các cụm từ cùng ô được hợp thành một vùng (hợp vùng, C5 · S87).
    """
    if not columns:
        return []
    cells = {}
    for r, row in enumerate(rows):
        for i in row:
            c = column_of(words[i], columns)
            cells.setdefault((r, c), []).append(i)
    out = []
    for (r, c), ids in sorted(cells.items()):
        xs = [words[i][0] for i in ids]
        ys = [words[i][1] for i in ids]
        xe = [words[i][0] + words[i][2] for i in ids]
        ye = [words[i][1] + words[i][3] for i in ids]
        out.append({"row": r, "col": c, "word_ids": sorted(ids, key=lambda i: words[i][0]),
                    "bbox": [min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)]})
    return out
