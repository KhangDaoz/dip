"""Chuẩn bị review bộ 5 (ảnh tự chụp): trang ảnh thu nhỏ theo loại + nhóm ảnh gần trùng (cùng tài liệu chụp lại).

Chỉ đọc data/5_self_captured/, không sửa ảnh hay manifest.csv (theo README của bộ 5).
Chạy (từ code/):  python scripts/review_set5.py
Kết quả: results/set5_review/<loại>_XX.jpg, near_duplicates.csv, labels_doc_type_draft.csv (bản nháp để Hiếu sửa)
"""
import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from common.experiment import results_dir
from common.io import read_image, write_image
from common.paths import DATASETS

ROOT = DATASETS[5]
CELL, COLS, ROWS = 300, 5, 4
PHASH_MAX = 10      # khoảng cách Hamming pHash ≤ 10 → nghi cùng tài liệu


def cell(img, text):
    s = CELL / max(img.shape[:2])
    small = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    box = np.full((CELL + 26, CELL, 3), 255, np.uint8)
    y, x = 26 + (CELL - small.shape[0]) // 2, (CELL - small.shape[1]) // 2
    box[y:y + small.shape[0], x:x + small.shape[1]] = small
    box[:26] = (60, 60, 60)
    cv2.putText(box, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(box, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def main():
    out = results_dir("set5_review")
    m = pd.read_csv(ROOT / "manifest.csv")
    hashes = {}
    for cat, g in m.groupby("category"):
        cells = []
        for _, r in g.iterrows():
            img = read_image(ROOT / r.relative_path)
            hashes[r.image_id] = imagehash.phash(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            cells.append(cell(img, f"{r.image_id}  {img.shape[1]}x{img.shape[0]}"))
        per = COLS * ROWS
        blank = np.full_like(cells[0], 255)
        for k in range(0, len(cells), per):
            chunk = cells[k:k + per] + [blank] * (per - len(cells[k:k + per]))
            sheet = np.vstack([np.hstack(chunk[i * COLS:(i + 1) * COLS]) for i in range(ROWS)])
            write_image(out / f"{cat}_{k // per + 1:02d}.jpg", sheet, jpeg_quality=85)

    ids = list(hashes)
    pairs = [(a, b, hashes[a] - hashes[b]) for i, a in enumerate(ids) for b in ids[i + 1:] if hashes[a] - hashes[b] <= PHASH_MAX]
    pd.DataFrame(pairs, columns=["image_a", "image_b", "phash_distance"]).sort_values("phash_distance") \
        .to_csv(out / "near_duplicates.csv", index=False, encoding="utf-8-sig")
    # gom cặp thành nhóm (thành phần liên thông)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, _ in pairs:
        parent[find(a)] = find(b)
    draft = m[["image_id", "relative_path", "category"]].rename(columns={"category": "doc_type"})
    draft["other_kind"] = ""            # với doc_type = other: medical_like (giấy ra viện, siêu âm...) / non_medical
    draft["near_dup_group"] = draft.image_id.map(lambda i: find(i))
    draft["note"] = ""
    draft.to_csv(out / "labels_doc_type_draft.csv", index=False, encoding="utf-8-sig")
    print(m.category.value_counts().to_string())
    print(f"cặp gần trùng (pHash ≤ {PHASH_MAX}): {len(pairs)}; số nhóm có ≥ 2 ảnh: "
          f"{(draft.near_dup_group.value_counts() > 1).sum()}")
    print("Xong:", out)


if __name__ == "__main__":
    main()
