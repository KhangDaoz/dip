"""Chạy pipeline đầu-cuối ① → ② → ③ / ④ trên một ảnh hoặc cả thư mục — dùng cho mốc tích hợp (KeHoach.md mục 12.4).

    python run_pipeline.py samples/                      # mọi ảnh trong samples/
    python run_pipeline.py anh.jpg --doc-type lab_report # ép loại tài liệu khi bộ phân loại chưa xong

Kết quả JSON của từng mắt xích lưu vào out/<image_id>_<bước>.json.
Bước nào còn NotImplementedError sẽ được báo "chưa làm" và dừng ở đó, không làm hỏng các ảnh khác.
"""
import argparse
from pathlib import Path

from common.contracts import DOC_TYPES
import cv2

from common.io import read_image, save_json, write_image
from common.paths import OUT_DIR, REPO_DIR
from cleaning.pipeline import clean
from prescription_segmentation.pipeline import extract_prescription
from preprocessing.pipeline import preprocess
from table_extraction.pipeline import extract_table

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PALETTE = [(0, 0, 220), (0, 150, 0), (200, 80, 0), (160, 0, 160), (0, 140, 200)]


def draw_boxes(img, boxes):
    """Vẽ khung từng dòng chỉ số / khoản thuốc (mỗi khung một màu, đánh số) — ảnh dùng cho demo."""
    vis = img.copy()
    t = max(2, img.shape[1] // 400)
    for i, (x, y, w, h) in enumerate(boxes):
        c = PALETTE[i % len(PALETTE)]
        cv2.rectangle(vis, (x, y), (x + w, y + h), c, t)
        cv2.putText(vis, str(i + 1), (x + 4, y + max(h // 2, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t / 2 + 0.3, c, t)
    return vis


def run_one(path: Path, out_dir: Path, doc_type: str = None) -> dict:
    image_id = path.stem
    status = {"image_id": image_id}
    stage = "① preprocess"
    try:
        pre = preprocess(read_image(path), image_id, out_dir, doc_type=doc_type)
        save_json(out_dir / f"{image_id}_1_preprocess.json", pre.to_dict())
        status[stage] = f"ok ({pre.doc_type}, {pre.corner_method})"
        if pre.doc_type == "other":
            status["kết thúc"] = "tài liệu khác → lưu làm tệp đính kèm"
            return status

        stage = "② clean"
        cleaned = clean(read_image(REPO_DIR / pre.rectified_path), pre.doc_type, image_id, out_dir)
        save_json(out_dir / f"{image_id}_2_clean.json", cleaned.to_dict())
        status[stage] = f"ok ({len(cleaned.components)} thành phần)"

        if pre.doc_type == "lab_report":
            stage = "③ table"
            result = extract_table(cleaned)
            status[stage] = f"ok ({len(result.rows)} dòng chỉ số)"
            boxes = [r.bbox for r in result.rows]
        else:
            stage = "④ prescription"
            result = extract_prescription(cleaned)
            status[stage] = f"ok ({len(result.items)} khoản thuốc)"
            boxes = [it.bbox for it in result.items]
        save_json(out_dir / f"{image_id}_3_result.json", result.to_dict())
        write_image(out_dir / f"{image_id}_overlay.png", draw_boxes(read_image(REPO_DIR / pre.rectified_path), boxes))
    except NotImplementedError as e:
        status[stage] = f"chưa làm: {e}"
    except Exception as e:  # một ảnh lỗi không chặn các ảnh khác
        status[stage] = f"LỖI: {type(e).__name__}: {e}"
    return status


def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline MedTrack đầu-cuối")
    parser.add_argument("input", type=Path, help="ảnh hoặc thư mục ảnh")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-type", choices=DOC_TYPES, default=None)
    args = parser.parse_args()

    paths = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"Không có ảnh nào trong {args.input}")
    for p in paths:
        status = run_one(p, args.out, args.doc_type)
        print(f"\n{status.pop('image_id')}")
        for k, v in status.items():
            print(f"  {k:<16} {v}")


if __name__ == "__main__":
    main()
