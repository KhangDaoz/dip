"""E-X1 — ablation: lỗi đầu-cuối đến từ khối nào? (plan §5.5)

Thay khối B / C của pipeline chính (YOLO + C3) bằng đáp án chuẩn, chấm F1 khoản thuốc trên 176 ảnh test:
    B chuẩn + C chuẩn    box + chữ của nhãn                           (= E-D1, kiểm tra khối D)
    B chuẩn + C3         box nhãn (có lớp) + VietOCR fine-tune đọc   → lỗi riêng của khối C
    YOLO    + C chuẩn    box YOLO, chữ lấy từ box đáp án ghép IoU≥0,5 → lỗi riêng của khối B
    YOLO    + C3         pipeline chính                              (= dòng dl_c3 của E-D2)
Cần: results/handoff/dl_c3.jsonl (python tools/make_handoff.py --preset dl_c3) và items_gt.json.
Khối D bật từ điển (như E-D2); cột f1_item_nolex là khi tắt từ điển.

    python experiments/e_x1_ablation.py
Kết quả: results/e_x1/{table.csv, config.json}; cache results/handoff/gtbox_c3.jsonl
"""
import json
import time

import pandas as pd

from _common import assemble_handoff, load_gt, score
from assemble.group import assemble
from assemble.lexicon import Lexicon
from common.contracts import TextBox
from common.experiment import results_dir, save_config, save_table
from common.io import read_image
from common.paths import RESULTS_DIR
from pipeline import get_engine, handoff_record
from vaipe.dataset import list_images

EXP_ID = "e_x1"


def gtbox_c3(images, path):
    """Box đáp án (giữ lớp) → VietOCR C3 đọc → file bàn giao (cache, chạy tiếp được)."""
    done = set()
    if path.exists():
        done = {json.loads(l)["image_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [im for im in images if im.image_id not in done]
    if todo:
        print(f"[gtbox_c3] OCR C3 trên box chuẩn: còn {len(todo)} ảnh")
        ocr, t0 = get_engine("C3"), time.perf_counter()
        with open(path, "a", encoding="utf-8") as f:
            for k, im in enumerate(todo, 1):
                boxes = [TextBox(box=list(a.box), label=a.label, score=1.0) for a in im.annotations]
                ocr.recognize(read_image(im.image_path), boxes)
                f.write(json.dumps(handoff_record(im.image_id, boxes), ensure_ascii=False) + "\n")
                if k % 25 == 0:
                    print(f"  {k}/{len(todo)} ({(time.perf_counter() - t0) / k:.1f}s/ảnh)")


def main():
    gt = load_gt("test")
    images = [im for im in list_images("test") if im.image_id in gt]
    handoff = RESULTS_DIR / "handoff" / "dl_c3.jsonl"
    if not handoff.exists():
        raise SystemExit(f"Thiếu {handoff} — chạy: python tools/make_handoff.py --preset dl_c3")
    lex = Lexicon.from_split("train")
    gtc3 = RESULTS_DIR / "handoff" / "gtbox_c3.jsonl"
    gtbox_c3(images, gtc3)

    rows = [
        score("B chuẩn + C chuẩn", {im.image_id: assemble(
            [TextBox(box=list(a.box), label=a.label, score=1.0, text=a.text) for a in im.annotations],
            lexicon=lex) for im in images}, gt),
        score("B chuẩn + C3", assemble_handoff(gtc3, lex), gt),
        score("YOLO + C chuẩn", assemble_handoff(handoff, lex, use_gt_text=True), gt),
        score("YOLO + C3 (pipeline chính)", assemble_handoff(handoff, lex), gt),
    ]
    results_dir(EXP_ID)
    save_table(pd.DataFrame(rows), EXP_ID)
    save_config({"split": "test", "n_images": len(images), "handoff": str(handoff), "lexicon_size": len(lex)}, EXP_ID)


if __name__ == "__main__":
    main()
