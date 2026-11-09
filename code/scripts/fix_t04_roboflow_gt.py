"""Vẽ lại khung khoản thuốc cho 10 ảnh Roboflow có GT lệch trong results/t04 (02/11, Thắng).

Tọa độ [x0, y0, x1, y1] theo ảnh gốc 640×640, đọc trên ảnh có lưới tọa độ rồi kiểm lại bằng overlay.
Bản cũ do AI sinh được giữ ở t04_gt_30_images.v1_ai.json; chạy lại script luôn đọc từ bản cũ nên kết quả tất định.
Chạy (từ thư mục code/): python scripts/fix_t04_roboflow_gt.py

Không còn dùng từ 08/11: bộ Roboflow đơn thuốc Ấn Độ đã bỏ, S1 chỉ đánh giá trên ảnh VAIPE. Giữ lại để tái lập lịch sử.
"""
import shutil

from common.io import load_json, save_json
from common.paths import RESULTS_DIR

P = RESULTS_DIR / "t04" / "t04_gt_30_images.json"
BK = RESULTS_DIR / "t04" / "t04_gt_30_images.v1_ai.json"
if not BK.exists():
    shutil.copyfile(P, BK)
gt = load_json(BK)

def rows(x0, x1, ys):
    return [[x0, a, x1, b] for a, b in zip(ys[:-1], ys[1:])]

FIX = {
    "JUAaXKjfLp": [[66, 47, 553, 180], [66, 182, 553, 267], [66, 268, 553, 421], [66, 422, 553, 523], [66, 524, 553, 639]],
    # bảng nghiêng: mép trên lấy theo đường kẻ bên trái, mép dưới theo bên phải → các khung chồng nhau ~6 px
    "Cr8v9ErGON": [[148, 311, 537, 337], [148, 330, 537, 387], [148, 379, 537, 419], [148, 409, 537, 451], [148, 444, 537, 495], [148, 488, 537, 543]],
    "cyDywHcgUz": rows(48, 546, [284, 318, 350, 380, 418, 449, 482]),
    "2hJUKn84zu": rows(32, 607, [176, 227, 260, 298]),
    "r44Af5FhPQ": [[40, 374, 512, 388], [40, 388, 512, 402], [40, 408, 512, 426]],
    "cfeCMQmVdM": [[401, 205, 586, 250], [334, 241, 506, 283], [345, 272, 639, 347]],
    "7GovIjrjcz": [[221, 256, 533, 291], [209, 302, 531, 328], [248, 332, 542, 361], [160, 396, 553, 425]],
    "k3X27qHyNV": [[60, 171, 470, 220], [82, 219, 489, 277], [42, 268, 477, 322], [42, 313, 482, 366],
                   [42, 365, 491, 391], [42, 404, 494, 440], [60, 452, 504, 499], [55, 499, 506, 541]],
    "4QGAELx2TC": [[374, 260, 554, 320], [398, 318, 554, 368], [416, 370, 568, 434], [422, 430, 526, 472],
                   [424, 488, 568, 532], [430, 544, 562, 572], [444, 590, 542, 618]],
    # Bảng viết tay chật, 7 dòng y lệnh (GT cũ: 6 khung chia đều 25 px). Không đọc chắc được → loại khỏi đánh giá.
    "7GZshzMtAt": rows(11, 557, [336, 350, 361, 384, 395, 409, 420, 462]),
}
EXCLUDE = {"7GZshzMtAt": "Bảng y lệnh viết tay chật, ranh giới khoản không xác định chắc chắn; loại khỏi S1/S3"}

def xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]

for e in gt:
    key = next((k for k in FIX if k in e["candidate_id"]), None)
    e["exclude_from_eval"] = False
    if key is None:
        continue
    boxes = [xywh(b) for b in FIX[key]]
    old = e["items"]
    if len(boxes) != len(old):           # 7GZshzMtAt: 6 → 7 khoản, tên chưa đọc được
        null_fields = {"name": None, "strength": None, "dose": {"morning": None, "noon": None, "afternoon": None, "evening": None},
                       "days": None, "quantity_text": None, "usage_text": None, "verified_transcript": False, "unresolved": True}
        old = [{"fields_gt": dict(null_fields)} for _ in boxes]
    items = []
    for k, (b, o) in enumerate(zip(boxes, old), 1):
        items.append({"item_id": k, "bbox": b,
                      "line_boxes": [{"line_id": 1, "bbox": b, "text": (o["fields_gt"].get("name") or ""), "label": "medicine_item"}],
                      "fields_gt": o["fields_gt"]})
    e["items"] = items
    e["n_items"] = len(items)
    e["n_lines"] = len(items)
    r = e["medicine_region_bbox"]
    xs0 = min([r[0]] + [b[0] for b in boxes]); ys0 = min([r[1]] + [b[1] for b in boxes])
    xs1 = max([r[0] + r[2]] + [b[0] + b[2] for b in boxes]); ys1 = max([r[1] + r[3]] + [b[1] + b[3] for b in boxes])
    e["medicine_region_bbox"] = [xs0, ys0, xs1 - xs0, ys1 - ys0]
    e["unresolved_items_count"] = sum(bool(i["fields_gt"].get("unresolved")) for i in items)
    e["gt_revision"] = "2026-11-02: vẽ lại khung khoản theo ảnh (bản AI cũ: t04_gt_30_images.v1_ai.json)"
    if key in EXCLUDE:
        e["exclude_from_eval"] = True
        e["exclude_reason"] = EXCLUDE[key]
save_json(P, gt)
print("ok", sum(e.get("gt_revision") is not None for e in gt), "ảnh sửa")
