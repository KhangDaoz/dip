from __future__ import annotations

import argparse
import html
import json
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import gradio as gr

from common.contracts import BOX_LABELS
from common.paths import SAMPLES_DIR
from pipeline import PRESETS, Pipeline, draw_overlay
from pipeline.core import COLORS

MODES = [
    ("YOLOv8 + VietOCR tinh chỉnh (khuyên dùng)", "dl_c3"),
    ("YOLOv8 + VietOCR gốc", "dl_c2"),
    ("YOLOv8 + Tesseract", "hybrid"),
    ("Cổ điển: Sauvola + Tesseract", "classical"),
]
ENHANCE = [
    ("Không", "none"),
    ("Chỉnh nghiêng", "geom"),
    ("Chỉnh nghiêng + khử bóng", "light"),
    ("Đầy đủ (ảnh chụp điện thoại)", "full"),
]
FIELD_VI = {"drugname": "Tên thuốc", "quantity": "Số lượng", "usage": "Cách dùng",
            "diagnose": "Chẩn đoán", "date": "Ngày", "other": "Khác"}
DETECTOR_SUB = {"yolo": "YOLOv8 · 6 lớp", "classical": "Sauvola + phép giãn"}
OCR_SUB = {"C0": "Tesseract", "C1": "Tesseract", "C2": "VietOCR", "C3": "VietOCR tinh chỉnh"}

_ICONS = {
    "image": '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/>'
             '<path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
    "contrast": '<circle cx="12" cy="12" r="10"/><path d="M12 18a6 6 0 0 0 0-12v12z"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
            '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>',
    "scan-text": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
                 '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
                 '<path d="M7 8h8"/><path d="M7 12h10"/><path d="M7 16h6"/>',
    "braces": '<path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1"/>'
              '<path d="M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    "pill": '<path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/><path d="m8.5 8.5 7 7"/>',
    "calendar": '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/>'
                '<path d="M3 10h18"/>',
    "clipboard": '<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/>'
                 '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
             '<path d="M12 9v4"/><path d="M12 17h.01"/>',
}


def icon(name: str, size: int = 18) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{_ICONS[name]}</svg>')


def _e(s) -> str:
    return html.escape(str(s)) if s not in (None, "") else ""


def _hex(bgr) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


_PIPES: dict = {}


def get_pipe(preset: str, enhance: str, use_lexicon: bool) -> Pipeline:
    key = (preset, enhance, use_lexicon)
    if key not in _PIPES:
        _PIPES[key] = Pipeline.preset(preset, enhance=enhance, **({} if use_lexicon else {"lexicon": None}))
    return _PIPES[key]


STATUS_TEXT = {"ready": "Ready", "processing": "Processing", "completed": "Completed", "error": "Error"}


def status_html(state: str, msg: str = "") -> str:
    return (f"<div class='status {state}'><span class='pill'><span class='dot'></span>{STATUS_TEXT[state]}</span>"
            f"{f'<span class=msg>{_e(msg)}</span>' if msg else ''}</div>")


def header_html(state: str = "ready", msg: str = "Tải ảnh phiếu thuốc lên để bắt đầu") -> str:
    return status_html(state, msg)


HEADER = f"""<div class='hdr'>
  <div class='logo'>{icon('scan-text', 22)}</div>
  <div><div class='title'>Prescription OCR</div><div class='sub'>AI-powered Prescription Understanding</div></div>
</div>"""


def stages_html(state: str = "idle", mode: str = "dl_c3", enhance: str = "none", timing: dict | None = None) -> str:
    p = PRESETS.get(mode, PRESETS["dl_c3"])
    timing = timing or {}
    steps = [
        ("image", "Image Input", "Ảnh quét / ảnh chụp", None),
        ("contrast", "Preprocessing", "Xử lý ảnh cổ điển", "A"),
        ("scan", "Detection", DETECTOR_SUB[p["detector"]], "B"),
        ("scan-text", "OCR", OCR_SUB[p["ocr"]], "C"),
        ("braces", "Structured JSON", "Ghép khoản thuốc", "D"),
    ]
    cells = []
    for i, (ic, name, sub, key) in enumerate(steps):
        skipped = key == "A" and enhance == "none"
        if skipped:
            cls, meta = "skipped", "Bỏ qua"
        elif state == "completed":
            cls = "done"
            meta = "✓ " + (f"{timing[key]:.2f}s" if key in timing else "Đã nhận")
        elif state == "processing":
            cls, meta = ("done", "✓ Đã nhận") if key is None else ("running", "Đang xử lý…")
        elif state == "error":
            cls, meta = "error", "—"
        else:
            cls, meta = "idle", "Chờ"
        if i:
            cells.append(f"<div class='arrow'>{icon('chevron-right', 16)}</div>")
        cells.append(f"<div class='step {cls}'><div class='sic'>{icon(ic, 18)}</div>"
                     f"<div class='stx'><div class='sn'>{name}</div><div class='ss'>{_e(sub)}</div>"
                     f"<div class='sm'>{meta}</div></div></div>")
    return f"<div class='pipe'>{''.join(cells)}</div>"


def section(ic: str, title: str, note: str = "") -> str:
    return (f"<div class='sec'><span class='sec-ic'>{icon(ic, 16)}</span><span>{title}</span>"
            f"{f'<span class=sec-note>{note}</span>' if note else ''}</div>")


LEGEND_HTML = "<div class='legend'>" + "".join(
    f"<span><i style='background:{_hex(COLORS[l])}'></i>{FIELD_VI[l]}</span>" for l in BOX_LABELS
) + "<span><i class='ring'></i>Khoản thuốc (đánh số)</span></div>"


def summary_html(res=None) -> str:
    def cell(ic, label, value, cls=""):
        return (f"<div class='kv {cls}'><div class='kv-l'>{icon(ic, 14)}{label}</div>"
                f"<div class='kv-v' title='{_e(value)}'>{_e(value) or '<span class=muted>—</span>'}</div></div>")
    n = str(len(res.items)) if res is not None else ""
    return ("<div class='summary'>" + cell("pill", "Số thuốc", n, "num")
            + cell("calendar", "Ngày khám", res.date if res else "")
            + cell("clipboard", "Chẩn đoán", res.diagnose if res else "", "wide") + "</div>")


EMPTY_DRUGS = "<div class='empty'>Chưa có dữ liệu. Kết quả sẽ hiện ở đây sau khi nhận diện.</div>"


def drugs_html(res, use_lexicon: bool) -> str:
    if not res.items:
        return "<div class='empty'>Không tìm thấy khoản thuốc nào trên ảnh.</div>"
    rows = []
    for i, it in enumerate(res.items, 1):
        name = it.lexicon_match or it.name
        qty = f"{it.quantity} {_e(it.unit)}".strip() if it.quantity is not None else "—"
        uses = "".join(f"<span class='chip'>{_e(u)}</span>" for u in it.usage) or "<span class='muted'>—</span>"
        flag = (f"<span class='flag' title='Tên không có trong danh mục thuốc, cần đối chiếu với ảnh'>"
                f"{icon('alert', 12)}Cần kiểm tra</span>") if use_lexicon and not it.lexicon_match else ""
        fixed = (f"<div class='fixed'>OCR đọc: {_e(it.name)}</div>"
                 if it.lexicon_match and it.lexicon_match.casefold() != it.name.casefold() else "")
        rows.append(f"<div class='drug'><div class='idx'>{i}</div><div class='dbody'>"
                    f"<div class='dname'>{_e(name)}{flag}</div>{fixed}<div class='chips'>{uses}</div></div>"
                    f"<div class='dqty'><span>Số lượng</span><b>{qty}</b></div></div>")
    return "".join(rows)


def on_image_change(image, mode, enhance):
    msg = "Ảnh đã sẵn sàng. Bấm “Nhận diện & Trích xuất”" if image is not None else "Tải ảnh phiếu thuốc lên để bắt đầu"
    return header_html("ready", msg), stages_html("idle", mode, enhance)


def on_start(image, mode, enhance):
    if image is None:
        return header_html("error", "Chưa có ảnh đầu vào"), stages_html("idle", mode, enhance)
    return header_html("processing", "Đang chạy pipeline…"), stages_html("processing", mode, enhance)


def run(image, mode: str, enhance: str, use_lexicon: bool):
    skip = gr.skip()
    if image is None:
        return (header_html("error", "Chưa có ảnh đầu vào. Hãy tải ảnh lên hoặc chọn ảnh mẫu"),
                stages_html("idle", mode, enhance), skip, skip, skip, skip, skip, skip, skip)
    img = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR if image.shape[-1] == 4 else cv2.COLOR_RGB2BGR) \
        if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    try:
        pipe = get_pipe(mode, enhance, use_lexicon)
        img_a, boxes, timing = pipe.read(img)
        t = time.perf_counter()
        from assemble.group import assemble_result
        res = assemble_result("upload", boxes, lexicon=pipe.lexicon)
        timing["D"] = time.perf_counter() - t
        res.timing = timing
    except Exception as e:
        traceback.print_exc()
        msg = "Chế độ này cần cài Tesseract trên máy" if "tesseract" in str(e).lower() else f"{type(e).__name__}: {e}"
        return (header_html("error", msg), stages_html("error", mode, enhance),
                summary_html(), EMPTY_DRUGS, None, None, None, None, gr.update(value=None, interactive=False))

    overlay = cv2.cvtColor(draw_overlay(img_a, res), cv2.COLOR_BGR2RGB)
    enhanced = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    ocr_rows = [[i, FIELD_VI.get(b.label, "—"), b.text, f"{b.score:.0%}", f"{b.text_conf:.0%}"]
                for i, b in enumerate(sorted(res.boxes, key=lambda b: (b.box[1], b.box[0])), 1)]
    data = res.to_dict()
    path = Path(tempfile.gettempdir()) / "prescription_result.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(timing.values())
    return (header_html("completed", f"{len(res.items)} khoản thuốc · {total:.1f}s"),
            stages_html("completed", mode, enhance, timing), summary_html(res), drugs_html(res, use_lexicon),
            overlay, enhanced, ocr_rows, data, gr.update(value=str(path), interactive=True))


def clear(mode, enhance):
    return (None, header_html(), stages_html("idle", mode, enhance), summary_html(), EMPTY_DRUGS,
            None, None, None, None, gr.update(value=None, interactive=False))


CSS = """
:root {--ink:#0f172a; --ink2:#475569; --ink3:#94a3b8; --line:#e6eaf0; --soft:#f8fafc; --blue:#2563eb;
  --blue-50:#eff6ff; --blue-100:#dbeafe}
footer {display:none !important}
.gradio-container {width:100% !important; max-width:1360px !important; margin:0 auto !important;
  padding:20px 24px !important}
.html-container {padding:0 !important}
/* trạng thái và dải pipeline tự báo tiến độ → không để Gradio làm mờ khi đang chờ */
#status-box, #stages-box, #status-box *, #stages-box * {opacity:1 !important}
#status-box .eta-bar, #stages-box .eta-bar, #status-box .progress-text, #stages-box .progress-text {display:none !important}
button.secondary {border:1px solid #e2e8f0 !important}

/* header */
#topbar {align-items:center; background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px 18px;
  box-shadow:0 1px 2px rgba(15,23,42,.04)}
.hdr {display:flex; align-items:center; gap:14px}
.logo {width:42px; height:42px; border-radius:10px; background:var(--blue); color:#fff; display:grid; place-items:center}
.title {font-size:20px; font-weight:650; color:var(--ink); letter-spacing:-.01em; line-height:1.2}
.sub {font-size:13px; color:var(--ink2)}
.status {display:flex; align-items:center; justify-content:flex-end; gap:10px; font-size:13px; color:var(--ink2)}
.status .pill {display:inline-flex; align-items:center; gap:7px; font-weight:600; padding:4px 11px;
  border-radius:999px; border:1px solid transparent; white-space:nowrap}
.status .dot {width:7px; height:7px; border-radius:50%}
.status .msg {max-width:420px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.status.ready .pill {background:#f8fafc; color:#475569; border-color:#e2e8f0} .status.ready .dot {background:#94a3b8}
.status.processing .pill {background:var(--blue-50); color:var(--blue); border-color:var(--blue-100)}
.status.processing .dot {background:var(--blue); animation:pulse 1.2s ease-in-out infinite}
.status.completed .pill {background:#f0fdf4; color:#15803d; border-color:#bbf7d0}
.status.completed .dot {background:#16a34a}
.status.error .pill {background:#fef2f2; color:#b91c1c; border-color:#fecaca} .status.error .dot {background:#dc2626}
.status.error .msg {color:#b91c1c}
@keyframes pulse {50% {opacity:.35}}

/* pipeline */
.pipe {display:flex; align-items:stretch; gap:6px; background:#fff; border:1px solid var(--line); border-radius:12px;
  padding:12px; box-shadow:0 1px 2px rgba(15,23,42,.04)}
.step {flex:1; min-width:0; display:flex; align-items:flex-start; gap:10px; padding:10px 12px; border-radius:10px;
  border:1px solid var(--line); background:var(--soft)}
.sic {flex:none; width:34px; height:34px; border-radius:8px; display:grid; place-items:center; background:#fff;
  color:var(--ink3); border:1px solid var(--line)}
.stx {flex:1; min-width:0; line-height:1.35}
.sn {font-size:13.5px; font-weight:600; color:var(--ink); white-space:nowrap}
.ss {font-size:12px; color:var(--ink2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.sm {margin-top:3px; font-size:11.5px; color:var(--ink3); font-variant-numeric:tabular-nums; white-space:nowrap}
.arrow {display:grid; place-items:center; color:#cbd5e1}
.step.running {border-color:var(--blue-100); background:var(--blue-50)}
.step.running .sic {color:var(--blue); border-color:var(--blue-100)}
.step.running .sm {color:var(--blue); animation:pulse 1.2s ease-in-out infinite}
.step.done {background:#fff}
.step.done .sic {background:var(--blue-50); color:var(--blue); border-color:var(--blue-100)}
.step.done .sm {color:#15803d; font-weight:600}
.step.skipped {opacity:.55}
.step.error .sic {color:#dc2626}

/* panels */
/* Gradio 6.31 không gắn elem_classes cho Column, nên dùng elem_id */
#left-panel, #right-panel {background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px;
  box-shadow:0 1px 2px rgba(15,23,42,.04); gap:14px}
#dl-btn {flex:none !important; min-width:0 !important; width:auto !important; padding:4px 14px !important}
.sec {display:flex; align-items:center; gap:8px; font-size:14px; font-weight:600; color:var(--ink)}
.sec-ic {width:26px; height:26px; border-radius:7px; display:grid; place-items:center; background:var(--blue-50);
  color:var(--blue)}
.sec-note {margin-left:auto; font-size:12px; font-weight:400; color:var(--ink3)}
#run-btn {min-height:44px; font-weight:600}
.legend {display:flex; flex-wrap:wrap; gap:4px 14px; font-size:12px; color:var(--ink2); padding:2px 2px 6px}
.legend i {display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px}
.legend i.ring {background:transparent; border:2px solid #ff0000; width:10px; height:10px}

/* summary + drugs */
.summary {display:grid; grid-template-columns:110px 150px 1fr; gap:10px}
.kv {border:1px solid var(--line); border-radius:10px; padding:10px 12px; background:var(--soft); min-width:0}
.kv-l {display:flex; align-items:center; gap:6px; font-size:12px; color:var(--ink2)}
.kv-v {margin-top:4px; font-size:14px; font-weight:600; color:var(--ink); overflow:hidden; display:-webkit-box;
  -webkit-line-clamp:2; -webkit-box-orient:vertical; word-break:break-word}
.kv.num .kv-v {font-size:22px; line-height:1.1; color:var(--blue)}
.drug {display:flex; gap:12px; align-items:flex-start; padding:12px 4px; border-bottom:1px solid var(--line)}
.drug:last-child {border-bottom:none}
.idx {flex:none; width:26px; height:26px; border-radius:7px; background:var(--blue-50); color:var(--blue);
  font-size:12.5px; font-weight:700; display:grid; place-items:center}
.dbody {flex:1; min-width:0}
.dname {font-size:14.5px; font-weight:600; color:var(--ink); word-break:break-word}
.fixed {font-size:12px; color:var(--ink3); margin-top:1px}
.chips {display:flex; flex-wrap:wrap; gap:6px; margin-top:6px}
.chip {font-size:12px; color:var(--ink2); background:#f1f5f9; border-radius:6px; padding:2px 8px}
.flag {display:inline-flex; align-items:center; gap:4px; margin-left:8px; font-size:11px; font-weight:500;
  color:#b45309; background:#fffbeb; border:1px solid #fde68a; border-radius:6px; padding:0 6px; vertical-align:2px}
.dqty {flex:none; text-align:right; min-width:72px}
.dqty span {display:block; font-size:11px; color:var(--ink3)}
.dqty b {font-size:14px; color:var(--ink)}
.empty {padding:36px 12px; text-align:center; font-size:13px; color:var(--ink3); border:1px dashed var(--line);
  border-radius:10px; background:var(--soft)}
.muted {color:var(--ink3); font-weight:400}

@media (max-width: 1100px) {.pipe {flex-wrap:wrap} .arrow {display:none} .step {flex:1 1 30%}}
@media (max-width: 720px) {.gradio-container {padding:12px 16px !important} .step {flex:1 1 100%}
  .summary {grid-template-columns:1fr 1fr} .kv.wide {grid-column:1 / -1} .status .msg {display:none}}
"""

FORCE_LIGHT = """() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); }
}"""

THEME = gr.themes.Base(primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate, radius_size="md",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
    body_background_fill="#f5f7fa",
    block_background_fill="#ffffff",
    block_border_color="#e6eaf0",
    block_border_width="1px",
    block_shadow="none",
    block_label_text_color="#475569",
    block_title_text_color="#0f172a",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f8fafc",
    button_secondary_border_color="#e2e8f0",
    button_secondary_text_color="#334155",
)


def build_ui() -> gr.Blocks:
    examples = [[str(p)] for p in sorted(SAMPLES_DIR.glob("*.png"))]
    modes = [m for m in MODES if m[1] in PRESETS]
    with gr.Blocks(title="Prescription OCR") as demo:
        with gr.Row(elem_id="topbar"):
            gr.HTML(HEADER, scale=3, padding=False)
            status = gr.HTML(header_html(), scale=2, padding=False, elem_id="status-box")
        stages = gr.HTML(stages_html(), padding=False, elem_id="stages-box")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_id="left-panel"):
                gr.HTML(section("image", "Ảnh phiếu thuốc"), padding=False)
                inp = gr.Image(type="numpy", height=340, show_label=False, sources=["upload", "clipboard"],
                               placeholder="Kéo thả ảnh vào đây\nhoặc bấm để chọn ảnh", buttons=["fullscreen"])
                with gr.Row():
                    clear_btn = gr.Button("Xóa", variant="secondary", scale=1)
                    btn = gr.Button("Nhận diện & Trích xuất", variant="primary", scale=3, elem_id="run-btn")
                with gr.Accordion("Tùy chọn xử lý", open=False):
                    mode = gr.Dropdown(modes, value="dl_c3", label="Mô hình phát hiện và đọc chữ")
                    enhance = gr.Radio(ENHANCE, value="none", label="Tiền xử lý ảnh")
                    lexicon = gr.Checkbox(value=True, label="Đối chiếu tên thuốc với danh mục")
                if examples:
                    gr.Examples(examples, inputs=inp, label="Ảnh mẫu", examples_per_page=5)

                gr.HTML(section("scan", "Kết quả phát hiện", "Vùng chữ theo lớp"), padding=False)
                with gr.Tabs():
                    with gr.Tab("Detection"):
                        gr.HTML(LEGEND_HTML, padding=False)
                        overlay = gr.Image(show_label=False, height=520, interactive=False,
                                           buttons=["download", "fullscreen"])
                    with gr.Tab("Sau tiền xử lý"):
                        enhanced = gr.Image(show_label=False, height=520, interactive=False,
                                            buttons=["download", "fullscreen"])

            with gr.Column(scale=5, elem_id="right-panel"):
                with gr.Row(equal_height=True):
                    gr.HTML(section("braces", "Kết quả trích xuất"), padding=False, scale=1)
                    dl = gr.DownloadButton("Tải JSON", variant="secondary", size="sm", interactive=False, scale=0,
                                           min_width=96, elem_id="dl-btn")
                summary = gr.HTML(summary_html(), padding=False)
                with gr.Tabs():
                    with gr.Tab("Structured JSON"):
                        js = gr.JSON(show_label=False, open=True, max_height=1150, buttons=["copy"])
                    with gr.Tab("Danh sách thuốc"):
                        drugs = gr.HTML(EMPTY_DRUGS, padding=False)
                    with gr.Tab("Văn bản OCR"):
                        ocr = gr.Dataframe(headers=["#", "Loại vùng", "Nội dung OCR", "Detection", "OCR"],
                                           show_label=False, wrap=True, interactive=False, max_height=1150,
                                           column_widths=["44px", "110px", "auto", "88px", "70px"])

        outs = [status, stages, summary, drugs, overlay, enhanced, ocr, js, dl]
        inp.change(on_image_change, [inp, mode, enhance], [status, stages], queue=False, show_progress="hidden")
        for c in (mode, enhance):
            c.change(lambda m, e: stages_html("idle", m, e), [mode, enhance], stages, queue=False,
                     show_progress="hidden")
        btn.click(on_start, [inp, mode, enhance], [status, stages], queue=False, show_progress="hidden") \
           .then(run, [inp, mode, enhance, lexicon], outs, api_name="run", show_progress="minimal")
        clear_btn.click(clear, [mode, enhance], [inp, *outs], queue=False, api_name=False, show_progress="hidden")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-preload", action="store_true", help="không nạp sẵn pipeline dl_c3 lúc khởi động")
    args = ap.parse_args()
    if not args.no_preload:
        print("Nạp sẵn pipeline dl_c3…")
        get_pipe("dl_c3", "none", True)
    build_ui().queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                              theme=THEME, css=CSS, js=FORCE_LIGHT)
