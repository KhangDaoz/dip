"""Demo Gradio: tải ảnh đơn thuốc → bảng khoản thuốc + JSON + ảnh overlay (plan §7.1).

    python app.py            # mở http://127.0.0.1:7860
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402

from common.paths import SAMPLES_DIR  # noqa: E402
from pipeline import PRESET_PIPELINES, Pipeline, draw_overlay  # noqa: E402

_PIPES = {}


def get_pipe(preset, enhance):
    key = (preset, enhance)
    if key not in _PIPES:
        _PIPES[key] = Pipeline.preset(preset, enhance=enhance)
    return _PIPES[key]


def run(image_rgb, preset, enhance):
    if image_rgb is None:
        return None, [], {}
    img = image_rgb[:, :, ::-1].copy()
    if preset == "vlm":
        from assemble.vlm import extract
        res = extract(img, "upload")[0]
    else:
        res = get_pipe(preset, enhance)(img, "upload")
    rows = [[i, it.lexicon_match or it.name, it.quantity, it.unit or "", " / ".join(it.usage)]
            for i, it in enumerate(res.items, 1)]
    d = res.to_dict()
    d.pop("boxes", None)
    return draw_overlay(img, res)[:, :, ::-1], rows, d


examples = [[str(p), "dl", "none"] for p in sorted(SAMPLES_DIR.glob("*.png"))[:6]]

with gr.Blocks(title="Trích xuất đơn thuốc") as demo:
    gr.Markdown("## Trích xuất thông tin đơn thuốc VAIPE\nYOLOv8 → VietOCR → luật ghép khoản · đối chứng cổ điển / VLM")
    with gr.Row():
        inp = gr.Image(label="Ảnh đơn thuốc", type="numpy")
        out_img = gr.Image(label="Vùng chữ & khoản thuốc")
    with gr.Row():
        preset = gr.Dropdown(list(PRESET_PIPELINES) + ["vlm"], value="dl", label="Pipeline")
        enhance = gr.Dropdown(["none", "geom", "light", "full"], value="none", label="Tiền xử lý (khối A)")
        btn = gr.Button("Chạy", variant="primary")
    table = gr.Dataframe(headers=["#", "Tên thuốc", "SL", "Đơn vị", "Cách dùng"], label="Khoản thuốc")
    js = gr.JSON(label="JSON")
    btn.click(run, [inp, preset, enhance], [out_img, table, js])
    if examples:
        gr.Examples(examples, [inp, preset, enhance])

if __name__ == "__main__":
    demo.launch()
