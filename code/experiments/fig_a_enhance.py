"""Hình minh hoạ khối A cho báo cáo / slide.

Hình 1 (a_steps.png): ảnh test L4 qua từng bước A1–A5 (gốc L4 → nắn phối cảnh → chỉnh nghiêng → khử bóng → CLAHE).
Hình 2 (a_levels.png): cùng một ảnh ở 3 mức suy giảm L0 / L2 / L4.
Hình 3 (a_shadow.png): khử bóng — ảnh có bóng, nền ước lượng bằng phép đóng, kết quả chia nền.

    python experiments/fig_a_enhance.py                        # ảnh mặc định
    python experiments/fig_a_enhance.py --image VAIPE_P_TRAIN_129
Kết quả: results/fig_a/*.png (dpi 200)
"""
import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.experiment import results_dir, save_figure  # noqa: E402
from common.io import read_image, to_gray  # noqa: E402
from common.paths import VAIPE_DEGRADED  # noqa: E402
from enhance.clahe import clahe  # noqa: E402
from enhance.deskew import deskew  # noqa: E402
from enhance.rectify import rectify  # noqa: E402
from enhance.shadow import background_closing, remove_shadows  # noqa: E402

EXP_ID = "fig_a"


def show(ax, img, title):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)   # thang cố định: không tự kéo giãn tương phản
    else:
        ax.imshow(img[:, :, ::-1])
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="image_id trong tập test (mặc định: ảnh đầu tiên)")
    args = ap.parse_args()
    results_dir(EXP_ID)
    paths = sorted((VAIPE_DEGRADED / "L4").glob("*.png"))
    if not paths:
        raise SystemExit("Thiếu ảnh suy giảm — chạy tools/make_degraded.py")
    name = f"{args.image}.png" if args.image else paths[0].name
    levels = {lv: read_image(VAIPE_DEGRADED / lv / name) for lv in ("L0", "L2", "L4")}

    # Hình 1: các bước trên ảnh L4
    a0 = levels["L4"]
    a1, method = rectify(a0)
    a2, ang = deskew(a1)
    a3 = remove_shadows(a2)
    a5 = clahe(a3)
    fig, axs = plt.subplots(1, 5, figsize=(13, 4.2))
    for ax, im, t in zip(axs, [a0, a1, a2, a3, a5],
                         ["Ảnh L4", f"A1 nắn phối cảnh ({method})", f"A2 chỉnh nghiêng ({ang:+.2f}°)",
                          "A3 khử bóng (đóng, SE 81)", "A5 CLAHE (clip 2,0)"]):
        show(ax, im, t)
    fig.tight_layout()
    save_figure(fig, EXP_ID, "a_steps")

    # Hình 2: 3 mức suy giảm
    fig, axs = plt.subplots(1, 3, figsize=(8.5, 4.2))
    for ax, (lv, im) in zip(axs, levels.items()):
        show(ax, im, lv)
    fig.tight_layout()
    save_figure(fig, EXP_ID, "a_levels")

    # Hình 3: khử bóng trên kênh xám của ảnh L4 đã nắn + chỉnh nghiêng
    g = to_gray(a2)
    bg = background_closing(g, kernel=81)
    flat = remove_shadows(g)
    fig, axs = plt.subplots(1, 3, figsize=(8.5, 4.2))
    for ax, im, t in zip(axs, [g, bg, flat], ["Ảnh xám có bóng", "Nền ước lượng (phép đóng)", "Ảnh / nền × 255"]):
        show(ax, im, t)
    fig.tight_layout()
    save_figure(fig, EXP_ID, "a_shadow")
    print(f"Ảnh {name}: rectify={method}, skew={ang:+.2f}° → {results_dir(EXP_ID)}")


if __name__ == "__main__":
    main()
