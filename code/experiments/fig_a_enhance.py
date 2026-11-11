import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.experiment import results_dir, save_figure
from common.io import read_image, to_gray
from common.paths import VAIPE_DEGRADED
from enhance.clahe import clahe
from enhance.deskew import deskew
from enhance.rectify import rectify
from enhance.shadow import background_closing, remove_shadows

EXP_ID = "fig_a"


def show(ax, img, title):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
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

    fig, axs = plt.subplots(1, 3, figsize=(8.5, 4.2))
    for ax, (lv, im) in zip(axs, levels.items()):
        show(ax, im, lv)
    fig.tight_layout()
    save_figure(fig, EXP_ID, "a_levels")

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
