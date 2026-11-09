"""T3 — Khi nào cần ngưỡng cục bộ? F-measure theo mức hỏng (hình chính của Trường). Kế hoạch: KeHoach.md mục 6.2.

Dữ liệu: bảng từng mẫu của T2 (test bộ 5, không biến dạng hình học). So Sauvola với Otsu, có và không khử bóng,
theo từng loại hỏng: bóng, mờ, dấu đỏ, lóa, JPEG, nhiễu. F-measure chấm bỏ dải viền.
Kết quả: results/t3/{by_level.csv, figure.png}

Chạy (từ code/): python experiments/t3_degradation.py   (sau t2_binarize.py)
"""
import matplotlib.pyplot as plt
import pandas as pd

from common.experiment import results_dir, save_config, save_figure, save_table
from common.paths import RESULTS_DIR

EXP_ID = "t3"
CONFIG = {"methods": ["sauvola", "otsu"], "factors": {"shadow_level": "Mức bóng", "blur_level": "Mức mờ",
                                                      "stamp_level": "Mức dấu đỏ", "glare_level": "Mức lóa"}}  # JPEG và nhiễu luôn đi cùng nhau trong bộ 5 → không có mẫu chỉ một loại


def main():
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    df = pd.read_csv(RESULTS_DIR / "t2" / "table.csv")
    df = df[df["method"].isin(CONFIG["methods"])]
    clean_level = {"shadow_level": 0, "blur_level": 0, "stamp_level": 0, "glare_level": 0, "jpeg_quality": 100,
                   "noise_sigma": 0, "pen_level": 0}
    rows = []
    for f in CONFIG["factors"]:
        # chỉ giữ mẫu mà MỌI loại hỏng khác ở mức sạch → đường cong tách riêng ảnh hưởng của loại hỏng f
        others = [o for o in clean_level if o != f and o in df.columns]
        sub = df[(df[others] == pd.Series({o: clean_level[o] for o in others})).all(axis=1)]
        for (lv, m, sr), g in sub.groupby([f, "method", "shadow_removal"]):
            rows.append({"factor": f, "level": lv, "method": m, "shadow_removal": sr, "n": len(g), "f": g["f"].mean()})
    out = pd.DataFrame(rows)
    save_table(out, EXP_ID, "by_level")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    style = {("sauvola", True): ("#c44", "-"), ("otsu", True): ("#36c", "-"),
             ("sauvola", False): ("#c44", "--"), ("otsu", False): ("#36c", "--")}
    for ax, (f, title) in zip(axes.ravel(), CONFIG["factors"].items()):
        for (m, sr), (c, ls) in style.items():
            g = out[(out["factor"] == f) & (out["method"] == m) & (out["shadow_removal"] == sr)]
            ax.plot(g["level"], g["f"], ls, marker="o", color=c,
                    label=f"{m} {'+ khử bóng' if sr else '(không khử bóng)'}")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("F-measure (bỏ dải viền)")
    axes[0].legend(fontsize=7)
    fig.suptitle("T3 — Sauvola so với Otsu theo mức hỏng (test bộ 5)")
    fig.tight_layout()
    save_figure(fig, EXP_ID)
    print(out.pivot_table(index=["factor", "level"], columns=["method", "shadow_removal"], values="f").round(3))


if __name__ == "__main__":
    main()
