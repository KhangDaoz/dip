"""<MÃ> — <câu hỏi của thí nghiệm>. Người làm: <tên>. Kế hoạch: team/KeHoach.md mục <x.y>.

Dữ liệu: <bộ, split>. Độ đo: <...>. Kết quả: results/<mã>/.
Chạy (từ code/): python experiments/<mã>_<tên>.py
"""
from common.experiment import results_dir, save_config, save_figure, save_table
from common.seed import set_seed

EXP_ID = "x0"
CONFIG = {"seed": 42, "split": "valid"}


def load_data(config: dict):
    raise NotImplementedError("load_data")


def run(data, config: dict):
    """Trả về pandas DataFrame, mỗi dòng = một mẫu × một phương án."""
    raise NotImplementedError("run")


def plot(df, config: dict):
    raise NotImplementedError("plot")


def main():
    set_seed(CONFIG["seed"])
    results_dir(EXP_ID)
    save_config(CONFIG, EXP_ID)
    df = run(load_data(CONFIG), CONFIG)
    save_table(df, EXP_ID)
    save_figure(plot(df, CONFIG), EXP_ID)


if __name__ == "__main__":
    main()
