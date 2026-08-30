import os
import shutil
import pandas as pd

# ===================== 配置 =====================

# 原目录 → 新目录
OLD_DIR = "../results_NPP_PWL_VP_seed"
NEW_DIR = "../results_NPP_PWL_score_VP_seed"

MOVE_FILES = False   # True=move  False=copy

earthquakes = ["Visso", "Norcia", "Campotosto"]
seeds = [1]  # 可扩展

# 新header
NEW_HEADER = [
    "NPP_PWL_score_time_like",
    "NPP_PWL_score_mag_like",
    "LLpoiss",
    "ntrain"
]


# ===================== Mcut生成 =====================
def frange_1dp(start: float, stop: float, step: float = 0.1):
    a = int(round(start * 10))
    b = int(round(stop * 10))
    s = int(round(step * 10))
    for x in range(a, b, s):
        yield round(x / 10.0, 1)


# ===================== 主逻辑 =====================
def main():

    if not os.path.exists(OLD_DIR):
        raise FileNotFoundError(f"{OLD_DIR} not found")

    # -------- Step 1: 重命名文件夹 --------
    if not os.path.exists(NEW_DIR):
        os.rename(OLD_DIR, NEW_DIR)
        print(f"[DIR RENAME] {OLD_DIR} → {NEW_DIR}")
    else:
        print(f"[DIR EXISTS] Using {NEW_DIR}")

    total_expected = 0
    n_done = 0
    n_missing = 0
    n_skipped = 0

    for earthquake in earthquakes:

        # Mcut规则
        if earthquake == "Campotosto":
            mcuts = list(frange_1dp(1.3, 3.1, 0.1))
        else:
            mcuts = list(frange_1dp(1.2, 3.1, 0.1))

        for Mcut in mcuts:
            for seed in seeds:

                total_expected += 1

                # 原文件名
                old_fname = (
                    f"results_NPP_PWL_score_VP_"
                    f"Mcut{Mcut}_{earthquake}_seed{seed}.csv"
                )
                old_path = os.path.join(NEW_DIR, old_fname)

                # 新文件名
                new_fname = (
                    f"results_NPP_PWL_score_VP_"
                    f"Mcut{Mcut}_{earthquake}_seed{seed}.csv"
                )
                new_path = os.path.join(NEW_DIR, new_fname)

                if not os.path.exists(old_path):
                    print(f"[MISSING] {old_path}")
                    n_missing += 1
                    continue

                if os.path.exists(new_path):
                    print(f"[SKIP] Exists: {new_path}")
                    n_skipped += 1
                    continue

                # -------- Step 2: 读取并修改header --------
                try:
                    df = pd.read_csv(old_path)
                except Exception as e:
                    print(f"[READ ERROR] {old_fname}: {e}")
                    continue

                if len(df.columns) != 4:
                    print(f"[WARNING] {old_fname} column count != 4")

                df.columns = NEW_HEADER

                # -------- Step 3: 写入新文件 --------
                df.to_csv(new_path, index=False)

                # 删除旧文件
                if MOVE_FILES:
                    os.remove(old_path)

                print(f"[UPDATED] {old_fname} → {new_fname}")

                n_done += 1

    print("\n===== Summary =====")
    print(f"Expected (enumerated): {total_expected}")
    print(f"Done                 : {n_done}")
    print(f"Missing              : {n_missing}")
    print(f"Skipped              : {n_skipped}")


if __name__ == "__main__":
    main()
