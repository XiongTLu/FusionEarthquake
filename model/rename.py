import os
import re
import shutil

# ====== 配置 ======
SRC_DIR = "./ckp_NPP_PWL_fixed_seed"
DST_DIR = "./ckp_NPP_PWL_score_fixed_seed"

MOVE_FILES = False

earthquakes = ["Visso", "Norcia", "Campotosto"]

# 需要处理的4种文件类型
TEMPLATES = [
    "ckp_NPP_PWL_fixed_Mcut{Mcut}_{earthquake}_seed1.h5"
]

# 输出文件名前缀替换规则：ckp_NPP_PWL_fixed_ -> ckp_NPP_PWL_score_fixed_
def rename_to_vp(fname: str) -> str:
    if not fname.startswith("ckp_NPP_PWL_fixed_"):
        raise ValueError(f"Unexpected filename prefix: {fname}")
    return "ckp_NPP_PWL_score_fixed_" + fname[len("ckp_NPP_PWL_fixed_"):]

# frange: [start, stop) step，按 0.1 步进，避免浮点误差
def frange_1dp(start: float, stop: float, step: float = 0.1):
    a = int(round(start * 10))
    b = int(round(stop * 10))
    s = int(round(step * 10))
    for x in range(a, b, s):
        yield round(x / 10.0, 1)

# ================= 主逻辑 =================
def main():
    if not os.path.isdir(SRC_DIR):
        raise FileNotFoundError(f"SRC_DIR not found: {SRC_DIR}")
    os.makedirs(DST_DIR, exist_ok=True)

    total_expected = 0
    n_done = 0
    n_missing = 0
    n_skipped_exist = 0

    for earthquake in earthquakes:
        # ② Mcut随 earthquake 变化
        if earthquake == "Campotosto":
            mcuts = list(frange_1dp(1.3, 3.1, 0.1))  # 1.3 ... 3.0
        else:
            mcuts = list(frange_1dp(1.2, 3.1, 0.1))  # 1.2 ... 3.0

        for Mcut in mcuts:
            # ④ params_cut = Mcut
            params_cut = Mcut

            for tmpl in TEMPLATES:
                total_expected += 1

                src_name = tmpl.format(Mcut=Mcut, params_cut=params_cut, earthquake=earthquake)
                src_path = os.path.join(SRC_DIR, src_name)

                dst_name = rename_to_vp(src_name)
                dst_path = os.path.join(DST_DIR, dst_name)

                if not os.path.exists(src_path):
                    print(f"[MISSING] {src_path}")
                    n_missing += 1
                    continue

                if os.path.exists(dst_path):
                    print(f"[SKIP] Exists: {dst_path}")
                    n_skipped_exist += 1
                    continue

                if MOVE_FILES:
                    shutil.move(src_path, dst_path)
                    print(f"[MOVE] {src_path} -> {dst_path}")
                else:
                    shutil.copy2(src_path, dst_path)
                    print(f"[COPY] {src_path} -> {dst_path}")

                n_done += 1

    print("\n===== Summary =====")
    print(f"Expected (by enumeration): {total_expected}")
    print(f"Done                    : {n_done}")
    print(f"Missing source files    : {n_missing}")
    print(f"Skipped (dst exists)    : {n_skipped_exist}")

if __name__ == "__main__":
    main()