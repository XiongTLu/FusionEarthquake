import numpy as np
import pandas as pd
import sys
import os

def frange(start, stop, step):
    x = start
    while x < stop - 1e-12:
        yield x
        x += step
import numpy as np
import pandas as pd
import os


type_model = 'NPP_benchmark'

datasets = [
    'ComCat',
    'QTM_SaltonSea',
    'QTM_SanJac',
    'SCEDC',
    'WHITE'
]

mcut_selects = ['min', 'max']
seeds = [1, 26, 123, 1337, 3407]

result_dir = f'./results_{type_model}_seed'
out_dir = './summary_results_benchmark'
os.makedirs(out_dir, exist_ok=True)

# 是否严格要求5个seed结果都存在
# True: 缺一个就报错
# False: 缺失则跳过，但会在结果里记录 missing_seeds
STRICT_MISSING = True


def get_mcut(dataset, mcut_select):
    if dataset == 'ComCat':
        return 2.5 if mcut_select == 'min' else 3.0

    elif dataset == 'QTM_SaltonSea':
        return 1.0 if mcut_select == 'min' else 3.0

    elif dataset == 'QTM_SanJac':
        return 1.0 if mcut_select == 'min' else 3.0

    elif dataset == 'SCEDC':
        return 2.0 if mcut_select == 'min' else 3.0

    elif dataset == 'WHITE':
        return 0.6 if mcut_select == 'min' else 3.0

    else:
        raise ValueError(f'Unknown dataset: {dataset}')


all_seed_rows = []
summary_rows = []

for dataset in datasets:
    for mcut_select in mcut_selects:

        Mcut = get_mcut(dataset, mcut_select)

        seed_rows = []
        missing_seeds = []

        print(
            f"\n========== Summary NPP benchmark: "
            f"dataset={dataset}, Mcut_select={mcut_select}, Mcut={Mcut} =========="
        )

        for seed in seeds:

            results_file_name = (
                f'results_{type_model}_{dataset}_Mcut{Mcut}_seed{seed}.csv'
            )
            results_path = os.path.join(result_dir, results_file_name)

            if not os.path.exists(results_path):
                missing_seeds.append(seed)
                msg = f"[Warning] Missing file: {results_path}"

                if STRICT_MISSING:
                    raise FileNotFoundError(msg)
                else:
                    print(msg)
                    continue

            results_df = pd.read_csv(results_path)

            time_LL = results_df['NPP_time_like'].values
            mag_LL = results_df['NPP_mag_like'].values
            poiss_LL = results_df['LLpoiss'].values[0]

            time_LL_mean = float(np.mean(time_LL))
            mag_LL_mean = float(np.mean(mag_LL))
            poiss_LL_mean = float(poiss_LL)

            one_seed_row = {
                'NPP_time_LL': time_LL_mean,
                'NPP_mag_LL': mag_LL_mean,
                'Poisson_LL': poiss_LL_mean,
            }

            seed_rows.append(one_seed_row)
            all_seed_rows.append(one_seed_row)

            print(
                f"[{dataset}] Mcut_select={mcut_select}, "
                f"Mcut={Mcut}, seed={seed}, "
                f"time LL={time_LL_mean:.6f}, "
                f"mag LL={mag_LL_mean:.6f}, "
                f"Poisson LL={poiss_LL_mean:.6f}"
            )

        if len(seed_rows) == 0:
            print(
                f"[Skip] No valid results for "
                f"dataset={dataset}, Mcut_select={mcut_select}, Mcut={Mcut}"
            )
            continue

        seed_df = pd.DataFrame(seed_rows)

        # 5个seed之间的均值和标准差
        summary_rows.append({
            'NPP_time_LL_mean': seed_df['NPP_time_LL'].mean(),
            'NPP_time_LL_std': seed_df['NPP_time_LL'].std(ddof=1),

            'NPP_mag_LL_mean': seed_df['NPP_mag_LL'].mean(),
            'NPP_mag_LL_std': seed_df['NPP_mag_LL'].std(ddof=1),

            'Poisson_LL_mean': seed_df['Poisson_LL'].mean(),
            'Poisson_LL_std': seed_df['Poisson_LL'].std(ddof=1),
        })


# 保存每个 seed 的原始汇总结果
all_seed_df = pd.DataFrame(all_seed_rows)

seed_out_path = os.path.join(
    out_dir,
    'summary_NPP_benchmark_each_seed.csv'
)

all_seed_df.to_csv(seed_out_path, index=False)
print("\n🚀 Saved each-seed summary:", seed_out_path)


# 保存跨 seed 的均值和标准差结果
summary_df = pd.DataFrame(summary_rows)

summary_out_path = os.path.join(
    out_dir,
    'summary_NPP_benchmark_multi_seed_mean_std.csv'
)

summary_df.to_csv(summary_out_path, index=False)
print("🚀 Saved multi-seed mean/std summary:", summary_out_path)
type_model = 'NPP_trunc'
M0pred = 3.0
time_step = 20
earthquakes = ['Visso', 'Norcia', 'Campotosto']
seed = 1   # 只用单个 seed

out_dir = f'./summary_results'
os.makedirs(out_dir, exist_ok=True)

for earthquake in earthquakes:

    if earthquake == 'Campotosto':
        mcuts = [round(m, 1) for m in frange(1.3, 3.1, 0.1)]
    else:
        mcuts = [round(m, 1) for m in frange(1.2, 3.1, 0.1)]

    rows = []

    for Mcut in mcuts:

        path = f'./results_{type_model}_seed'
        results_file_name = f'results_{type_model}_Mcut{Mcut}_{earthquake}_seed{seed}.csv'
        results_path = os.path.join(path, results_file_name)

        results_df = pd.read_csv(results_path)

        time_LL = results_df['NPP_time_like'].values
        mag_LL = results_df['NPP_mag_like'].values
        poiss_LL = results_df['LLpoiss'].values[0]

        time_LL_mean = float(np.mean(time_LL))
        mag_LL_mean = float(np.mean(mag_LL))

        rows.append({
            'Mcut': Mcut,
            'NPP_time_LL': time_LL_mean,
            'NPP_mag_LL': mag_LL_mean,
            'Poisson_LL': poiss_LL,
        })

        print(f"[{earthquake}] Mcut={Mcut:.1f} time LL={time_LL_mean} mag LL={mag_LL_mean}")

    out_path = os.path.join(
        out_dir,
        f"summary_NPP_seed{seed}_{earthquake}.csv"
    )

    pd.DataFrame(rows).sort_values("Mcut").to_csv(out_path, index=False)
    print("🚀Saved:", out_path)
