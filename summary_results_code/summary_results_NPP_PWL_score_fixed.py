import numpy as np
import pandas as pd
import sys
import os

def frange(start, stop, step):
    x = start
    while x < stop - 1e-12:
        yield x
        x += step


def compute_NPP_PWL(earthquake, Mcut):
    """
    Single-run calculator (seed = 1 only)
    """

    seed = 1

    path = f'./results_NPP_PWL_score_fixed_seed'
    results_file_name = f'results_NPP_PWL_score_fixed_Mcut{Mcut}_{earthquake}_seed{seed}.csv'
    results_path = os.path.join(path, results_file_name)

    results_df = pd.read_csv(results_path)

    time_LL = results_df['NPP_PWL_score_time_like'].values
    mag_LL = results_df['NPP_PWL_score_mag_like'].values
    poiss_LL = results_df['LLpoiss'].values[0]

    time_LL_mean = float(np.mean(time_LL))
    mag_LL_mean = float(np.mean(mag_LL))

    return time_LL_mean, mag_LL_mean, poiss_LL


def main():

    earthquakes = ['Visso', 'Norcia', 'Campotosto']

    out_dir = f'./summary_results'
    os.makedirs(out_dir, exist_ok=True)

    for earthquake in earthquakes:

        if earthquake == 'Campotosto':
            mcuts = [round(m, 1) for m in frange(1.3, 3.1, 0.1)]
        else:
            mcuts = [round(m, 1) for m in frange(1.2, 3.1, 0.1)]

        rows = []

        for Mcut in mcuts:

            time_LL, mag_LL, poiss_LL = compute_NPP_PWL(
                earthquake,
                Mcut
            )

            rows.append({
                'Mcut': Mcut,
                'NPP_PWL_score_fixed_time_LL': time_LL,
                'NPP_PWL_score_fixed_mag_LL': mag_LL,
                'Poisson_LL': poiss_LL,
            })

            print(f"[{earthquake}] Mcut={Mcut:.1f} "
                  f"time LL={time_LL:.6f} "
                  f"mag LL={mag_LL:.6f}")

        out_path = os.path.join(
            out_dir,
            f"summary_NPP_PWL_score_fixed_seed1_{earthquake}.csv"
        )

        pd.DataFrame(rows).sort_values("Mcut").to_csv(out_path, index=False)
        print("🚀 Saved:", out_path)


if __name__ == "__main__":
    main()
