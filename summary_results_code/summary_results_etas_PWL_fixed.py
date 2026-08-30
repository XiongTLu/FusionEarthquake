import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # E:\Fusion
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv
import numpy as np
import pandas as pd

from model.utils import ( 
    truncate_catalog_by_threshold,
    datetime_to_hours,
    append_burn_in_to_test_set,
    find_Poisson_MLE,
    collate_likelihoods,
    find_Poisson_likelihood_scores
)

def frange(start, stop, step):
    """float range: [start, stop) with step, safe for rounding to 1 decimal."""
    x = start
    while x < stop - 1e-12:
        yield x
        x += step

def select_training_testing_partition(earthquake):
    if earthquake == 'Visso':
        timeupto = 1200
    elif earthquake == 'Norcia':
        timeupto = 1800
    elif earthquake == 'Campotosto':
        timeupto = 3600
    else:
        print('Invalid argument')
        sys.exit()
    return timeupto

def compute_etas_PWL(type_model, Mdat, earthquake, Mcut, M0pred, Mmax, time_step):
    """
    Single-run calculator
    """

    # params id used in filenames
    if type_model == 'PWL_fixed':
        params_id = 3.0
    elif type_model == 'PWL_VP':
        params_id = Mcut

    # ----- load saved inputs -----
    path = f'./etas_inputs_{type_model}'
    test_mag_file_name = f'ETAS_{type_model}_test_mag_Mcut{Mcut}_params{params_id}_{earthquake}.csv'
    test_time_file_name = f'ETAS_{type_model}_test_time_Mcut{Mcut}_params{params_id}_{earthquake}.csv'

    mag_path = os.path.join(path, test_mag_file_name)
    time_path = os.path.join(path, test_time_file_name)

    test_mag_df  = pd.read_csv(mag_path)
    test_time_df = pd.read_csv(time_path)

    if len(Mdat) != len(test_time_df):
        raise ValueError(f"Mdat length mismatch: Mdat={len(Mdat)} csv={len(test_time_df)} "
                     f"for {earthquake}, Mcut={Mcut}")

    mag_rate_full  = test_mag_df['mag rate full'].values
    mag_intg_full  = test_mag_df['mag intg full'].values
    time_rate_full = test_time_df['time rate full'].values
    time_intg_full = test_time_df['time intg full'].values
    mask_full      = test_mag_df['mag mask full'].values.astype(bool)

    if len(test_mag_df) != len(test_time_df):
        raise ValueError(f"Length mismatch: mag={len(test_mag_df)} time={len(test_time_df)} "
                         f"for {earthquake}, Mcut={Mcut}")

    # ----- load ETAS parameters (beta) -----
    param_path = f'./ETAS_parameters/params_Mcut_{params_id}_{earthquake}.csv'
    with open(param_path) as f:
        reader = csv.reader(f)
        MLE = dict(reader)
        del MLE['train_time']
        params = {k: float(v) for k, v in MLE.items()}
    # print(f'ETAS parameters:{params}')

    eps = 1e-10

    beta_3 = params['beta']
    Mmax = float(Mmax)
    Delta = Mmax - Mcut
    delta = M0pred - Mcut
    gamma = 1.5 * np.log(10.0)
    alpha = gamma - beta_3  
    Z_norm = 1.0 - np.exp(-beta_3 * Delta)

    if Mcut >= M0pred:
        p_3_cut = 1.0
    else:
        p_3_cut = (np.exp(-beta_3 * delta) - np.exp(-beta_3 * Delta)) / Z_norm

    w_m_full = np.exp(gamma * (Mdat - M0pred))
    C_wd = (
            beta_3 * np.exp(gamma * (Mcut - M0pred)) * (np.exp(alpha * Delta) - np.exp(alpha * delta))
        ) / (
            alpha * (np.exp(-beta_3 * delta) - np.exp(-beta_3 * Delta))
        )

    p_dw = (np.exp(-beta_3 * delta) - np.exp(-beta_3 * Delta)) / Z_norm

    # ----- cut mapping in segment-end space -----
    cut = int(time_step) + 1
    count_mag = int(mask_full[:cut].sum())
    count_time = int(mask_full[1:cut].sum())

    # ======================
    #   time LL prediction
    # ======================
    time_log_part = np.log(p_3_cut * time_rate_full + eps)

    seg_sum = 0.0
    time_LL_step_full = []  # only at segment ends
    for i in range(1, len(time_rate_full)):
        seg_sum += float(time_intg_full[i])
        w_i = w_m_full[i]
        if mask_full[i]:
            time_LL_step_full.append(w_i * time_log_part[i] - C_wd * p_3_cut * seg_sum)
            seg_sum = 0.0

    time_LL_step_full = np.asarray(time_LL_step_full, dtype=np.float64)
    time_LL_after_cut = time_LL_step_full[count_time:]
    time_LL_pred = float(time_LL_after_cut.mean()) 
    print(f'ETAS {type_model} time LL prediction (Mcut={Mcut}, M0pred={M0pred}): {time_LL_pred}')

    # ======================
    #   mag LL prediction
    # ======================
    w_mask = w_m_full[mask_full]
    mag_LL_full = w_mask * np.log(mag_rate_full[mask_full] / p_3_cut)
    mag_LL_after_cut = mag_LL_full[count_mag:]
    mag_LL_pred = float(mag_LL_after_cut.mean())
    print(f'ETAS {type_model} mag LL prediction (Mcut={Mcut}, M0pred={M0pred}): {mag_LL_pred}')

    return time_LL_pred, mag_LL_pred


def main():
    # ===== reading data =====
    AVN_catalog = pd.read_csv('./data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv')
    AVN_catalog = datetime_to_hours(AVN_catalog)
    AVN_catalog = AVN_catalog.dropna()

    # ===== settings =====
    type_model = 'PWL_fixed'       
    M0pred = 3.0
    time_step = 20
    Mmax = 8.0

    earthquakes = ['Visso', 'Norcia', 'Campotosto']

    out_dir = f'./summary_results'
    os.makedirs(out_dir, exist_ok=True)

    for earthquake in earthquakes:
        # define mcuts
        if earthquake == 'Campotosto':
            mcuts = [round(m, 1) for m in frange(1.3, 3.1, 0.1)]
        else:
            mcuts = [round(m, 1) for m in frange(1.2, 3.1, 0.1)]

        rows = []
        for Mcut in mcuts:
            # ===== truncated catalog =====
            truncated_catalog = truncate_catalog_by_threshold(AVN_catalog, Mcut)
            # print('Events above 3.0:  ' + str(sum(truncated_catalog['mw'] >= 3.0)))

            times = np.array(truncated_catalog['time'])
            mags  = np.array(truncated_catalog['mw'])
            # print('number of datapoints:' + str(times.shape))

            timeupto = select_training_testing_partition(earthquake)
            T_train = times[times < timeupto]
            M_train = mags[times < timeupto]
            T_test  = times[times >= timeupto]
            M_test  = mags[times >= timeupto]

            T_test, M_test = append_burn_in_to_test_set(T_train, T_test, M_train, M_test, time_step)
            # print('n train:' + str(T_train.shape))
            # print('n test:'  + str(T_test.shape))
            # print(f'forecasted events above 3.0: {sum(M_test >= 3.0)}')

            time_ll, mag_ll = compute_etas_PWL(
                type_model=type_model,
                Mdat=M_test,
                earthquake=earthquake,
                Mcut=Mcut,
                M0pred=M0pred,
                Mmax=Mmax,
                time_step=time_step
            )

            rows.append({
                "Mcut": Mcut,
                "ETAS_PWL_fixed_time_LL": time_ll,
                "ETAS_PWL_fixed_mag_LL": mag_ll,
            })

            print(f"[{earthquake}] Mcut={Mcut:.1f} time={time_ll} mag={mag_ll}")

        out_path = os.path.join(
            out_dir,
            f"summary_ETAS_{type_model}_{earthquake}.csv"
        )
        pd.DataFrame(rows).sort_values("Mcut").to_csv(out_path, index=False)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
