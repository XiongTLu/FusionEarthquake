import os
import csv
import numpy as np
import pandas as pd


def frange(start, stop, step):
    """float range: [start, stop) with step, safe for rounding to 1 decimal."""
    x = start
    while x < stop - 1e-12:
        yield x
        x += step


def compute_etas_trunc(type_model, earthquake, Mcut, M0pred, Mmax, time_step):
    """
    Single-run calculator
    """

    # params id used in filenames
    if type_model == 'trunc_fixed':
        params_id = 3.0
    elif type_model == 'trunc_VP':
        params_id = Mcut

    # ----- load saved inputs -----
    path = f'./etas_inputs_{type_model}'
    test_mag_file_name = f'ETAS_{type_model}_test_mag_Mcut{Mcut}_params{params_id}_{earthquake}.csv'
    test_time_file_name = f'ETAS_{type_model}_test_time_Mcut{Mcut}_params{params_id}_{earthquake}.csv'

    mag_path = os.path.join(path, test_mag_file_name)
    time_path = os.path.join(path, test_time_file_name)

    test_mag_df  = pd.read_csv(mag_path)
    test_time_df = pd.read_csv(time_path)

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

    beta_3 = params['beta']
    Delta = Mmax - Mcut
    delta = M0pred - Mcut
    Z_norm = 1.0 - np.exp(-beta_3 * Delta)
    p_3_cut = (np.exp(-beta_3 * delta) - np.exp(-beta_3 * Delta)) / Z_norm
    pd_tr = (np.exp(-beta_3 * delta) - np.exp(-beta_3 * Delta)) / Z_norm

    # ----- cut mapping in segment-end space -----
    cut = int(time_step) + 1
    count_mag = int(mask_full[:cut].sum())
    count_time = int(mask_full[1:cut].sum())

    # ======================
    #   time LL prediction
    # ======================
    time_log_part = np.log(pd_tr * time_rate_full)

    seg_sum = 0.0
    time_LL_step_full = []  # only at segment ends
    for i in range(1, len(time_rate_full)):
        seg_sum += float(time_intg_full[i])
        if mask_full[i]:
            time_LL_step_full.append(time_log_part[i] - pd_tr * seg_sum)
            seg_sum = 0.0

    time_LL_step_full = np.asarray(time_LL_step_full, dtype=np.float64)
    time_LL_after_cut = time_LL_step_full[count_time:]
    time_LL_pred = float(time_LL_after_cut.mean()) 
    print(f'ETAS {type_model} time LL prediction (Mcut={Mcut}, M0pred={M0pred}): {time_LL_pred}')

    # ======================
    #   mag LL prediction
    # ======================
    mag_LL_full = np.log(mag_rate_full[mask_full] / pd_tr)
    mag_LL_after_cut = mag_LL_full[count_mag:]
    mag_LL_pred = float(mag_LL_after_cut.mean())
    print(f'ETAS {type_model} mag LL prediction (Mcut={Mcut}, M0pred={M0pred}): {mag_LL_pred}')

    return time_LL_pred, mag_LL_pred


def main():
    # ===== settings =====  
    type_model = 'trunc_VP'      
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
            time_ll, mag_ll = compute_etas_trunc(
                type_model=type_model,
                earthquake=earthquake,
                Mcut=Mcut,
                M0pred=M0pred,
                Mmax=Mmax,
                time_step=time_step
            )

            rows.append({
                "Mcut": Mcut,
                "ETAS_VP_time_LL": time_ll,
                "ETAS_VP_mag_LL": mag_ll,
            })

            print(f"[{earthquake}] Mcut={Mcut:.1f} time={time_ll} mag={mag_ll}")

        out_path = os.path.join(
            out_dir,
            f"summary_ETAS_VP_{earthquake}.csv"
        )
        pd.DataFrame(rows).sort_values("Mcut").to_csv(out_path, index=False)
        print("Saved:", out_path)

if __name__ == "__main__":
    main()
