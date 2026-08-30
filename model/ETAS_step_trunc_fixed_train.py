import os
import sys
import csv
import math
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

from scipy.integrate import quad
from numpy import trapz as trapezoid

sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2.2)

from utils import ( 
    truncate_catalog_by_threshold,
    datetime_to_hours,
    append_burn_in_to_test_set,
    find_Poisson_MLE,
    collate_likelihoods,
    find_Poisson_likelihood_scores
)

from ETAS_step_trunc_fixed import etas_time_input, etas_mag_input

# ===================== MAG =====================
def save_etas_mag(path, mag_rate_vals_all, mag_intg_vals_all, mag_mask_full):
    pd.DataFrame({
        "mag rate full": np.asarray(mag_rate_vals_all),
        "mag intg full": np.asarray(mag_intg_vals_all),
        "mag mask full": np.asarray(mag_mask_full),
    }).to_csv(path, index=False)

    print(f"✅ Success saveing ETAS full Magnitude inputs 至: {path}")


def load_etas_mag(path):
    df = pd.read_csv(path)
    print(f"✅ Loading etas magnitude inputs: {path}")
    mag_rate_full = df["mag rate full"].to_numpy(float)
    mag_intg_full = df["mag intg full"].to_numpy(float)
    mag_mask_full = df["mag mask full"].to_numpy(bool)
    return mag_rate_full, mag_intg_full, mag_mask_full


# ===================== TIME =====================
def save_etas_time(path, time_rate_vals_all, time_intg_vals_all, time_mask_full):
    pd.DataFrame({
        "time rate full": np.asarray(time_rate_vals_all),
        "time intg full": np.asarray(time_intg_vals_all),
        "time mask full": np.asarray(time_mask_full),
    }).to_csv(path, index=False)
    print(f"✅ Success saveing ETAS Temporal inputs 至: {path}")


def load_etas_time(path):
    df = pd.read_csv(path)
    print(f"✅ Loading etas time inputs: {path}")
    time_rate_full = df["time rate full"].to_numpy(float)
    time_intg_full = df["time intg full"].to_numpy(float)
    time_mask_full = df["time mask full"].to_numpy(bool)
    return time_rate_full, time_intg_full, time_mask_full


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

def main():
    # ===== reading data =====
    AVN_catalog = pd.read_csv('./data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv')
    AVN_catalog = datetime_to_hours(AVN_catalog)
    AVN_catalog = AVN_catalog.dropna()

    # ===== input variable =====
    earthquake = sys.argv[1]
    Mcut = float(sys.argv[2])
    # earthquake = 'Visso'
    # Mcut = 1.2
    M0 = Mcut
    M0pred = 3.0
    time_step = 20
    size_chfn = 3
    size_cmfn = 3

    fixed_params_cut = 3.0
    Mmax = 8.0

    # ===== truncated catalog =====
    truncated_catalog = truncate_catalog_by_threshold(AVN_catalog, Mcut)
    print('Events above 3.0:  ' + str(sum(truncated_catalog['mw'] >= 3.0)))

    times = np.array(truncated_catalog['time'])
    mags  = np.array(truncated_catalog['mw'])
    print('number of datapoints:' + str(times.shape))

    timeupto = select_training_testing_partition(earthquake)
    T_train = times[times < timeupto]
    M_train = mags[times < timeupto]
    T_test  = times[times >= timeupto]
    M_test  = mags[times >= timeupto]

    T_test, M_test = append_burn_in_to_test_set(T_train, T_test, M_train, M_test, time_step)
    print('n train:' + str(T_train.shape))
    print('n test:'  + str(T_test.shape))
    print(f'forecasted events above 3.0: {sum(M_test >= 3.0)}')

    # ===== input ETAS parameters =====
    param_path = f'./ETAS_parameters/params_Mcut_{fixed_params_cut}_{earthquake}.csv'
    with open(param_path) as f:
        reader = csv.reader(f)
        MLE = dict(reader)
        del MLE['train_time']
        params = {k: float(v) for k, v in MLE.items()}
    print(f'ETAS parameters:{params}')


    # =========================================================
    #                 ETAS-trunc fixed inputs
    # =========================================================
    use_cache = True
    etas_input_file_path = f'./etas_inputs_trunc_fixed'
    os.makedirs(etas_input_file_path, exist_ok=True)

    # =========================================================
    #                 ETAS-trunc fixed: magnitude
    # =========================================================
    print("\n--- ETAS-trunc fixed magnitude training set ---")
    etas_train_mag_file = f'ETAS_trunc_fixed_train_mag_Mcut{Mcut}_params{fixed_params_cut}_{earthquake}.csv'
    etas_train_mag_path = os.path.join(etas_input_file_path, etas_train_mag_file)

    if use_cache and os.path.exists(etas_train_mag_path):
        (train_mag_rate_full,
         train_mag_intg_full,
         train_mag_mask_full) = load_etas_mag(etas_train_mag_path)
    else:
        (
         train_mag_rate_full,
         _,
         train_mag_intg_full,
         _,
         train_mag_ll_pred, 
         train_mag_mask_full) = etas_mag_input(
            T_train, M_train,
            Mcut=Mcut, M0pred=M0pred,
            params_3=params,
            time_step=time_step,
            Mmax=Mmax
        )
        save_etas_mag(
            etas_train_mag_path,
            train_mag_rate_full,
            train_mag_intg_full,
            train_mag_mask_full
        )

    print("\n--- ETAS-trunc fixed magnitude test set ---")
    etas_test_mag_file = f'ETAS_trunc_fixed_test_mag_Mcut{Mcut}_params{fixed_params_cut}_{earthquake}.csv'
    etas_test_mag_path = os.path.join(etas_input_file_path, etas_test_mag_file)
    if use_cache and os.path.exists(etas_test_mag_path):
        (
         test_mag_rate_full,
         test_mag_intg_full,
         test_mag_mask_full) = load_etas_mag(etas_test_mag_path)
    else:
        (
         test_mag_rate_full,
         _,
         test_mag_intg_full,
         _,
         test_mag_ll_pred,
         test_mag_mask_full) = etas_mag_input(
            T_test, M_test,
            Mcut=Mcut, M0pred=M0pred,
            params_3=params,
            time_step=time_step,
            Mmax=Mmax
        )
        save_etas_mag(
            etas_test_mag_path,
            test_mag_rate_full,
            test_mag_intg_full,
            test_mag_mask_full
        )

    # =========================================================
    #                 ETAS-trunc fixed: Time
    # =========================================================
    print("\n--- ETAS-trunc fixed temporal training set ---")
    etas_train_time_file = f'ETAS_trunc_fixed_train_time_Mcut{Mcut}_params{fixed_params_cut}_{earthquake}.csv'
    etas_train_time_path = os.path.join(etas_input_file_path, etas_train_time_file)
    if use_cache and os.path.exists(etas_train_time_path):
        (
         train_time_rate_full,
         train_time_intg_full,
         train_time_mask_full) = load_etas_time(etas_train_time_path)
    else:
        (
         train_time_rate_full,
         _,
         train_time_intg_full,
         _,
         train_time_ll_pred,
         train_time_mask_full) = etas_time_input(
            T_train, M_train,
            Mcut=Mcut, M0pred=M0pred,
            params_3=params,
            time_step=time_step,
            Mmax=Mmax,
        )
        save_etas_time(
            etas_train_time_path,
            train_time_rate_full,
            train_time_intg_full,
            train_time_mask_full
        )
    
    print("\n--- trunc-ETAS fixed temporal test set ---")
    etas_test_time_file = f'ETAS_trunc_fixed_test_time_Mcut{Mcut}_params{fixed_params_cut}_{earthquake}.csv'
    etas_test_time_path = os.path.join(etas_input_file_path, etas_test_time_file)
    if use_cache and os.path.exists(etas_test_time_path):
        (
         test_time_rate_full,
         test_time_intg_full,
         test_time_mask_full) = load_etas_time(etas_test_time_path)
    else:
        (
         test_time_rate_full,
         _,
         test_time_intg_full,
         _,
         test_time_ll_pred,
         test_time_mask_full) = etas_time_input(
            T_test, M_test,
            Mcut=Mcut, M0pred=M0pred,
            params_3=params,
            time_step=time_step,
            Mmax=Mmax,
        )
        save_etas_time(
            etas_test_time_path,
            test_time_rate_full,
            test_time_intg_full,
            test_time_mask_full
        )

    # =========================================================
    #           ETAS-trunc fixed 的时间 / 震级 mean LL
    # =========================================================
    time_mean = float(np.mean(test_time_ll_pred))
    mag_mean  = float(np.mean(test_mag_ll_pred))

    print("\n[ETAS-trunc fixed mean log-likelihoods | burn-in removed + masked by M>=M0pred]")
    print(f"ETAS-trunc fixed test  time      mean LL (masked): {time_mean:.6f}  (n={int(test_time_mask_full.sum())})")
    print(f"ETAS-trunc fixed test  magnitude mean LL (masked): {mag_mean:.6f}  (n={int(test_mag_mask_full.sum())})")


        
if __name__ == "__main__":
    main()
