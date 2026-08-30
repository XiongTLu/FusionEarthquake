
import csv
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

import seaborn as sns
sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2.2)

import math
import datetime

import tensorflow as tf

from ETAS import estimate_beta_value, maxlikelihoodETAS
from utils import truncate_catalog_by_threshold, datetime_to_hours, append_burn_in_to_test_set, find_Poisson_MLE, collate_likelihoods, find_Poisson_likelihood_scores

from ETAS_step_trunc_fixed import etas_time_input, etas_mag_input
from Fusion import Fusion

seed = int(sys.argv[3])
# seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

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


def main():
    #################################################################### Reading Data

    AVN_catalog = pd.read_csv('./data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv')

    AVN_catalog = datetime_to_hours(AVN_catalog)

    AVN_catalog = AVN_catalog.dropna()

    ##### Input variables of script
    # earthquake = 'Visso'
    earthquake = sys.argv[1]
    # Mcut = 1.2
    Mcut = float(sys.argv[2])
    M0 = Mcut
    M0pred = 3
    time_step = 20

    Mmax = 8.0
    fixed_params_cut = 3.0

    ##### subsetting catalog by Mcut

    truncated_catalog = truncate_catalog_by_threshold(AVN_catalog, Mcut)
    print('Events above 3.0:  '+str(sum(truncated_catalog['mw'] >= 3.0)))

    times = np.array(truncated_catalog['time'])
    mags = np.array(truncated_catalog['mw'])

    print('number of datapoints:' + str(times.shape))

    timeupto = select_training_testing_partition(earthquake)

    T_train = times[times<timeupto]
    M_train = mags[times<timeupto]
    T_test = times[times>=timeupto]
    M_test = mags[times>=timeupto]
    print('origin n test:' + str(M_test.shape))

    T_test, M_test = append_burn_in_to_test_set(T_train,T_test,M_train,M_test,time_step)

    print('n train:' + str(T_train.shape))
    print('n test:' + str(T_test.shape))

    # ===== input ETAS parameters =====
    param_path = f'./ETAS_parameters/params_Mcut_{fixed_params_cut}_{earthquake}.csv'
    with open(param_path) as f:
        reader = csv.reader(f)
        MLE = dict(reader)
        del MLE['train_time']
        params = {k: float(v) for k, v in MLE.items()}

    # =========================================================
    #                 ETAS trunc fixed inputs
    # =========================================================
    use_cache = True
    etas_input_file_path = f'./etas_inputs_trunc_fixed'
    os.makedirs(etas_input_file_path, exist_ok=True)

    # =========================================================
    #                 ETAS trunc fixed: magnitude
    # =========================================================
    print("\n--- ETAS trunc fixed magnitude training set ---")
    etas_train_mag_file = f'ETAS_trunc_fixed_train_mag_Mcut{Mcut}_params{fixed_params_cut}_{earthquake}.csv'
    etas_train_mag_path = os.path.join(etas_input_file_path, etas_train_mag_file)

    if use_cache and os.path.exists(etas_train_mag_path):
        (etas_train_mag_rate_full,
         etas_train_mag_intg_full,
         etas_train_mag_mask_full) = load_etas_mag(etas_train_mag_path)
    else:
        (
         etas_train_mag_rate_full,
         _,
         etas_train_mag_intg_full,
         _,
         etas_train_mag_ll_pred, 
         etas_train_mag_mask_full) = etas_mag_input(
            T_train, M_train,
            Mcut=Mcut, M0pred=M0pred,
            params_3=params,
            time_step=time_step,
            Mmax=Mmax
        )
        save_etas_mag(
            etas_train_mag_path,
            etas_train_mag_rate_full,
            etas_train_mag_intg_full,
            etas_train_mag_mask_full
        )

    print("\n--- ETAS trunc fixed magnitude test set ---")
    etas_test_mag_file = f'ETAS_trunc_fixed_test_mag_Mcut{Mcut}_params{fixed_params_cut}_{earthquake}.csv'
    etas_test_mag_path = os.path.join(etas_input_file_path, etas_test_mag_file)
    if use_cache and os.path.exists(etas_test_mag_path):
        (
         etas_test_mag_rate_full,
         etas_test_mag_intg_full,
         etas_test_mag_mask_full) = load_etas_mag(etas_test_mag_path)
    else:
        (
         etas_test_mag_rate_full,
         _,
         etas_test_mag_intg_full,
         _,
         etas_test_mag_ll_pred,
         etas_test_mag_mask_full) = etas_mag_input(
            T_test, M_test,
            Mcut=Mcut, M0pred=M0pred,
            params_3=params,
            time_step=time_step,
            Mmax=Mmax
        )
        save_etas_mag(
            etas_test_mag_path,
            etas_test_mag_rate_full,
            etas_test_mag_intg_full,
            etas_test_mag_mask_full
        )

    # =========================================================
    #                 ETAS trunc fixed: Time
    # =========================================================
    print("\n--- ETAS trunc fixed temporal training set ---")
    etas_train_time_file = f'ETAS_trunc_fixed_train_time_Mcut{Mcut}_params{fixed_params_cut}_{earthquake}.csv'
    etas_train_time_path = os.path.join(etas_input_file_path, etas_train_time_file)
    if use_cache and os.path.exists(etas_train_time_path):
        (
         etas_train_time_rate_full,
         etas_train_time_intg_full,
         etas_train_time_mask_full) = load_etas_time(etas_train_time_path)
    else:
        (
         etas_train_time_rate_full,
         _,
         etas_train_time_intg_full,
         _,
         etas_train_time_ll_pred,
         etas_train_time_mask_full) = etas_time_input(
            T_train, M_train,
            Mcut=Mcut, M0pred=M0pred,
            params_3=params,
            time_step=time_step,
            Mmax=Mmax,
            verbose=True
        )
        save_etas_time(
            etas_train_time_path,
            etas_train_time_rate_full,
            etas_train_time_intg_full,
            etas_train_time_mask_full
        )

    print("\n--- ETAS trunc fixed temporal test set ---")
    etas_test_time_file = f'ETAS_trunc_fixed_test_time_Mcut{Mcut}_params{fixed_params_cut}_{earthquake}.csv'
    etas_test_time_path = os.path.join(etas_input_file_path, etas_test_time_file)
    if use_cache and os.path.exists(etas_test_time_path):
        (
         etas_test_time_rate_full,
         etas_test_time_intg_full,
         etas_test_time_mask_full) = load_etas_time(etas_test_time_path)
    else:
        (
         etas_test_time_rate_full,
         _,
         etas_test_time_intg_full,
         _,
         etas_test_time_ll_pred,
         etas_test_time_mask_full) = etas_time_input(
            T_test, M_test,
            Mcut=Mcut, M0pred=M0pred,
            params_3=params,
            time_step=time_step,
            Mmax=Mmax,
            verbose=True
        )
        save_etas_time(
            etas_test_time_path,
            etas_test_time_rate_full,
            etas_test_time_intg_full,
            etas_test_time_mask_full
        )

    if T_train.shape[0] == etas_train_time_rate_full.shape[0]:
        print('\n✅ ETAS trunc fixed training set sizes match!')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Training # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

    ##################### Fusion trunc Model #########################
    ckp_dir = './ckp_Fusion_trunc_fixed_seed/'
    os.makedirs(ckp_dir, exist_ok=True)

    checkpoint = f'ckp_Fusion_trunc_fixed_Mcut{Mcut}_{earthquake}_seed{seed}.h5'
    weight_file = os.path.join(ckp_dir, checkpoint)

    # if True:
    if not os.path.isfile(weight_file):
        print('Training Network')

        start_time = datetime.datetime.now()

        model = (
            Fusion(time_step=time_step,
                   size_rnn=64,
                   size_nn=64,
                   size_layer_chfn=3,
                   size_layer_cmfn=3,
                   M0pred=M0pred,
                   seed=seed)
               .set_train_data(
                   T_train, 
                   M_train,
                   etas_train_time_intg_full,
                   etas_train_mag_intg_full,
                   etas_train_time_rate_full,
                   etas_train_mag_rate_full)
               .set_model(0)
               .compile(lr=1e-3)
               .fit_eval(epochs=400,batch_size=256,plot_training=False)
               .save_weights(ckp_dir, checkpoint))


        end_time = datetime.datetime.now()

        model.train_time = end_time - start_time
    
    else:
        print('Retrieving Network')
        model = (
            Fusion(time_step=time_step,
                   size_rnn=64,
                   size_nn=64,
                   size_layer_chfn=3,
                   size_layer_cmfn=3,
                   M0pred=M0pred,
                   seed=seed)
               .set_train_data(
                   T_train, 
                   M_train,
                   etas_train_time_intg_full,
                   etas_train_mag_intg_full,
                   etas_train_time_rate_full,
                   etas_train_mag_rate_full)
               .set_model(0)
               .load_weights(ckp_dir, checkpoint))        
        model.train_time = 0

    ##############################################################
    poissMLE = find_Poisson_MLE(T_train,M_train,M0pred)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Testing  # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    result_path = './results_Fusion_trunc_fixed_seed/'
    os.makedirs(result_path, exist_ok=True) 
    result_file = f'results_Fusion_trunc_fixed_Mcut{Mcut}_{earthquake}_seed{seed}.csv'
    path_to_results = os.path.join(result_path, result_file)

    pred_path = './pred_Fusion_trunc_fixed_seed/'
    os.makedirs(pred_path, exist_ok=True)
    pred_file = f'pred_Fusion_trunc_fixed_Mcut{Mcut}_{earthquake}_seed{seed}.csv'
    path_to_pred = os.path.join(pred_path, pred_file)   

    # if True:
    if not os.path.isfile(path_to_results):
        print('calculating test likelihoods')

        ##### Poisson  
        Poisson_time_scores = find_Poisson_likelihood_scores(poissMLE,T_test,M_test,M0pred,time_step)

        LLpoiss = Poisson_time_scores.mean()

        print('Poisson Likelihood:   '+str(LLpoiss))

        ##### Fusion BL VP
        model.set_test_data(
            T_test, 
            M_test,
            etas_test_time_intg_full,
            etas_test_mag_intg_full,
            etas_test_time_rate_full,
            etas_test_mag_rate_full
        )
        model.predict_eval(batch_size=512)
        print(f'Fusion trunc fixed Temporal Likelihood: {tf.reduce_mean(model.collated_LLtime).numpy():.4f}')
        print(f'Fusion trunc fixed Magnitude Likelihood: {tf.reduce_mean(model.collated_LLmag).numpy():.4f}')

    else: # read in results
        
        LL_results = pd.read_csv(path_to_results)

        LLpoiss = LL_results.LLpoiss[0]
        print('Poisson Likelihood:   '+str(LLpoiss))

        model.collated_LLtime = LL_results.Fusion_time_like
        print('Fusion trunc fixed Temporal Likelihood:   ' + str(model.collated_LLtime.mean()))

        model.collated_LLmag = LL_results.Fusion_mag_like
        print('Fusion trunc fixed Magnitude Likelihood:  '+str(model.collated_LLmag.mean()))
    ####################################################################### Generate output file
    # if True:
    if not os.path.isfile(path_to_results):
        npredictions = model.collated_LLtime.shape[0]
        print('test size ', npredictions)

        rate_time = np.asarray(model.fusion_rate_time).reshape(-1)
        rate_mag = np.asarray(model.fusion_rate_mag).reshape(-1)
        intg_time = np.asarray(model.fusion_intg_time).reshape(-1)
        intg_mag = np.asarray(model.fusion_intg_mag).reshape(-1)

        LL_results ={
            'Fusion_time_like':model.collated_LLtime,
            'Fusion_mag_like':model.collated_LLmag, 
            'LLpoiss': np.repeat(LLpoiss,npredictions),
            'ntrain': np.repeat(T_train.shape[0],npredictions),
        }

        pred_results = {
            'fusion_temp_rate':rate_time,
            'fusion_temp_intg':intg_time,
            'fusion_mag_rate':rate_mag,
            'fusion_mag_intg':intg_mag,
        }

        print('writing data')

        LL_results = pd.DataFrame(LL_results)
        LL_results.to_csv(path_to_results)
        print(f'✅ successful save LL result as: {path_to_results}')
        pred_results = pd.DataFrame(pred_results)
        pred_results.to_csv(path_to_pred)
        print(f'✅ successful save prediction values as: {path_to_pred}')

    else:
        print(f'🔍 results file already exists: {path_to_results}')
        print(f'🔍 prediction file already exists: {path_to_pred}')


if __name__ == '__main__':
    main()
    
