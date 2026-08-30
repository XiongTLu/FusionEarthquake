
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

import ETAS
from ETAS import estimate_beta_value, maxlikelihoodETAS
from utils import truncate_catalog_by_threshold, datetime_to_hours, append_burn_in_to_test_set, find_Poisson_MLE, collate_likelihoods, find_Poisson_likelihood_scores
from NPP_PWL_score import NPP_PWL_socre

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

def main():
    #################################################################### Reading Data

    AVN_catalog = pd.read_csv('./data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv')

    AVN_catalog = datetime_to_hours(AVN_catalog)

    AVN_catalog = AVN_catalog.dropna()

    ##### Input variables of script
    # earthquake = 'Visso'
    # Mcut = 1.2
    earthquake = sys.argv[1]
    Mcut = float(sys.argv[2])
    M0 = Mcut
    M0pred = 3
    time_step = 20

    Mmax = 8.0
    VP_params_cut = Mcut

    ##### subsetting catalog by Mcut

    truncated_catalog = truncate_catalog_by_threshold(AVN_catalog,Mcut)
    print('Events above 3.0:  '+str(sum(truncated_catalog['mw']>=3.0)))

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

    ############# input ETAS parameters and C_wd =====
    param_path = f'./ETAS_parameters/params_Mcut_{VP_params_cut}_{earthquake}.csv'
    with open(param_path) as f:
        reader = csv.reader(f)
        MLE = dict(reader)
        del MLE['train_time']
        params = {k: float(v) for k, v in MLE.items()}
    print(f'ETAS parameters:{params}')

    beta = params['beta']
    Delta = Mmax - Mcut
    delta = M0pred - Mcut
    gamma = 1.5 * np.log(10.0)
    alpha = gamma - beta
    Z_norm = 1.0 - np.exp(-beta * Delta)
    C_wd = (
            beta * np.exp(gamma * (Mcut - M0pred)) * (np.exp(alpha * Delta) - np.exp(alpha * delta))
        ) / (
            alpha * (np.exp(-beta * delta) - np.exp(-beta * Delta))
        )
    print('C_wd:  '+str(C_wd))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Training # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    ############# Neural Network
    ckp_dir = './ckp_NPP_PWL_score_VP_seed/'
    os.makedirs(ckp_dir, exist_ok=True)

    checkpoint = f'ckp_NPP_PWL_score_VP_Mcut{Mcut}_{earthquake}_seed{seed}.h5'
    weight_file = os.path.join(ckp_dir, checkpoint)

    # if True:
    if not os.path.isfile(weight_file):
        
        print('Training Network')

        start_time = datetime.datetime.now()

        npp = (NPP_PWL_socre(time_step=time_step,
                  size_rnn=64,
                  size_nn=64,
                  size_layer_chfn=3,
                  size_layer_cmfn=3,
                  M0pred=M0pred,
                  Mcut=Mcut, 
                  C_wd=C_wd,
                  gamma=gamma)
               .set_train_data(T_train,M_train)
               .set_model(0)
               .compile(lr=1e-3)
               .fit_eval(epochs=400,batch_size=256,plot_training=False)
               .save_weights(ckp_dir, checkpoint))
                             

        end_time = datetime.datetime.now()

        npp.train_time = end_time - start_time

    else:
        print('Retreiving Network')
        npp = (NPP_PWL_socre(time_step=time_step,
                  size_rnn=64,
                  size_nn=64,
                  size_layer_chfn=3,
                  size_layer_cmfn=3,
                  M0pred=M0pred,
                  Mcut=Mcut, 
                  C_wd=C_wd,
                  gamma=gamma)
               .set_train_data(T_train,M_train)
               .set_model(0,stateful=False)
               .load_weights(ckp_dir, checkpoint))        
        npp.train_time = 0

    ############ Poisson

    poissMLE = find_Poisson_MLE(T_train,M_train,M0pred)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Testing  # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    result_path = './results_NPP_PWL_score_VP_seed/'
    os.makedirs(result_path, exist_ok=True)
    result_file = f'results_NPP_PWL_score_VP_Mcut{Mcut}_{earthquake}_seed{seed}.csv'
    path_to_results = os.path.join(result_path, result_file)

    pred_path = './pred_NPP_PWL_score_VP_seed/'
    os.makedirs(pred_path, exist_ok=True)
    pred_file = f'pred_NPP_PWL_score_VP_Mcut{Mcut}_{earthquake}_seed{seed}.csv'
    path_to_pred = os.path.join(pred_path, pred_file)


    # if True:
    if not os.path.isfile(path_to_results):
        print('calculating likelihoods')

    ##### Poisson   

        Poisson_time_scores = find_Poisson_likelihood_scores(poissMLE,T_test,M_test,M0pred,time_step)

        LLpoiss = Poisson_time_scores.mean()

        print('Poisson Likelihood:   '+str(LLpoiss))

    ##### NN

        npp.set_test_data(T_test, M_test).predict_eval(batch_size=512)

        print('NPP_PWL_score_VP Temporal Likelihood:   ' + str(npp.collated_LLtime.mean()))
        print('NPP_PWL_score_VP Mark Likelihood:  '+str(npp.collated_LLmag.mean()))

    else: # read in results

        LL_results = pd.read_csv(path_to_results)

        npp.collated_LLtime = LL_results.NPP_PWL_score_time_like
        print('NPP_PWL_score_VP Temporal Likelihood:   ' + str(npp.collated_LLtime.mean()))

        LLpoiss = LL_results.LLpoiss[0]
        print('Poisson Likelihood:   '+str(LLpoiss))

        npp.NPP_mag_like = LL_results.NPP_PWL_score_mag_like
        print('NPP_PWL_score_VP Mark Likelihood:  '+str(npp.NPP_mag_like.mean()))

    ####################################################################### Generate output file 

    if not os.path.isfile(path_to_results):
    # if True:

        npredictions = npp.collated_LLtime.shape[0]
        print('test size ', npredictions)
        lam = np.asarray(npp.lam).reshape(-1) 
        Int_lam = np.asarray(npp.Int_lam).reshape(-1)
        mag_dist = np.asarray(npp.mag_dist).reshape(-1) 
        Int_mag = np.asarray(npp.Int_mag_dist).reshape(-1)

        LL_results ={
            'NPP_PWL_score_time_like':npp.collated_LLtime,
            'NPP_PWL_score_mag_like':npp.collated_LLmag, 
            'LLpoiss': np.repeat(LLpoiss,npredictions),
            'ntrain': np.repeat(T_train.shape[0],npredictions),
            }
    
        pred_results = {
            'NPP_PWL_score_temp_rate':lam,
            'NPP_PWL_score_temp_hazard': Int_lam,
            'NPP_PWL_score_mag_rate':mag_dist,
            'NPP_PWL_score_mag_hazard': Int_mag,
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
    
