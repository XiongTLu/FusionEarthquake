
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
from NeuralPP import NPP

seed = int(sys.argv[3])
# seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

def main():
    #################################################################### Reading Data

    AVN_catalog = pd.read_csv('./data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv')

    AVN_catalog = datetime_to_hours(AVN_catalog)

    AVN_catalog = AVN_catalog.dropna()

    ##### Input variables of script
    # earthquake_for_partition = 'Visso'
    earthquake_for_partition = sys.argv[1]
    # Mcut = 3.0
    Mcut = float(sys.argv[2])
    M0 = Mcut
    M0pred = 3
    time_step = 20

    ##### subsetting catalog by Mcut

    truncated_catalog = truncate_catalog_by_threshold(AVN_catalog,Mcut)
    print('Events above 3.0:  '+str(sum(truncated_catalog['mw']>=3.0)))

    times = np.array(truncated_catalog['time'])
    mags = np.array(truncated_catalog['mw'])

    print('number of datapoints:' + str(times.shape))


    timeupto = select_training_testing_partition(earthquake_for_partition)

    T_train = times[times<timeupto]
    M_train = mags[times<timeupto]
    T_test = times[times>=timeupto]
    M_test = mags[times>=timeupto]
    print('origin n test:' + str(M_test.shape))

    T_test, M_test = append_burn_in_to_test_set(T_train,T_test,M_train,M_test,time_step)

    print('n train:' + str(T_train.shape))
    print('n test:' + str(T_test.shape))



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Training # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    ########### ETAS

    path_to_ETAS_params = f'./ETAS_parameters/params_Mcut_{Mcut}_{earthquake_for_partition}.csv'

    # if True:
    if not os.path.isfile(path_to_ETAS_params):

        print('Findling MLE parameters')

        start_time = datetime.datetime.now()

        MLE = maxlikelihoodETAS(T_train,M_train,M0=M0)

        end_time = datetime.datetime.now()

        MLE['train_time'] = start_time - end_time

        ETAS_train_time = MLE['train_time']

        MLE['beta'] = estimate_beta_value(M_test,M0)

        with open(path_to_ETAS_params, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in MLE.items():
               writer.writerow([key, value])

    else:
        print('openning ETAS param')
        with open(path_to_ETAS_params) as csv_file:
            reader = csv.reader(csv_file)
            MLE = dict(reader)

            ETAS_train_time = MLE['train_time']
            del[MLE['train_time']]

            MLE = {k:float(v) for k, v in MLE.items()}

    print(MLE)  

    ############# Neural Network
    ckp_dir = './ckp_paper_seed/'
    os.makedirs(ckp_dir, exist_ok=True)

    checkpoint = f'checkpoint_Mcut{Mcut}_{earthquake_for_partition}_M0pred{M0pred}_step{time_step}_seed{seed}.h5'
    weight_file = os.path.join(ckp_dir, checkpoint)

    # if True:
    if not os.path.isfile(weight_file):
        
        print('Training Network')

        # start_time = datetime.datetime.now()

        npp = (NPP(time_step=time_step,
                  size_rnn=64,
                  size_nn=64,
                  size_layer_chfn=3,
                  size_layer_cmfn=3,
                  M0pred=M0pred)
               .set_train_data(T_train,M_train)
               .set_model(0)
               .compile(lr=1e-3)
               .fit_eval(epochs=400,batch_size=256,plot_training=False)
               .save_weights(ckp_dir, checkpoint))
                             
        
        # end_time = datetime.datetime.now()

        # npp.train_time = end_time - start_time

    else:
        print('Retreiving Network')
        npp = (NPP(time_step=time_step,
                  size_rnn=64,
                  size_nn=64,
                  size_layer_chfn=3,
                  size_layer_cmfn=3,
                  M0pred=M0pred)
               .set_train_data(T_train,M_train)
               .set_model(0,stateful=False)
               .load_weights(ckp_dir, checkpoint))        
        # npp.train_time = 0

    ############ Poisson

    poissMLE = find_Poisson_MLE(T_train,M_train,M0pred)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # #  Testing  # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    os.makedirs('./results_paper_seed/', exist_ok=True)

    path_to_results = f'./results_paper_seed/resultsMcut-{Mcut}_partition-{earthquake_for_partition}_M0pred_{M0pred}_seed{seed}.csv'

    if True:
    # if not os.path.isfile(path_to_results):

        print('calculating likelihoods')

    ##### Poisson   

        Poisson_time_scores = find_Poisson_likelihood_scores(poissMLE,T_test,M_test,M0pred,time_step)

        LLpoiss = Poisson_time_scores.mean()

        print('Poisson Likelihood:   '+str(LLpoiss))

    # ##### ETAS

    #     time_of_last_target_event = T_test[M_test>=M0pred].max() 

    #     ETAS_time_scores = ETAS.likelihood_scores(T_test,M_test,maxtime=time_of_last_target_event,Mcut=Mcut,M0pred=M0pred,params=MLE,time_step=time_step)

    #     print('ETAS Temporal Likelihood:   ' + str(ETAS_time_scores.mean()))

    #     ETAS_mag_scores = ETAS.magnitude_scores(T_test,M_test,Mcut=Mcut,M0pred=M0pred,params=MLE,time_step=time_step)

    #     print('ETAS Mark Likelihood:  '+str(ETAS_mag_scores.mean()))

    ##### NN

        npp.set_test_data(T_test, M_test).predict_eval(batch_size=512)

        print('NN Temporal Likelihood:   ' + str(npp.collated_LL.mean()))
        print('NN Mark Likelihood:  '+str(npp.LLmag.mean()))

    else: # read in results

        D = pd.read_csv(path_to_results)

        # ETAS_time_scores = D.ETAS_pointwise_like
        # print('ETAS Temporal Likelihood:   ' + str(ETAS_time_scores.mean()))

        npp.collated_LL = D.NN_pointwise_lik
        print('NN Temporal Likelihood:   ' + str(npp.collated_LL.mean()))

        LLpoiss = D.LLpoiss[0]
        print('Poisson Likelihood:   '+str(LLpoiss))

        # ETAS_mag_scores = D.ETASLLmarkvec
        # print('ETAS Mark Likelihood:  '+str(ETAS_mag_scores.mean()))
        npp.LLmag = D.nppLLmag
        print('NN Mark Likelihood:  '+str(npp.LLmag.mean()))

        npp.train_time = D.NN_train_time[0]

    ####################################################################### Generate output file 

    npredictions = npp.collated_LL.shape[0]
    print('test size ', npredictions)
    # print(npp.lam.shape[0])
    # print(npp.mag_dist.shape[0])
    lam = np.asarray(npp.lam).reshape(-1) 
    Int_lam = np.asarray(npp.Int_lam).reshape(-1)
    mag_dist = np.asarray(npp.mag_dist).reshape(-1) 
    Int_mag = np.asarray(npp.Int_mag_dist).reshape(-1)

    print(lam.shape)

    d ={
    # d ={'ETAS_pointwise_like':ETAS_time_scores,
        'NN_pointwise_lik':npp.collated_LL,
        # 'nn_temp_rate':lam,
        # 'nn_temp_hazard': Int_lam,
        # 'ETASLLmarkvec':ETAS_mag_scores,
        'nppLLmag':npp.LLmag, 
        # 'npp_mag_rate':mag_dist,
        # 'nn_mag_hazard': Int_mag,
        'LLpoiss': np.repeat(LLpoiss,npredictions),
        # 'ETAS_mu':np.repeat(MLE['mu'],npredictions),
        # 'ETAS_k0':np.repeat(MLE['k0'],npredictions),
        # 'ETAS_a':np.repeat(MLE['a'],npredictions),
        # 'ETAS_c':np.repeat(MLE['c'],npredictions),
        # 'ETAS_omega':np.repeat(MLE['omega'],npredictions),
        # 'ETAS_train_time': np.repeat(ETAS_train_time,npredictions),
        # 'NN_train_time': np.repeat(npp.train_time,npredictions),
        'ntrain': np.repeat(T_train.shape[0],npredictions),
        'time_step' : np.repeat(time_step,npredictions)
        }

    # if not os.path.isfile(path_to_results):
    if True:

        print('writing data')

        D = pd.DataFrame(d)

        D.to_csv(path_to_results)
        print(f'✅ successful save rate result as: {path_to_results}')


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




if __name__ == '__main__':
    main()
    
