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

from ETAS_PWL import estimate_beta_value, maxlikelihoodETAS_PWL
from utils import truncate_catalog_by_threshold, datetime_to_hours, append_burn_in_to_test_set, find_Poisson_MLE, collate_likelihoods, find_Poisson_likelihood_scores

seed = 1
np.random.seed(seed)

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
    earthquake_for_partition = 'Visso'
    # earthquake_for_partition = sys.argv[1]
    Mcut =    1.2 
    # Mcut = float(sys.argv[2])
    M0 = Mcut
    M0pred = 3
    time_step = 20
    Mmax = 8.0

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

    path_params = f'./ETAS_PWL_parameters'
    os.makedirs(path_params, exist_ok=True)
    file_name_params = f'ETAS_PWL_params_Mcut_{Mcut}_{earthquake_for_partition}.csv'

    path_to_ETAS_params = os.path.join(path_params, file_name_params)

    # if True:
    if not os.path.isfile(path_to_ETAS_params):

        print('Findling ETAS-PWL MLE parameters')

        start_time = datetime.datetime.now()

        MLE = maxlikelihoodETAS_PWL(T_train,M_train,M0=M0,Mcut=Mcut,Mmax=Mmax,M0pred=M0pred)

        end_time = datetime.datetime.now()

        MLE['train_time'] = end_time - start_time

        ETAS_train_time = MLE['train_time']

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


if __name__ == '__main__':
    main()
