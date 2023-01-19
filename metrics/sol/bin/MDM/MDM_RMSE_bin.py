import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

# from keras.models import load_model

import matplotlib.pyplot as plt
#import evidential_deep_learning as edl

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

import seaborn as sns

import math

from scipy.stats.stats import pearsonr
import random
from numpy.random import choice
import time
from scipy.stats import spearmanr

from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

import keras

import matplotlib.pyplot as plt
import numpy as np
from keras.regularizers import l2
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import RandomizedSearchCV

from hyperopt import Trials, STATUS_OK, tpe
# from hyperas import optim
# from hyperas.distributions import choice, uniform

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint

from keras.models import load_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
%matplotlib inline

from scipy.stats import spearmanr
from keras.initializers import random_normal, random_uniform
# from keras.initializers import normal

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from rdkit import Chem

import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import pickle
import json

import seaborn as sns

import math

def get_transformed_data(train, val, test, to_drop, y):
    x_train = train.drop(to_drop, axis=1)
    x_val = val.drop(to_drop, axis=1)
    x_test = test.drop(to_drop, axis=1)

    y_train = train[y].values
    y_val = val[y].values
    y_test = test[y].values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test, x_val, y_val, scaler


wdir = # MDM data path

to_remove = ['cas', 'ref', 'temp', 'log_kow', 'inchi', 'melting_point', 'sol_energy',
             'dip_mmnt', 'dip_mmnt_vol', 'quad_mmnt', 'mass']

train = pd.read_csv(wdir + "train.csv")
val = pd.read_csv(wdir + "val.csv")
test = pd.read_csv(wdir + "test.csv")

print(train.shape)
train = train.drop(to_remove, axis=1)
val = val.drop(to_remove, axis=1)
test = test.drop(to_remove, axis=1)
print(train.shape)

trainx = train
valx = val
testx = test

to_drop = ['log_sol', 'smiles']

x_train, y_train, x_test, y_test, x_val, y_val, sc = get_transformed_data(train=trainx,
                                                                          val=valx,
                                                                          test=testx,
                                                                          to_drop=to_drop,
                                                                          y="log_sol")


def get_bin_index(pc_train, pc_val, pc_test, num_bins):
    cut_val = pd.qcut(pc_train, q=num_bins, retbins=True)[1]  # equalObs(x_train[:, col_num], num_bins)

    val_in_bin_idx_ls = []
    val_out_bin_idx_ls = []
    test_in_bin_idx_ls = []
    train_in_bin_idx_ls = []
    train_out_bin_idx_ls = []

    for i in range(len(cut_val) - 1):
        val_in_bin_idx_temp = [i for i, x in enumerate((pc_val >= cut_val[i]) & (pc_val < cut_val[i + 1])) if x]
        val_out_bin_idx_temp = list(np.delete(range(len(pc_val)), val_in_bin_idx_temp, axis=0))
        test_in_bin_idx_temp = [i for i, x in enumerate((pc_test >= cut_val[i]) & (pc_test < cut_val[i + 1])) if x]
        train_in_bin_idx_temp = [i for i, x in enumerate((pc_train >= cut_val[i]) & (pc_train < cut_val[i + 1])) if x]
        train_out_bin_idx_temp = list(np.delete(range(len(pc_train)), train_in_bin_idx_temp, axis=0))

        val_in_bin_idx_ls.append(val_in_bin_idx_temp)
        val_out_bin_idx_ls.append(val_out_bin_idx_temp)
        test_in_bin_idx_ls.append(test_in_bin_idx_temp)
        train_in_bin_idx_ls.append(train_in_bin_idx_temp)
        train_out_bin_idx_ls.append(train_out_bin_idx_temp)

    return val_in_bin_idx_ls, val_out_bin_idx_ls, test_in_bin_idx_ls, train_in_bin_idx_ls, train_out_bin_idx_ls



# tpe hopt
act = {0: 'relu', 1: 'selu', 2: 'sigmoid'}
# args = {'a1': 2, 'a2': 1, 'a3': 1, 'a4': 2, 'a5': 0, 'bs': 2, 'd1': 0.10567353589351362, 'd2': 0.07099446638977362,\
#         'd3': 0.7078956756855795, 'd4': 0.6621719558843959, 'd5': 0.21112385307385717, 'h1': 576.0,\
#         'h2': 320.0, 'h3': 128.0, 'h4': 256.0, 'h5': 128.0, 'lr': 0, 'nfc': 0, 'opt': 1}


args = {'a1': 2, 'a2': 0, 'a3': 1, 'a4': 1, 'a5': 0, 'bs': 1, 'd1': 0.10696194799818459, 'd2': 0.6033824611348487, \
        'd3': 0.7388531806558837, 'd4': 0.9943053700072028, 'd5': 0.016358259737496605, 'h1': 128.0, 'h2': 576.0, \
        'h3': 448.0, 'h4': 256.0, 'h5': 128.0, 'lr': 0, 'nfc': 0, 'opt': 1}


# args={'a1': 2, 'a2': 0, 'a3': 1, 'a4': 1, 'a5': 0, 'bs': 1, 'd1': 0.5, 'd2': 0.6,\
#      'd3': 0.7388531806558837, 'd4': 0.9943053700072028, 'd5': 0.016358259737496605, 'h1': 128.0, 'h2': 576.0,\
#      'h3': 448.0, 'h4': 256.0, 'h5': 128.0, 'lr': 0, 'nfc': 0, 'opt': 1}

def create_model(x_train):
    model = Sequential()
    model.add(Dense(int(args['h1']), input_shape=(x_train.shape[1],)))
    model.add(Activation(act[args['a1']]))
    model.add(Dropout(args['d1']))
    model.add(Dense(int(args['h2'])))
    model.add(Activation(act[args['a2']]))
    model.add(Dropout(args['d2']))

    # if {{choice(['two','three'])}} =='three':
    #     if args['nfc'] == 3:

    #     model.add(Dense( int(args['h3']) ))
    #     model.add(Activation( act[args['a3']] ))
    #     model.add(Dropout( args['d3'] ))

    #     if args['nfc'] == 4:
    #         model.add(Dense( int(args['h3']) ))
    #         model.add(Activation( args['a3'] ))
    #         model.add(Dropout( args['d3'] ))

    #         model.add(Dense( int(args['h4']) ))
    #         model.add(Activation( args['a4'] ))
    #         model.add(Dropout( args['d4'] ))

    #     if args['nfc'] == 5:
    #         model.add(Dense( int(args['h3']) ))
    #         model.add(Activation( args['a3'] ))
    #         model.add(Dropout( args['d3'] ))

    #         model.add(Dense( int(args['h4']) ))
    #         model.add(Activation( args['a4'] ))
    #         model.add(Dropout( args['d4'] ))

    #         model.add( Dense(int(args['h5']) ))
    #         model.add(Activation( args['a5'] ))
    #         model.add(Dropout( args['d5'] ))

    model.add(Dense(1, activation='linear'))

    rmsprop = keras.optimizers.RMSprop(lr=10 ** -3)
    opt = rmsprop

    model.compile(loss='mean_squared_error', optimizer=opt)

    return model

def get_RMSE_bin_revomal_results(x_train, y_train,
                                 x_val, y_val,
                                 x_test, y_test,
                                 val_in_bin_idx_ls, train_in_bin_idx_ls, test_in_bin_idx_ls, n_exp):
    test_rmse_in_bin_ls = []
    test_rmse_remain_ls = []
    test_rmse_in_bin_nr_ls = []
    test_rmse_remain_nr_ls = []
    for j in range(n_exp):
        test_rmse_in_bin = []
        test_rmse_remain = []
        for i in range(len(train_in_bin_idx_ls)):
            train_ind = train_in_bin_idx_ls[i]
            val_ind = val_in_bin_idx_ls[i]
            test_ind = test_in_bin_idx_ls[i]

            x_train_now = np.delete(x_train, train_ind, axis=0)
            y_train_now = np.delete(y_train, train_ind, axis=0)
            x_val_now = np.delete(x_val, val_ind, axis=0)
            y_val_now = np.delete(y_val, val_ind, axis=0)

            model = create_model(x_train_now)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)
            result = model.fit(x_train_now, y_train_now, batch_size = 64, epochs = 1000,
                               verbose = 0, validation_data = (x_val_now,y_val_now), callbacks = [es])

            y_test_pred = model(x_test)
            test_rmse_in_bin.append(mean_squared_error(y_test[test_ind], np.array(y_test_pred[:, 0])[test_ind], squared=False))
            test_rmse_remain.append(mean_squared_error(np.delete(y_test, test_ind, axis=0), np.delete(np.array(y_test_pred[:, 0]), test_ind, axis=0), squared=False))

        model = create_model(x_train)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)
        result = model.fit(x_train, y_train, batch_size = 64, epochs = 1000,
                           verbose = 0, validation_data = (x_val,y_val), callbacks = [es])
        y_test_pred = model(x_test)
        test_rmse_in_bin_nr = []
        test_rmse_remain_nr = []
        for i in range(len(test_in_bin_idx_ls)):
            test_ind = test_in_bin_idx_ls[i]
            test_rmse_in_bin_nr.append(mean_squared_error(y_test[test_ind], np.array(y_test_pred[:, 0])[test_ind], squared=False))
            test_rmse_remain_nr.append(mean_squared_error(np.delete(y_test, test_ind, axis=0), np.delete(np.array(y_test_pred[:, 0]), test_ind, axis=0), squared=False))
        test_rmse_in_bin_ls.append(test_rmse_in_bin)
        test_rmse_remain_ls.append(test_rmse_remain)
        test_rmse_in_bin_nr_ls.append(test_rmse_in_bin_nr)
        test_rmse_remain_nr_ls.append(test_rmse_remain_nr)
    test_rmse_in_bin_df = pd.DataFrame(test_rmse_in_bin_ls)
    test_rmse_remain_df = pd.DataFrame(test_rmse_remain_ls)
    test_rmse_in_bin_nr_df = pd.DataFrame(test_rmse_in_bin_nr_ls)
    test_rmse_remain_nr_df = pd.DataFrame(test_rmse_remain_nr_ls)
    return test_rmse_in_bin_df, test_rmse_remain_df, test_rmse_in_bin_nr_df, test_rmse_remain_nr_df


def get_RMSE_result(x_train, y_train, x_val, y_val, x_test, y_test, num_pc, num_bins, pca_ncomp, n_exp):
    pca = PCA(n_components=pca_ncomp)
    pca.fit(x_train)
    pca_space_train = pca.transform(x_train)
    pca_space_val = pca.transform(x_val)
    pca_space_test = pca.transform(x_test)
    err_perc_pc = []

    test_rmse_in_bin_df_ls = []
    test_rmse_remain_df_ls = []
    test_rmse_in_bin_nr_df_ls = []
    test_rmse_remain_nr_df_ls = []

    for k in range(num_pc):
        val_in_bin_idx_ls, val_out_bin_idx_ls, test_in_bin_idx_ls, train_in_bin_idx_ls, train_out_bin_idx_ls = \
            get_bin_index(pca_space_train[:, k], pca_space_val[:, k], pca_space_test[:, k], num_bins)
        test_rmse_in_bin_df, test_rmse_remain_df, test_rmse_in_bin_nr_df, test_rmse_remain_nr_df = get_RMSE_bin_revomal_results(
            x_train, y_train,
            x_val, y_val,
            x_test, y_test,
            val_in_bin_idx_ls, train_in_bin_idx_ls, test_in_bin_idx_ls, n_exp)
        err_perc_pc.append((test_rmse_in_bin_df.mean() - test_rmse_in_bin_nr_df.mean()) / test_rmse_in_bin_nr_df.mean())
        test_rmse_in_bin_df_ls.append(test_rmse_in_bin_df)
        test_rmse_remain_df_ls.append(test_rmse_remain_df)
        test_rmse_in_bin_nr_df_ls.append(test_rmse_in_bin_nr_df)
        test_rmse_remain_nr_df_ls.append(test_rmse_remain_nr_df)

    return np.array(err_perc_pc).reshape(1, num_pc * num_bins)[
               0], test_rmse_in_bin_df_ls, test_rmse_remain_df_ls, test_rmse_in_bin_nr_df_ls, test_rmse_remain_nr_df_ls

err_perc_pc, test_rmse_in_bin_df_ls, test_rmse_remain_df_ls, test_rmse_in_bin_nr_df_ls, test_rmse_remain_nr_df_ls = get_RMSE_result(x_train, y_train, x_val, y_val, x_test, y_test, 3, 5, 20, 3)

np.savez('sol_mdm_error.npz', err_perc_pc, test_rmse_in_bin_df_ls,
         test_rmse_remain_df_ls,
         test_rmse_in_bin_nr_df_ls,
         test_rmse_remain_nr_df_ls)






