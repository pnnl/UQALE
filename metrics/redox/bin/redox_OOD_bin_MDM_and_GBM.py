
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from keras.models import load_model

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
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

# redox MDM data path
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
val = pd.read_csv('val.csv')


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


to_remove = ['SMILES', 'Standard InChIKey', 'inchi', 'smiles']

print(train.shape)
train = train.drop(to_remove, axis=1)
val = val.drop(to_remove, axis=1)
test = test.drop(to_remove, axis=1)
print(train.shape)

trainx = train
valx = val
testx = test

to_drop = ['Vox(V)', 'Vred(V)']

x_train, y_train, x_test, y_test, x_val, y_val, sc = get_transformed_data(train=trainx,
                                                                          val=valx,
                                                                          test=testx,
                                                                          to_drop=to_drop,
                                                                          y="Vox(V)")


pca = PCA(n_components=20)
pca.fit(x_train)
pca_space_train = pca.transform(x_train)
pca_space_val = pca.transform(x_val)
pca_space_test = pca.transform(x_test)

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


def get_unc_bin_revomal_results(x_train, y_train, x_val, y_val, x_test, y_test,
                                val_in_bin_idx, test_in_bin_idx, train_in_bin_idx, n_exp, unc_func):
    unc_in_bin_ls = []
    unc_remain_ls = []
    unc_in_bin_nr_ls = []
    unc_remain_nr_ls = []
    for j in range(n_exp):
        unc_in_bin = []
        unc_remain = []
        for i in range(len(train_in_bin_idx)):
            train_ind = train_in_bin_idx[i]
            val_ind = val_in_bin_idx[i]
            test_ind = test_in_bin_idx[i]

            x_train_now = np.delete(x_train, train_ind, axis=0)
            y_train_now = np.delete(y_train, train_ind, axis=0)
            x_val_now = np.delete(x_val, val_ind, axis=0)
            y_val_now = np.delete(y_val, val_ind, axis=0)

            total_uncertainty = unc_func(x_train_now, y_train_now, x_val_now, y_val_now, x_test, y_test)

            unc_in_bin.append(np.mean(np.array(total_uncertainty)[test_ind]))
            unc_remain.append(np.mean(np.delete(np.array(total_uncertainty), test_ind, axis=0)))

        total_uncertainty = unc_func(x_train, y_train, x_val, y_val, x_test, y_test)

        unc_in_bin_nr = []
        unc_remain_nr = []
        for i in range(len(test_in_bin_idx)):
            test_ind = test_in_bin_idx[i]
            unc_in_bin_nr.append(np.mean(np.array(total_uncertainty)[test_ind]))
            unc_remain_nr.append(np.mean(np.delete(np.array(total_uncertainty), test_ind, axis=0)))

        unc_in_bin_ls.append(unc_in_bin)
        unc_remain_ls.append(unc_remain)
        unc_in_bin_nr_ls.append(unc_in_bin_nr)
        unc_remain_nr_ls.append(unc_remain_nr)

    unc_in_bin_df = pd.DataFrame(unc_in_bin_ls)
    unc_remain_df = pd.DataFrame(unc_remain_ls)
    unc_in_bin_nr_df = pd.DataFrame(unc_in_bin_nr_ls)
    unc_remain_nr_df = pd.DataFrame(unc_remain_nr_ls)

    return unc_in_bin_df, unc_remain_df, unc_in_bin_nr_df, unc_remain_nr_df


def get_unc_result(x_train, y_train, x_val, y_val, x_test, y_test,
                   pca_space_train, pca_space_val, pca_space_test,
                   num_pc, num_bins, n_exp, unc_func):
    unc_perc = []
    test_unc_in_bin_df_ls = []
    test_unc_remain_df_ls = []
    test_unc_in_bin_nr_df_ls = []
    test_unc_remain_nr_df_ls = []
    for k in range(num_pc):
        val_in_bin_idx_ls, val_out_bin_idx_ls, test_in_bin_idx_ls, train_in_bin_idx_ls, train_out_bin_idx_ls = \
            get_bin_index(pca_space_train[:, k], pca_space_val[:, k], pca_space_test[:, k], num_bins)
        test_unc_in_bin_df, test_unc_remain_df, test_unc_in_bin_nr_df, test_unc_remain_nr_df = get_unc_bin_revomal_results(
            x_train, y_train, x_val, y_val, x_test, y_test,
            val_in_bin_idx_ls, test_in_bin_idx_ls, train_in_bin_idx_ls,
            n_exp, unc_func)
        unc_perc.append((test_unc_in_bin_df.mean() - test_unc_in_bin_nr_df.mean()) / test_unc_in_bin_nr_df.mean())
        test_unc_in_bin_df_ls.append(test_unc_in_bin_df)
        test_unc_remain_df_ls.append(test_unc_remain_df)
        test_unc_in_bin_nr_df_ls.append(test_unc_in_bin_nr_df)
        test_unc_remain_nr_df_ls.append(test_unc_remain_nr_df)
    return np.array(unc_perc).reshape(1, num_bins * num_pc)[
               0], test_unc_in_bin_df_ls, test_unc_remain_df_ls, test_unc_in_bin_nr_df_ls, test_unc_remain_nr_df_ls


# MDM error

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

# tpe hopt
act = {0: 'relu', 1: 'selu', 2: 'sigmoid'}
# args = {'a1': 2, 'a2': 1, 'a3': 1, 'a4': 2, 'a5': 0, 'bs': 2, 'd1': 0.10567353589351362, 'd2': 0.07099446638977362,\
#         'd3': 0.7078956756855795, 'd4': 0.6621719558843959, 'd5': 0.21112385307385717, 'h1': 576.0,\
#         'h2': 320.0, 'h3': 128.0, 'h4': 256.0, 'h5': 128.0, 'lr': 0, 'nfc': 0, 'opt': 1}


# args={'a1': 2, 'a2': 0, 'a3': 1, 'a4': 1, 'a5': 0, 'bs': 1, 'd1': 0.10696194799818459, 'd2': 0.6033824611348487,\
#       'd3': 0.7388531806558837, 'd4': 0.9943053700072028, 'd5': 0.016358259737496605, 'h1': 128.0, 'h2': 576.0,\
#       'h3': 448.0, 'h4': 256.0, 'h5': 128.0, 'lr': 0, 'nfc': 0, 'opt': 1}

# args={'a1': 2, 'a2': 0, 'a3': 1, 'a4': 1, 'a5': 0, 'bs': 1, 'd1': 0.5, 'd2': 0.6,\
#      'd3': 0.7388531806558837, 'd4': 0.9943053700072028, 'd5': 0.016358259737496605, 'h1': 128.0, 'h2': 576.0,\
#      'h3': 448.0, 'h4': 256.0, 'h5': 128.0, 'lr': 0, 'nfc': 0, 'opt': 1}


args = {'a1': 2, 'a2': 2, 'a3': 0, 'a4': 1, 'a5': 2, 'bs': 2, 'd1': 0.14172091923172192, 'd2': 0.04405799333743232, \
        'd3': 0.011693878279452341, 'd4': 0.4804983581922393, 'd5': 0.43108843234185323, 'h1': 256.0, 'h2': 128.0, \
        'h3': 384.0, 'h4': 128.0, 'h5': 576.0, 'lr': 0, 'nfc': 0, 'opt': 0}


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

err_perc_pc, test_rmse_in_bin_df_ls, test_rmse_remain_df_ls, test_rmse_in_bin_nr_df_ls, test_rmse_remain_nr_df_ls = get_RMSE_result(x_train, y_train, x_val, y_val, x_test, y_test, 3, 5, 20, 2)

np.savez('redox_mdm_error.npz', err_perc_pc, test_rmse_in_bin_df_ls,
         test_rmse_remain_df_ls,
         test_rmse_in_bin_nr_df_ls,
         test_rmse_remain_nr_df_ls)


# GBM error

from sklearn.ensemble import GradientBoostingRegressor


def get_RMSE_bin_revomal_results_gbm(x_train, y_train,
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

            mid_model = GradientBoostingRegressor(loss="ls")
            mid_model.fit(x_train_now, y_train_now)

            y_test_pred = np.array(mid_model.predict(x_test))
            test_rmse_in_bin.append(mean_squared_error(y_test[test_ind], y_test_pred[test_ind], squared=False))
            test_rmse_remain.append(
                mean_squared_error(np.delete(y_test, test_ind, axis=0), np.delete(y_test_pred, test_ind, axis=0),
                                   squared=False))

        mid_model = GradientBoostingRegressor(loss="ls")
        mid_model.fit(x_train, y_train)

        y_test_pred = np.array(mid_model.predict(x_test))
        test_rmse_in_bin_nr = []
        test_rmse_remain_nr = []
        for i in range(len(test_in_bin_idx_ls)):
            test_ind = test_in_bin_idx_ls[i]
            test_rmse_in_bin_nr.append(mean_squared_error(y_test[test_ind], y_test_pred[test_ind], squared=False))
            test_rmse_remain_nr.append(
                mean_squared_error(np.delete(y_test, test_ind, axis=0), np.delete(y_test_pred, test_ind, axis=0),
                                   squared=False))
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
        test_rmse_in_bin_df, test_rmse_remain_df, test_rmse_in_bin_nr_df, test_rmse_remain_nr_df = get_RMSE_bin_revomal_results_gbm(
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


err_perc_pc_gbm, test_rmse_in_bin_df_ls_gbm, test_rmse_remain_df_ls_gbm, test_rmse_in_bin_nr_df_ls_gbm, test_rmse_remain_nr_df_ls_gbm = get_RMSE_result(x_train, y_train, x_val, y_val, x_test, y_test, 3, 5, 20, 1)

np.savez('redox_gbm_error.npz', err_perc_pc_gbm, test_rmse_in_bin_df_ls_gbm,
         test_rmse_remain_df_ls_gbm,
         test_rmse_in_bin_nr_df_ls_gbm,
         test_rmse_remain_nr_df_ls_gbm)

# GBM

def get_uncertainty_gbm(x_train_now, y_train_now, x_val_now, y_val_now, x_test, y_test):
    lower_model = GradientBoostingRegressor(loss="quantile",
                                            alpha=LOWER_ALPHA)
    # The mid model will use the default loss
    mid_model = GradientBoostingRegressor(loss="ls")
    upper_model = GradientBoostingRegressor(loss="quantile",
                                            alpha=UPPER_ALPHA)

    lower_model.fit(x_train_now, y_train_now)
    mid_model.fit(x_train_now, y_train_now)
    upper_model.fit(x_train_now, y_train_now)

    predictions = pd.DataFrame(y_test)

    predictions['lower'] = lower_model.predict(x_test)
    predictions['mid'] = mid_model.predict(x_test)
    predictions['upper'] = upper_model.predict(x_test)

    GB_unc = np.array(np.sqrt(abs(predictions['upper'] - predictions['lower']) / 2))

    return GB_unc

unc_perc_gbm, test_unc_in_bin_df_ls_gbm, test_unc_remain_df_ls_gbm, test_unc_in_bin_nr_df_ls_gbm, test_unc_remain_nr_df_ls_gbm = get_unc_result(x_train, y_train, x_val, y_val, x_test, y_test,
                              pca_space_train, pca_space_val, pca_space_test,
                              3, 5, 1, get_uncertainty_gbm)

np.savez('redox_gbm_gbm.npz', unc_perc_gbm, test_unc_in_bin_df_ls_gbm,
         test_unc_remain_df_ls_gbm,
         test_unc_in_bin_nr_df_ls_gbm,
         test_unc_remain_nr_df_ls_gbm)



# MDM + GBM

from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Model

LOWER_ALPHA = 0.1
UPPER_ALPHA = 0.9


def get_uncertainty_union(x_train_now, y_train_now, x_val_now, y_val_now, x_test, y_test):
    model = create_model(x_train_now)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)
    result = model.fit(x_train_now, y_train_now, batch_size=64, epochs=1000,
                       verbose=0, validation_data=(x_val_now, y_val_now), callbacks=[es])

    model2 = Model(model.input, model.layers[-2].output)
    new_training_x = model2(x_train_now)
    new_test_x = model2(x_test)

    lower_model = GradientBoostingRegressor(loss="quantile",
                                            alpha=LOWER_ALPHA)
    mid_model = GradientBoostingRegressor(loss="ls")
    upper_model = GradientBoostingRegressor(loss="quantile",
                                            alpha=UPPER_ALPHA)

    lower_model.fit(new_training_x, y_train_now)
    mid_model.fit(new_training_x, y_train_now)
    upper_model.fit(new_training_x, y_train_now)

    predictions = pd.DataFrame(y_test)

    predictions['lower'] = lower_model.predict(new_test_x)
    predictions['mid'] = mid_model.predict(new_test_x)
    predictions['upper'] = upper_model.predict(new_test_x)

    GB_unc = np.array(np.sqrt(abs(predictions['upper'] - predictions['lower']) / 2))

    return GB_unc

unc_perc_union, test_unc_in_bin_df_ls, test_unc_remain_df_ls, test_unc_in_bin_nr_df_ls, test_unc_remain_nr_df_ls = get_unc_result(x_train, y_train, x_val, y_val, x_test, y_test,
                              pca_space_train, pca_space_val, pca_space_test,
                              3, 5, 1, get_uncertainty_union)

np.savez('redox_mdm_gbm.npz', unc_perc_union, test_unc_in_bin_df_ls,
         test_unc_remain_df_ls,
         test_unc_in_bin_nr_df_ls,
         test_unc_remain_nr_df_ls)

# MDM + EDL

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*tf.math.log(np.pi/v)  \
        - alpha*tf.math.log(twoBlambda)  \
        + (alpha+0.5) * tf.math.log(v*(y-gamma)**2 + twoBlambda)  \
        + tf.math.lgamma(alpha)  \
        - tf.math.lgamma(alpha+0.5)

    return tf.reduce_mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*tf.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*tf.math.log(tf.abs(v2)/tf.abs(v1))  \
        - 0.5 + a2*tf.math.log(b1/b2)  \
        - (tf.math.lgamma(a1) - tf.math.lgamma(a2))  \
        + (a1 - a2)*tf.math.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = tf.abs(y-gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi

    return tf.reduce_mean(reg) if reduce else reg

def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg

from tensorflow.keras.layers import Layer


class DenseNormalGamma(Layer):
    def __init__(self, units):
        super(DenseNormalGamma, self).__init__()
        self.units = int(units)
        self.dense = Dense(4 * self.units, activation=None)

    def evidence(self, x):
        # return tf.exp(x)
        return tf.nn.softplus(x)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4 * self.units)

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config['units'] = self.units
        return base_config

# tpe hopt
act = {0:'relu', 1:'selu', 2:'sigmoid'}

args={'a1': 2, 'a2': 2, 'a3': 0, 'a4': 1, 'a5': 2, 'bs': 2, 'd1': 0.14172091923172192, 'd2': 0.04405799333743232, \
      'd3': 0.011693878279452341, 'd4': 0.4804983581922393, 'd5': 0.43108843234185323, 'h1': 256.0, 'h2': 128.0, \
      'h3': 384.0, 'h4': 128.0, 'h5': 576.0, 'lr': 0, 'nfc': 0, 'opt': 0}


# Define our model with an evidential output
model = tf.keras.Sequential([
    Dense(int(args['h1']), input_shape = (x_train.shape[1],)),
    Activation(act[args['a1']] ),
    Dropout(args['d1'] ),
    Dense(int(args['h2'])  ),
    Activation(act[args['a2']] ),
    Dropout(args['d2'] ),
    DenseNormalGamma(1),
])

# Custom loss function to handle the custom regularizer coefficient
def EvidentialRegressionLoss(true, pred):
    return EvidentialRegression(true, pred, coeff=0.003)


def get_uncertainty_EDL(x_train_now, y_train_now, x_val_now, y_val_now, x_test, y_test):
    model = tf.keras.Sequential([
        Dense(int(args['h1']), input_shape=(x_train.shape[1],)),
        Activation(act[args['a1']]),
        Dropout(args['d1']),
        Dense(int(args['h2'])),
        Activation(act[args['a2']]),
        Dropout(args['d2']),
        DenseNormalGamma(1),
    ])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    # Compile and fit the model!
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=EvidentialRegressionLoss)
    model.fit(x_train_now, y_train_now, batch_size=64, epochs=1000, validation_data=(x_val_now, y_val_now),
              verbose=0, callbacks=[callback])

    y_test_pred = model(x_test)
    data_uncertainty = y_test_pred[:, 3] / (y_test_pred[:, 2] - 1)
    model_uncertainty = y_test_pred[:, 3] / (y_test_pred[:, 1] * (y_test_pred[:, 2] - 1))
    total_uncertainty = model_uncertainty + data_uncertainty

    return np.array(total_uncertainty)

unc_perc_EDL, test_unc_in_bin_df_ls_EDL, test_unc_remain_df_ls_EDL, test_unc_in_bin_nr_df_ls_EDL, test_unc_remain_nr_df_ls_EDL  = get_unc_result(x_train, y_train, x_val, y_val, x_test, y_test,
                              pca_space_train, pca_space_val, pca_space_test,
                              3, 5, 1, get_uncertainty_EDL)

np.savez('redox_mdm_edl.npz', unc_perc_EDL, test_unc_in_bin_df_ls_EDL,
         test_unc_remain_df_ls_EDL,
         test_unc_in_bin_nr_df_ls_EDL,
         test_unc_remain_nr_df_ls_EDL)

# MDM + MCDO

def predict_dist(X, model, num_samples):
    preds = [model(X, training=True) for _ in range(num_samples)]
    return np.hstack(preds)

def predict_point(X, model, num_samples):
    pred_dist = predict_dist(X, model, num_samples)
    return pred_dist.mean(axis=1)


def get_uncertainty_MCD(x_train_now, y_train_now, x_val_now, y_val_now, x_test, y_test):
    model = create_model(x_train_now)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)
    result = model.fit(x_train_now, y_train_now, batch_size=64, epochs=1000,
                       verbose=0, validation_data=(x_val_now, y_val_now), callbacks=[es])

    y_pred_dist = predict_dist(x_test, model, 100)
    y_pred_dist_std = []
    for i in range(len(y_pred_dist)):
        y_pred_dist_std.append(y_pred_dist[i].std())

    return np.array(y_pred_dist_std)

unc_perc_MCD, test_unc_in_bin_df_ls, test_unc_remain_df_ls, test_unc_in_bin_nr_df_ls, test_unc_remain_nr_df_ls = \
get_unc_result(x_train, y_train, x_val, y_val, x_test, y_test,
               pca_space_train, pca_space_val, pca_space_test,
               3, 5, 1, get_uncertainty_MCD)

np.savez('redox_mdm_mcd.npz', unc_perc_MCD, test_unc_in_bin_df_ls,
         test_unc_remain_df_ls,
         test_unc_in_bin_nr_df_ls,
         test_unc_remain_nr_df_ls)

# MDM + ensemble

def predict_dist_ensemble(X, model_ensemble):
    preds = [model_ensemble[i].predict(X) for i in range(10)]
    return np.hstack(preds)

def predict_point_ensemble(X, model_ensemble):
    pred_dist = predict_dist_ensemble(X, model_ensemble)
    return pred_dist.mean(axis=1)


def get_uncertainty_Ens(x_train_now, y_train_now, x_val_now, y_val_now, x_test, y_test):
    model_ensemble = []
    for _ in range(10):
        model = create_model(x_train_now)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)
        result = model.fit(x_train_now, y_train_now, batch_size=64, epochs=1000,
                           verbose=0, validation_data=(x_val_now, y_val_now), callbacks=[es])
        model_ensemble.append(model)

    y_pred_dist_ensemble = predict_dist_ensemble(x_test, model_ensemble)
    y_pred_dist_std_ensemble = []
    for i in range(len(y_pred_dist_ensemble)):
        y_pred_dist_std_ensemble.append(y_pred_dist_ensemble[i].std())

    return y_pred_dist_std_ensemble


unc_perc_Ens, test_unc_in_bin_df_ls, test_unc_remain_df_ls, test_unc_in_bin_nr_df_ls, test_unc_remain_nr_df_ls = \
get_unc_result(x_train, y_train, x_val, y_val, x_test, y_test,
               pca_space_train, pca_space_val, pca_space_test, 3, 5, 1, get_uncertainty_Ens)

np.savez('redox_mdm_ens.npz', unc_perc_Ens, test_unc_in_bin_df_ls,
         test_unc_remain_df_ls,
         test_unc_in_bin_nr_df_ls,
         test_unc_remain_nr_df_ls)

# MDM + MVE

def Gaussian_NLL_MVE(y, mu, sigma):
    loss = tf.math.log(sigma) + 0.5*tf.math.log(2*np.pi) + 0.5*((y-mu)/sigma)**2
    return loss

def MVE(y_true, MVE_output):
    mu, sigma = tf.split(MVE_output, 2, axis=-1)
    loss_nll = Gaussian_NLL_MVE(y_true, mu, sigma)
    return loss_nll

from tensorflow.keras.layers import Layer


class DenseNormal(Layer):
    def __init__(self, units):
        super(DenseNormal, self).__init__()
        self.units = int(units)
        self.dense = Dense(2 * self.units)

    def call(self, x):
        output = self.dense(x)
        mu, logsigma = tf.split(output, 2, axis=-1)
        sigma = tf.nn.softplus(logsigma) + 1e-6
        return tf.concat([mu, sigma], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.units)

    def get_config(self):
        base_config = super(DenseNormal, self).get_config()
        base_config['units'] = self.units
        return base_config

# tpe hopt
act = {0:'relu', 1:'selu', 2:'sigmoid'}

args={'a1': 2, 'a2': 2, 'a3': 0, 'a4': 1, 'a5': 2, 'bs': 2, 'd1': 0.14172091923172192, 'd2': 0.04405799333743232, \
      'd3': 0.011693878279452341, 'd4': 0.4804983581922393, 'd5': 0.43108843234185323, 'h1': 256.0, 'h2': 128.0, \
      'h3': 384.0, 'h4': 128.0, 'h5': 576.0, 'lr': 0, 'nfc': 0, 'opt': 0}


# Custom loss function to handle the custom regularizer coefficient
def MVELoss(true, pred):
    return MVE(true, pred)


def get_uncertainty_MVE(x_train_now, y_train_now, x_val_now, y_val_now, x_test, y_test):
    model = tf.keras.Sequential([
        Dense(int(args['h1']), input_shape=(x_train.shape[1],)),
        Activation(act[args['a1']]),
        Dropout(args['d1']),
        Dense(int(args['h2'])),
        Activation(act[args['a2']]),
        Dropout(args['d2']),
        DenseNormal(1),
    ])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=MVELoss)
    model.fit(x_train_now, y_train_now, batch_size=64, epochs=1000, verbose=0,
              validation_data=(x_val_now, y_val_now), callbacks=[callback])

    y_test_pred = model(x_test)
    MVE_unc = np.array(y_test_pred[:, 1])

    return MVE_unc

unc_perc_MVE, test_unc_in_bin_df_ls, test_unc_remain_df_ls, test_unc_in_bin_nr_df_ls, test_unc_remain_nr_df_ls = \
get_unc_result(x_train, y_train, x_val, y_val, x_test, y_test,
               pca_space_train, pca_space_val, pca_space_test,
               3, 5, 1, get_uncertainty_MVE)

np.savez('redox_mdm_mve.npz', unc_perc_MVE, test_unc_in_bin_df_ls,
         test_unc_remain_df_ls,
         test_unc_in_bin_nr_df_ls,
         test_unc_remain_nr_df_ls)




