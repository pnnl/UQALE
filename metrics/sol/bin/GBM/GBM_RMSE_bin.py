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

from sklearn.ensemble import GradientBoostingRegressor

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

np.savez('sol_gbm_error.npz', err_perc_pc_gbm, test_rmse_in_bin_df_ls_gbm,
         test_rmse_remain_df_ls_gbm,
         test_rmse_in_bin_nr_df_ls_gbm,
         test_rmse_remain_nr_df_ls_gbm)