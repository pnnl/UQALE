
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


from sklearn.ensemble import GradientBoostingRegressor


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

unc_perc_gbm, test_unc_in_bin_df_ls, test_unc_remain_df_ls, test_unc_in_bin_nr_df_ls, test_unc_remain_nr_df_ls = get_unc_result(x_train, y_train, x_val, y_val, x_test, y_test,
                              pca_space_train, pca_space_val, pca_space_test,
                              3, 5, 1, get_uncertainty_gbm)

np.savez('sol_gbm.npz', unc_perc_gbm, test_unc_in_bin_df_ls,
         test_unc_remain_df_ls,
         test_unc_in_bin_nr_df_ls,
         test_unc_remain_nr_df_ls)