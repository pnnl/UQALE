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

# import seaborn as sns

import math

from scipy.stats.stats import pearsonr
import random
from numpy.random import choice
import time

import glob
from sklearn.decomposition import PCA


import keras

import matplotlib.pyplot as plt
import numpy as np
from keras.regularizers import l2
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import RandomizedSearchCV

# from hyperopt import Trials, STATUS_OK, tpe
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
# %matplotlib inline

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

# import seaborn as sns

import math
from sklearn.metrics import r2_score

from sklearn.ensemble import GradientBoostingRegressor

LOWER_ALPHA = 0.1
UPPER_ALPHA = 0.9

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

    rmsprop = tf.keras.optimizers.RMSprop(lr=10 ** -3)
    opt = rmsprop

    model.compile(loss='mean_squared_error', optimizer=opt)

    return model


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


sol_cos_mat = np.load('') # embedding similarity data path
sol_cos_mat = sol_cos_mat['arr_0']

similarity_train_embeddings = sol_cos_mat[0:14576, 0:14576]


pca = PCA(n_components=20)
pca.fit(x_train)
pca_space_train = pca.transform(x_train)
pca_space_val = pca.transform(x_val)
pca_space_test = pca.transform(x_test)



starting_percs = [0.05, 0.1, 0.2, 0.4]
nums_mols_to_add = [25, 50, 100, 250, 500, 1000]



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

'''
def get_test_rmse_R2(x_train_now, y_train_now, x_val_now, y_val_now, x_test, y_test, test_in_bin_idx):
    model = create_model(x_train_now)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)
    result = model.fit(x_train_now, y_train_now, batch_size=64, epochs=1000,
                       verbose=0, validation_data=(x_val_now, y_val_now), callbacks=[es])

    y_test_pred = np.array(model(x_test))
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_R2 = r2_score(y_pred=y_test_pred, y_true=y_test)
    test_rmse_bin = mean_squared_error(y_test[test_in_bin_idx], y_test_pred[test_in_bin_idx], squared=False)
    test_R2_bin = r2_score(y_pred=y_test_pred[test_in_bin_idx], y_true=y_test[test_in_bin_idx])

    return test_rmse, test_rmse_bin, test_R2, test_R2_bin
'''



def get_test_rmse_R2_GBM(x_train_now, y_train_now, x_val_now, y_val_now, x_test, y_test, test_in_bin_idx):
    # lower_model = GradientBoostingRegressor(loss="quantile", alpha=LOWER_ALPHA)
    mid_model = GradientBoostingRegressor(loss="ls")
    # upper_model = GradientBoostingRegressor(loss="quantile", alpha=UPPER_ALPHA)
    # lower_model.fit(x_train_now, y_train_now)
    mid_model.fit(x_train_now, y_train_now)
    # upper_model.fit(x_train_now, y_train_now)

    y_test_pred = np.array(mid_model.predict(x_test))
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_R2 = r2_score(y_pred=y_test_pred, y_true=y_test)
    test_rmse_bin = mean_squared_error(y_test[test_in_bin_idx], y_test_pred[test_in_bin_idx], squared=False)
    test_R2_bin = r2_score(y_pred=y_test_pred[test_in_bin_idx], y_true=y_test[test_in_bin_idx])

    y_test_out = np.delete(y_test, test_in_bin_idx, axis=0)
    y_test_pred_out = np.delete(y_test_pred, test_in_bin_idx, axis=0)
    test_rmse_out = mean_squared_error(y_test_out, y_test_pred_out, squared=False)
    test_R2_out = r2_score(y_pred=y_test_pred_out, y_true=y_test_out)

    return test_rmse, test_rmse_bin, test_R2, test_R2_bin, test_rmse_out, test_R2_out


def get_density_unc(similarity, train_ind_remain, train_ind, num_NNs):
    similarity_temp = similarity[train_ind_remain, :]
    similarity_temp = similarity_temp[:, train_ind]

    val_NN_mean_dist = []
    for j in range(similarity_temp.shape[0]):
        temp_sim_array = np.array(similarity_temp[j, :])
        temp_NN_array = temp_sim_array[temp_sim_array.argsort()[-num_NNs:][::-1]]
        temp_NN_array_mean = np.mean(temp_NN_array)
        if temp_NN_array_mean < 0:
            temp_NN_array_mean = 0
        val_NN_mean_dist.append(temp_NN_array_mean)
    val_NN_mean_dist = np.array(val_NN_mean_dist)

    uncertainty = 1 - val_NN_mean_dist
    uncert_prob_dist = uncertainty / sum(uncertainty)

    return uncertainty, uncert_prob_dist


def get_AL_one_step_result(similarity, train_in_bin_idx, train_out_bin_idx, val_in_bin_idx, val_out_bin_idx,
                           test_in_bin_idx,
                           x_train, y_train, x_val, y_val, x_test, y_test,
                           n_runs, num_NNs, starting_perc, num_mols_to_add, use_random=False):
    test_rmse_ls = []
    test_rmse_bin_ls = []
    test_rmse_out_ls = []
    test_R2_ls = []
    test_R2_bin_ls = []
    test_R2_out_ls = []

    bin_in_train_count_ls = []
    # prob_dist = []
    # unc_dist = []

    for j in range(n_runs):
        # print('Run #:', j+1)

        test_rmse = []
        test_rmse_bin = []
        test_rmse_out = []
        test_R2 = []
        test_R2_bin = []
        test_R2_out = []
        bin_in_train_count = []

        x_val_now = x_val[val_out_bin_idx,]
        y_val_now = y_val[val_out_bin_idx]

        # random.seed(0)
        train_ind = random.sample(train_out_bin_idx, round(x_train.shape[0] * starting_perc))
        x_train_now = x_train[train_ind,]
        y_train_now = y_train[train_ind]
        x_train_remain = np.delete(x_train, train_ind, axis=0)
        y_train_remain = np.delete(y_train, train_ind, axis=0)

        train_ind_remain = np.delete(range(len(x_train)), train_ind, axis=0)

        test_rmse_0, test_rmse_bin_0, test_R2_0, test_R2_bin_0, test_rmse_out_0, test_R2_out_0 = get_test_rmse_R2_GBM(x_train_now, y_train_now,
                                                                               x_val_now, y_val_now,
                                                                               x_test, y_test,
                                                                               test_in_bin_idx)

        uncertainty, uncert_prob_dist = get_density_unc(similarity, train_ind_remain, train_ind, num_NNs)
        '''
        train_in_bin_ls = [i for i, x in enumerate(train_ind_remain) if x in train_in_bin_idx]
        
        train_in_bin_prob = uncert_prob_dist[train_in_bin_ls]
        train_not_in_bin_prob = np.delete(uncert_prob_dist, train_in_bin_ls, axis=0)

        prob_dist.append(train_in_bin_prob)
        prob_dist.append(train_not_in_bin_prob)

        train_in_bin_unc = uncertainty[train_in_bin_ls]
        train_not_in_bin_unc = np.delete(uncertainty, train_in_bin_ls, axis=0)

        unc_dist.append(train_in_bin_unc)
        unc_dist.append(train_not_in_bin_unc)
        '''
        if len(uncertainty) >= num_mols_to_add:
            if use_random:
                add_ind = random.sample(range(x_train_remain.shape[0]), num_mols_to_add)
            else:
                add_ind = choice(range(len(uncertainty)), num_mols_to_add, p=uncert_prob_dist, replace=False)
        else:
            add_ind = range(len(uncertainty))

        x_add_samples = x_train_remain[add_ind]
        y_add_samples = y_train_remain[add_ind]

        bin_in_train_count.append(len(list(set(train_ind_remain[add_ind]) & set(train_in_bin_idx))))

        x_train_now = np.concatenate((x_train_now, x_add_samples))
        y_train_now = np.concatenate((y_train_now, y_add_samples))
        '''
        train_ind = np.concatenate((train_ind, train_ind_remain[add_ind]))

        x_train_remain = np.delete(x_train_remain, add_ind, axis=0)
        y_train_remain = np.delete(y_train_remain, add_ind, axis=0)

        train_ind_remain = np.delete(train_ind_remain, add_ind, axis=0)
        '''
        test_rmse_one_step, test_rmse_bin_one_step, test_R2_one_step, test_R2_bin_one_step, test_rmse_out_one_step, test_R2_out_one_step = get_test_rmse_R2_GBM(x_train_now,
                                                                                                           y_train_now,
                                                                                                           x_val_now,
                                                                                                           y_val_now,
                                                                                                           x_test,
                                                                                                           y_test,
                                                                                                           test_in_bin_idx)

        test_rmse.append(test_rmse_0)
        test_rmse_bin.append(test_rmse_bin_0)
        test_rmse_out.append(test_rmse_out_0)
        test_R2.append(test_R2_0)
        test_R2_bin.append(test_R2_bin_0)
        test_R2_out.append(test_R2_out_0)
        test_rmse.append(test_rmse_one_step)
        test_rmse_bin.append(test_rmse_bin_one_step)
        test_rmse_out.append(test_rmse_out_one_step)
        test_R2.append(test_R2_one_step)
        test_R2_bin.append(test_R2_bin_one_step)
        test_R2_out.append(test_R2_out_one_step)

        test_rmse_ls.append(test_rmse)
        test_rmse_bin_ls.append(test_rmse_bin)
        test_rmse_out_ls.append(test_rmse_out)
        test_R2_ls.append(test_R2)
        test_R2_bin_ls.append(test_R2_bin)
        test_R2_out_ls.append(test_R2_out)

        bin_in_train_count_ls.append(bin_in_train_count)

    test_rmse_df = pd.DataFrame(test_rmse_ls)
    test_rmse_bin_df = pd.DataFrame(test_rmse_bin_ls)
    test_rmse_out_df = pd.DataFrame(test_rmse_out_ls)
    test_R2_df = pd.DataFrame(test_R2_ls)
    test_R2_bin_df = pd.DataFrame(test_R2_bin_ls)
    test_R2_out_df = pd.DataFrame(test_R2_out_ls)
    bin_in_train_count_df = pd.DataFrame(bin_in_train_count_ls)

    return test_rmse_df, test_rmse_bin_df, test_R2_df, test_R2_bin_df, bin_in_train_count_df, test_rmse_out_df, test_R2_out_df


def get_AL_RMSE_matrix(x_train, y_train, x_val, y_val, x_test, y_test, similarity,
                       pca_space_train, pca_space_val, pca_space_test,
                       starting_percs, nums_mols_to_add, num_PCs, num_bins, n_runs, num_NNs, use_random):
    final_rmse_ls = []
    final_rmse_bin_ls = []
    final_rmse_out_ls = []
    final_R2_ls = []
    final_R2_bin_ls = []
    final_R2_out_ls = []
    final_in_bin_count_ls = []
    for i in range(len(starting_percs)):
        nums_mols_to_add_rmse_ls = []
        nums_mols_to_add_rmse_bin_ls = []
        nums_mols_to_add_rmse_out_ls = []
        nums_mols_to_add_R2_ls = []
        nums_mols_to_add_R2_bin_ls = []
        nums_mols_to_add_R2_out_ls = []
        nums_mols_to_add_in_bin_count_ls = []
        for j in range(len(nums_mols_to_add)):
            RMSE_ls = []
            RMSE_bins_ls = []
            RMSE_outs_ls = []
            R2_ls = []
            R2_bins_ls = []
            R2_outs_ls = []
            in_bin_count_ls = []
            for k in range(num_PCs):
                val_in_bin_idx_ls, val_out_bin_idx_ls, test_in_bin_idx_ls, train_in_bin_idx_ls, train_out_bin_idx_ls = get_bin_index(
                    pca_space_train[:, k], pca_space_val[:, k], pca_space_test[:, k], num_bins)
                for l in range(num_bins):
                    test_rmse_df, test_rmse_bin_df, test_R2_df, test_R2_bin_df, bin_in_train_count_df, test_rmse_out_df, test_R2_out_df = get_AL_one_step_result(similarity,
                                                                                                   train_in_bin_idx_ls[
                                                                                                       l],
                                                                                                   train_out_bin_idx_ls[
                                                                                                       l],
                                                                                                   val_in_bin_idx_ls[l],
                                                                                                   val_out_bin_idx_ls[
                                                                                                       l],
                                                                                                   test_in_bin_idx_ls[
                                                                                                       l],
                                                                                                   x_train, y_train,
                                                                                                   x_val, y_val, x_test,
                                                                                                   y_test,
                                                                                                   n_runs, num_NNs,
                                                                                                   starting_percs[i],
                                                                                                   nums_mols_to_add[j],
                                                                                                   use_random)
                    RMSE_ls.append(test_rmse_df.mean(axis=0)[1])
                    RMSE_bins_ls.append(test_rmse_bin_df.mean(axis=0)[1])
                    RMSE_outs_ls.append(test_rmse_out_df.mean(axis=0)[1])
                    R2_ls.append(test_R2_df.mean(axis=0)[1])
                    R2_bins_ls.append(test_R2_bin_df.mean(axis=0)[1])
                    R2_outs_ls.append(test_R2_out_df.mean(axis=0)[1])
                    in_bin_count_ls.append(bin_in_train_count_df.mean(axis=0)[0])
            nums_mols_to_add_rmse_ls.append(RMSE_ls)
            nums_mols_to_add_rmse_bin_ls.append(RMSE_bins_ls)
            nums_mols_to_add_rmse_out_ls.append(RMSE_outs_ls)
            nums_mols_to_add_R2_ls.append(R2_ls)
            nums_mols_to_add_R2_bin_ls.append(R2_bins_ls)
            nums_mols_to_add_R2_out_ls.append(R2_outs_ls)
            nums_mols_to_add_in_bin_count_ls.append(in_bin_count_ls)
        final_rmse_ls.append(nums_mols_to_add_rmse_ls)
        final_rmse_bin_ls.append(nums_mols_to_add_rmse_bin_ls)
        final_rmse_out_ls.append(nums_mols_to_add_rmse_out_ls)
        final_R2_ls.append(nums_mols_to_add_R2_ls)
        final_R2_bin_ls.append(nums_mols_to_add_R2_bin_ls)
        final_R2_out_ls.append(nums_mols_to_add_R2_out_ls)
        final_in_bin_count_ls.append(nums_mols_to_add_in_bin_count_ls)

    return final_rmse_ls, final_rmse_bin_ls, final_R2_ls, final_R2_bin_ls, final_in_bin_count_ls, final_rmse_out_ls, final_R2_out_ls




start = time.time()
'''
density_final_rmse_ls, density_final_rmse_bin_ls, density_final_R2_ls, density_final_R2_bin_ls, density_final_in_bin_count_ls, final_rmse_out_ls, final_R2_out_ls = \
    get_AL_RMSE_matrix(x_train, y_train, x_val, y_val, x_test, y_test, similarity_train_embeddings,
                              pca_space_train, pca_space_val, pca_space_test,
                              starting_percs, nums_mols_to_add, 3, 5, 10, 3, False)

density_final_rmse_df = pd.DataFrame(density_final_rmse_ls)
density_final_rmse_df.to_csv('sol_gbm_embed_10runs_RMSE_df.csv', header=False, index=False)
density_final_rmse_bin_df = pd.DataFrame(density_final_rmse_bin_ls)
density_final_rmse_bin_df.to_csv('sol_gbm_embed_10runs_bin_RMSE_df.csv', header=False, index=False)
density_final_R2_df = pd.DataFrame(density_final_R2_ls)
density_final_R2_df.to_csv('sol_gbm_embed_10runs_R2_df.csv', header=False, index=False)
density_final_R2_bin_df = pd.DataFrame(density_final_R2_bin_ls)
density_final_R2_bin_df.to_csv('sol_gbm_embed_10runs_bin_R2_df.csv', header=False, index=False)
density_final_in_bin_count_df = pd.DataFrame(density_final_in_bin_count_ls)
density_final_in_bin_count_df.to_csv('sol_gbm_embed_10runs_in_bin_count_df.csv', header=False, index=False)

final_rmse_out_df = pd.DataFrame(final_rmse_out_ls)
final_rmse_out_df.to_csv('sol_gbm_embed_10runs_out_RMSE_df.csv', header=False, index=False)
final_R2_out_df = pd.DataFrame(final_R2_out_ls)
final_R2_out_df.to_csv('sol_gbm_embed_10runs_out_R2_df.csv', header=False, index=False)


'''
random_final_rmse_ls, random_final_rmse_bin_ls, random_final_R2_ls, random_final_R2_bin_ls, random_final_in_bin_count_ls, random_final_rmse_out_ls, random_final_R2_out_ls = \
    get_AL_RMSE_matrix(x_train, y_train, x_val, y_val, x_test, y_test, similarity_train_embeddings,
                              pca_space_train, pca_space_val, pca_space_test,
                              starting_percs, nums_mols_to_add, 3, 5, 10, 3, True)

random_final_rmse_df = pd.DataFrame(random_final_rmse_ls)
random_final_rmse_df.to_csv('sol_gbm_random_10runs_RMSE_df.csv', header=False, index=False)
random_final_rmse_bin_df = pd.DataFrame(random_final_rmse_bin_ls)
random_final_rmse_bin_df.to_csv('sol_gbm_random_10runs_bin_RMSE_df.csv', header=False, index=False)
random_final_R2_df = pd.DataFrame(random_final_R2_ls)
random_final_R2_df.to_csv('sol_gbm_random_10runs_R2_df.csv', header=False, index=False)
random_final_R2_bin_df = pd.DataFrame(random_final_R2_bin_ls)
random_final_R2_bin_df.to_csv('sol_gbm_random_10runs_bin_R2_df.csv', header=False, index=False)
random_final_in_bin_count_df = pd.DataFrame(random_final_in_bin_count_ls)
random_final_in_bin_count_df.to_csv('sol_gbm_random_10runs_in_bin_count_df.csv', header=False, index=False)


random_final_rmse_out_df = pd.DataFrame(random_final_rmse_out_ls)
random_final_rmse_out_df.to_csv('sol_gbm_random_10runs_out_RMSE_df.csv', header=False, index=False)
random_final_R2_out_df = pd.DataFrame(random_final_R2_out_ls)
random_final_R2_out_df.to_csv('sol_gbm_random_10runs_out_R2_df.csv', header=False, index=False)


end = time.time()

print(end - start)



