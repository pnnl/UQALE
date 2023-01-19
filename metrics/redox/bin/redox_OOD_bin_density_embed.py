from __future__ import division
from __future__ import unicode_literals
import numpy as np
from rdkit import Chem
import multiprocessing
import logging
import torch
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error
import torch.nn as nn

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset

from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
# from utils import mol2graph
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import pandas as pd
from sklearn.model_selection import train_test_split
plt.style.use("ggplot")

from random import randrange
import itertools
from torch_geometric.nn import EdgeConv
import random
import os
# import deepchem as dc
# from deepchem.splits.splitters import ScaffoldSplitter

from pickle import dump, load
from sklearn.metrics import mean_absolute_error

import pickle



# from gnn_utils import create_data_list



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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


from sklearn.decomposition import PCA


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




redox_cos_mat = np.load('') # redox embeddings similarity data
redox_cos_mat = redox_cos_mat['arr_0']
similarity_test_embeddings = redox_cos_mat[(x_train.shape[0]+x_val.shape[0]):, 0:x_train.shape[0]]

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


err_perc_pc = np.array([2.72245957, 0.10157775, 0.06082016, 0.06738169, 0.1399547 ,
       0.27810159, 0.02023011, 0.09069245, 0.05760513, 0.16794629,
       0.16572783, 0.07566679, 0.02402144, 0.07742064, 0.23801566])



def get_density_test_unc(similarity_test_fingerprint_np, train_ind_remain, num_NNs):
    similarity_temp = similarity_test_fingerprint_np[:, train_ind_remain]

    NN_mean_dist = []
    for j in range(similarity_temp.shape[0]):
        temp_sim_array = np.array(similarity_temp[j, :])
        temp_NN_array = temp_sim_array[temp_sim_array.argsort()[-num_NNs:][::-1]]
        temp_NN_array_mean = np.mean(temp_NN_array)
        if temp_NN_array_mean < 0:
            temp_NN_array_mean = 0
        NN_mean_dist.append(temp_NN_array_mean)
    NN_mean_dist = np.array(NN_mean_dist)

    uncertainty = 1 - NN_mean_dist
    # uncert_prob_dist = uncertainty / sum(uncertainty)

    return uncertainty# , uncert_prob_dist

def get_unc_bin_revomal_results_density(similarity_test_fingerprint_np, num_NNs,
                                        test_in_bin_idx, train_in_bin_idx, n_exp, unc_func_GNN):
    unc_in_bin_ls = []
    unc_remain_ls = []
    unc_in_bin_nr_ls = []
    unc_remain_nr_ls = []
    for j in range(n_exp):
        unc_in_bin = []
        unc_remain = []
        for i in range(len(train_in_bin_idx)):
            train_ind = train_in_bin_idx[i]
            test_ind = test_in_bin_idx[i]

            train_ind_remain = np.delete(range(len(x_train)), train_ind, axis=0)

            test_total_uncertainty = unc_func_GNN(similarity_test_fingerprint_np, train_ind_remain, num_NNs)

            unc_in_bin.append(np.mean(np.array(test_total_uncertainty)[test_ind]))
            unc_remain.append(np.mean(np.delete(np.array(test_total_uncertainty), test_ind, axis=0)))

        train_ind_remain = range(len(x_train))
        test_total_uncertainty = unc_func_GNN(similarity_test_fingerprint_np, train_ind_remain, num_NNs)

        unc_in_bin_nr = []
        unc_remain_nr = []
        for i in range(len(test_in_bin_idx)):
            test_ind = test_in_bin_idx[i]
            unc_in_bin_nr.append(np.mean(np.array(test_total_uncertainty)[test_ind]))
            unc_remain_nr.append(np.mean(np.delete(np.array(test_total_uncertainty), test_ind, axis=0)))

        unc_in_bin_ls.append(unc_in_bin)
        unc_remain_ls.append(unc_remain)
        unc_in_bin_nr_ls.append(unc_in_bin_nr)
        unc_remain_nr_ls.append(unc_remain_nr)

    unc_in_bin_df = pd.DataFrame(unc_in_bin_ls)
    unc_remain_df = pd.DataFrame(unc_remain_ls)
    unc_in_bin_nr_df = pd.DataFrame(unc_in_bin_nr_ls)
    unc_remain_nr_df = pd.DataFrame(unc_remain_nr_ls)

    return unc_in_bin_df, unc_remain_df, unc_in_bin_nr_df, unc_remain_nr_df


def get_unc_result(similarity_test_fingerprint_np, num_NNs,
                   pca_space_train, pca_space_val, pca_space_test, num_pc, num_bins, n_exp, unc_func_GNN):
    unc_perc = []
    test_unc_in_bin_df_ls = []
    test_unc_remain_df_ls = []
    test_unc_in_bin_nr_df_ls = []
    test_unc_remain_nr_df_ls = []
    for k in range(num_pc):
        val_in_bin_idx_ls, val_out_bin_idx_ls, test_in_bin_idx_ls, train_in_bin_idx_ls, train_out_bin_idx_ls = \
            get_bin_index(pca_space_train[:, k], pca_space_val[:, k], pca_space_test[:, k], num_bins)
        test_unc_in_bin_df, test_unc_remain_df, test_unc_in_bin_nr_df, test_unc_remain_nr_df = \
            get_unc_bin_revomal_results_density(similarity_test_fingerprint_np, num_NNs,
                                                test_in_bin_idx_ls, train_in_bin_idx_ls, n_exp, unc_func_GNN)
        unc_perc.append((test_unc_in_bin_df.mean() - test_unc_in_bin_nr_df.mean()) / test_unc_in_bin_nr_df.mean())
        test_unc_in_bin_df_ls.append(test_unc_in_bin_df)
        test_unc_remain_df_ls.append(test_unc_remain_df)
        test_unc_in_bin_nr_df_ls.append(test_unc_in_bin_nr_df)
        test_unc_remain_nr_df_ls.append(test_unc_remain_nr_df)
    return np.array(unc_perc).reshape(1, num_bins * num_pc)[0], \
           test_unc_in_bin_df_ls, test_unc_remain_df_ls, test_unc_in_bin_nr_df_ls, test_unc_remain_nr_df_ls


unc_perc_density_fp, test_unc_in_bin_df_ls, test_unc_remain_df_ls, test_unc_in_bin_nr_df_ls, test_unc_remain_nr_df_ls = \
    get_unc_result(similarity_test_embeddings, 3, pca_space_train, pca_space_val, pca_space_test, 3, 5, 1, get_density_test_unc)

np.savez('redox_gnn_density.npz', unc_perc_density_fp, test_unc_in_bin_df_ls,
         test_unc_remain_df_ls,
         test_unc_in_bin_nr_df_ls,
         test_unc_remain_nr_df_ls)

print(pearsonr(err_perc_pc, unc_perc_density_fp))
print(spearmanr(err_perc_pc, unc_perc_density_fp))













