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

from scipy import stats
from scipy.stats import spearmanr
from scipy.stats.stats import pearsonr
import random
from numpy.random import choice
import time

from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp

import scipy
from sklearn.metrics import mutual_info_score

import torch

redox_pubchem = pd.read_csv('') # redox pubchem similarity data

redox_pubchem_mdm = pd.read_csv('') # redox pubchem MDM data


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

temp_colnames = list(set(train.columns.values) & set(redox_pubchem_mdm.columns.values))
temp_colnames.append('Vox(V)')
temp_colnames.append('Vred(V)')

train = train[temp_colnames]
val = val[temp_colnames]
test = test[temp_colnames]

trainx = train
valx = val
testx = test

to_drop = ['Vox(V)', 'Vred(V)']

x_train,y_train, x_test, y_test, x_val, y_val, sc = get_transformed_data(train   = trainx,
                                                                         val     = valx,
                                                                         test    = testx,
                                                                         to_drop = to_drop,
                                                                         y       = "Vox(V)")

redox_pubchem_mdm = redox_pubchem_mdm[list(set(train.columns.values) & set(redox_pubchem_mdm.columns.values))]
redox_pubchem_mdm_sc = sc.fit_transform(redox_pubchem_mdm)


# density

z = torch.load('')[1] # redox embeddings data
z_train = z[:59102,:]
z_pub = pd.read_pickle(r'') # redox pubchem embeddings data
z_all = torch.cat((z_train, z_pub[1]), dim=0)


def get_cosine_similarity_matrix(z):
    # z = torch.load(embeddings_path)[1]

    # scale_z
    z = z / torch.sqrt(torch.sum(z ** 2, axis=1)).reshape(-1, 1)

    distance_m = torch.ones((z.shape[0], z.shape[0])) * np.nan

    for i in range(z.shape[0]):
        distance_m[i, i:] = torch.sum((z[i] * z[i:]), axis=1)  # upper triangular

    distance_m = distance_m.numpy()

    return np.fmax(distance_m, distance_m.T)  # copy to lower triangular

similarity_matrix = get_cosine_similarity_matrix(z_all)
similarity_matrix_pub = similarity_matrix[59102:, :]

similarity_temp = similarity_matrix_pub
num_NNs = 3
NN_mean_dist = []
for j in range(similarity_temp.shape[0]):
    temp_sim_array = np.array(similarity_temp[j, :])
    temp_NN_array = temp_sim_array[temp_sim_array.argsort()[-num_NNs:][::-1]]
    NN_mean_dist.append(np.mean(temp_NN_array))
NN_mean_dist = np.array(NN_mean_dist)

uncertainty = 1 - NN_mean_dist

spearmanr(uncertainty, redox_pubchem['max_sim'])
pearsonr(uncertainty, redox_pubchem['max_sim'])



