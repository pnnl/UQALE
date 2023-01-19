
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
from scipy.stats import spearmanr
from scipy import stats

from tensorflow.keras.layers import Layer
import torch

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

train = train[list(set(train.columns.values) & set(Lizzy_max.columns.values))]
val = val[list(set(val.columns.values) & set(Lizzy_max.columns.values))]
test = test[list(set(test.columns.values) & set(Lizzy_max.columns.values))]

trainx = train
valx = val
testx = test

to_drop = ['log_sol', 'smiles']

x_train, y_train, x_test, y_test, x_val, y_val, sc = get_transformed_data(train=trainx,
                                                                          val=valx,
                                                                          test=testx,
                                                                          to_drop=to_drop,
                                                                          y="log_sol")


pubchem_max = # pubchem MDM data path
pubchem_max_similarity = # pubchem similarity data path

pubchem_new_smiles = list(pubchem_max['orig_smiles'])
pubchem_orig_smiles = list(pubchem_max_similarity['smiles'])
indices_A = [pubchem_orig_smiles.index(x) for x in pubchem_new_smiles]
pubchem_orig_sim = pubchem_max_similarity['max_sim'][indices_A]
pubchem_all_max_sim = pubchem_max_similarity['max_sim']


# fingerprint
pubchem_mols = [Chem.MolFromSmiles(x) for x in pubchem_orig_smiles]
pubchem_fingerprints = [Chem.RDKFingerprint(x) for x in pubchem_mols]

pnnl_mols = [Chem.MolFromSmiles(x) for x in list(train['smiles'])]
pnnl_fingerprints = [Chem.RDKFingerprint(x) for x in pnnl_mols]

similarity_matrix = np.zeros((len(pnnl_fingerprints), len(pubchem_fingerprints)))
for i, fingerprint1 in enumerate(pnnl_fingerprints):
    for j, fingerprint2 in enumerate(pubchem_fingerprints):
        similarity_matrix[i,j] = DataStructs.FingerprintSimilarity(fingerprint1,fingerprint2)

similarity_temp = similarity_matrix
num_NNs = 3

NN_mean_dist = []
for j in range(similarity_temp.shape[1]):
    temp_sim_array = np.array(similarity_temp[:, j])
    temp_NN_array = temp_sim_array[temp_sim_array.argsort()[-num_NNs:][::-1]]
    NN_mean_dist.append(np.mean(temp_NN_array))
NN_mean_dist = np.array(NN_mean_dist)

uncertainty = 1- NN_mean_dist

spearmanr(uncertainty, pubchem_all_max_sim)
pearsonr(uncertainty, pubchem_all_max_sim)


# embedding

z = torch.load('')[1] # solubility embeddings path
z_train = z[:14576,:]
z_pub = pd.read_pickle(r'') # pubchem embeddings path

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
similarity_matrix_pub = similarity_matrix[14576:, :]
similarity_temp = similarity_matrix_pub
num_NNs = 3
NN_mean_dist = []
for j in range(similarity_temp.shape[0]):
    temp_sim_array = np.array(similarity_temp[j, :])
    temp_NN_array = temp_sim_array[temp_sim_array.argsort()[-num_NNs:][::-1]]
    NN_mean_dist.append(np.mean(temp_NN_array))
NN_mean_dist = np.array(NN_mean_dist)

uncertainty = 1- NN_mean_dist
spearmanr(uncertainty, pubchem_all_max_sim)
pearsonr(uncertainty, pubchem_all_max_sim)






