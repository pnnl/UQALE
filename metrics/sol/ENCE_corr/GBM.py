
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
from scipy import stats
from scipy.stats.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor

def ENCE(RMV, RMSE):
    return sum(abs(np.array(RMV) - np.array(RMSE))/np.array(RMV))/len(RMV)

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


LOWER_ALPHA = 0.1
UPPER_ALPHA = 0.9
lower_model = GradientBoostingRegressor(loss="quantile",
                                        alpha=LOWER_ALPHA)
mid_model = GradientBoostingRegressor(loss="ls")
upper_model = GradientBoostingRegressor(loss="quantile",
                                        alpha=UPPER_ALPHA)

lower_model.fit(x_train, y_train)
mid_model.fit(x_train, y_train)
upper_model.fit(x_train, y_train)

predictions = pd.DataFrame(y_test)
predictions['lower'] = lower_model.predict(x_test)
predictions['mid'] = mid_model.predict(x_test)
predictions['upper'] = upper_model.predict(x_test)


GB_unc = np.array(np.sqrt(abs(predictions['upper'] - predictions['lower'])/2))
GB_unc_ordered_idx = np.argsort(GB_unc)
GB_unc_ordered = GB_unc.copy()
GB_unc_ordered.sort()

RMV = []
RMSE = []

for i in range(10):
    l = GB_unc_ordered[(i * 128):((i + 1) * 128)]
    RMV_i = math.sqrt(sum(map(lambda x: x * x, l)) / len(l))
    RMV.append(RMV_i)

    p = predictions['mid'][GB_unc_ordered_idx[(i * 128):((i + 1) * 128)]]
    t = y_test[GB_unc_ordered_idx[(i * 128):((i + 1) * 128)]]
    RMSE_i = math.sqrt(sum((p - t) ** 2) / len(p))
    RMSE.append(RMSE_i)

    print(RMV_i, RMSE_i)

print(ENCE(RMV, RMSE))

print(spearmanr(GB_unc, abs(predictions['mid'] - y_test)))
print(pearsonr(GB_unc, abs((predictions['mid'] - y_test)))
