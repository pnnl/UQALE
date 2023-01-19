
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
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats.stats import pearsonr
from rdkit import Chem

import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import pickle
import json

import seaborn as sns

import math


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


from sklearn.ensemble import GradientBoostingRegressor
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

predictions = pd.DataFrame(y_train)
predictions['lower'] = lower_model.predict(x_train)
predictions['mid'] = mid_model.predict(x_train)
predictions['upper'] = upper_model.predict(x_train)

GB_unc_train = np.array(np.sqrt(abs(predictions['upper'] - predictions['lower'])/2))

pubchem_predictions = pd.DataFrame()
pubchem_predictions['lower'] = lower_model.predict(redox_pubchem_mdm_sc)
pubchem_predictions['mid'] = mid_model.predict(redox_pubchem_mdm_sc)
pubchem_predictions['upper'] = upper_model.predict(redox_pubchem_mdm_sc)
pubchem_GB_unc_train = np.array(np.sqrt(abs(pubchem_predictions['upper'] - pubchem_predictions['lower'])/2))
spearmanr(pubchem_GB_unc_train, redox_pubchem['max_sim'])
pearsonr(pubchem_GB_unc_train, redox_pubchem['max_sim'])


