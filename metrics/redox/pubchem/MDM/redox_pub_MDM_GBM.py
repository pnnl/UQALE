
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


# tpe hopt
act = {0: 'relu', 1: 'selu', 2: 'sigmoid'}
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

model = create_model(x_train)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
#os.system("rm best_model.h5")
#mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=0)

result = model.fit(x_train, y_train, batch_size = 64, epochs = 1000,
          verbose = 0, validation_data = (x_val,y_val), callbacks = [es])


from keras.models import Model
model2 = Model(model.input, model.layers[-2].output)

pred = model(x_test)
pred2 = model2(x_test)
new_data = model2(x_train)

from sklearn.ensemble import GradientBoostingRegressor
LOWER_ALPHA = 0.1
UPPER_ALPHA = 0.9
lower_model = GradientBoostingRegressor(loss="quantile",
                                        alpha=LOWER_ALPHA)
mid_model = GradientBoostingRegressor(loss="ls")
upper_model = GradientBoostingRegressor(loss="quantile",
                                        alpha=UPPER_ALPHA)

lower_model.fit(new_data, y_train)
mid_model.fit(new_data, y_train)
upper_model.fit(new_data, y_train)

# Record actual values on test set
predictions = pd.DataFrame(y_train)
# Predict
predictions['lower'] = lower_model.predict(new_data)
predictions['mid'] = mid_model.predict(new_data)
predictions['upper'] = upper_model.predict(new_data)

pubchem_max_train_new = model2(redox_pubchem_mdm_sc)
pubchem_predictions = pd.DataFrame()
pubchem_predictions['lower'] = lower_model.predict(pubchem_max_train_new)
pubchem_predictions['mid'] = mid_model.predict(pubchem_max_train_new)
pubchem_predictions['upper'] = upper_model.predict(pubchem_max_train_new)
pubchem_GB_unc_train = np.array(np.sqrt(abs(pubchem_predictions['upper'] - pubchem_predictions['lower'])/2))

spearmanr(pubchem_GB_unc_train, redox_pubchem['max_sim'])
pearsonr(pubchem_GB_unc_train, redox_pubchem['max_sim'])
