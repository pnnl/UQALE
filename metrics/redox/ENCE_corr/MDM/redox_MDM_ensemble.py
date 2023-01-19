
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
import math
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats.stats import pearsonr

# MDM data path
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

x_train,y_train, x_test, y_test, x_val, y_val, sc = get_transformed_data(train   = trainx,
                                                                         val     = valx,
                                                                         test    = testx,
                                                                         to_drop = to_drop,
                                                                         y       = "Vox(V)")

# tpe hopt
act = {0:'relu', 1:'selu', 2:'sigmoid'}

args={'a1': 2, 'a2': 2, 'a3': 0, 'a4': 1, 'a5': 2, 'bs': 2, 'd1': 0.14172091923172192, 'd2': 0.04405799333743232, \
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


model_ensemble = []
for _ in range(20):
    model = create_model(x_train)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)
    #os.system("rm best_model.h5")
    #mc = ModelCheckpoint('ensemble_best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    result = model.fit(x_train, y_train, batch_size = 64, epochs = 100,
              verbose = 0, validation_data = (x_val,y_val), callbacks = [es])
    model_ensemble.append(model)

def predict_dist_ensemble(X, model_ensemble):
    preds = [model_ensemble[i].predict(X) for i in range(20)]
    return np.hstack(preds)

def predict_point_ensemble(X, model_ensemble):
    pred_dist = predict_dist_ensemble(X, model_ensemble)
    return pred_dist.mean(axis=1)

y_pred_dist_ensemble = predict_dist_ensemble(x_test, model_ensemble)
y_pred_ensemble = predict_point_ensemble(x_test, model_ensemble)
y_pred_dist_std_ensemble = []
for i in range(len(y_pred_dist_ensemble)):
    y_pred_dist_std_ensemble.append(y_pred_dist_ensemble[i].std())

y_pred_dist_std_ensemble = []
for i in range(len(y_pred_dist_ensemble)):
    y_pred_dist_std_ensemble.append(y_pred_dist_ensemble[i].std())

y_pred_dist_std_ordered_idx_ensemble = np.argsort(y_pred_dist_std_ensemble)

y_pred_dist_std_ordered_ensemble = y_pred_dist_std_ensemble.copy()
y_pred_dist_std_ordered_ensemble.sort()

RMV_ensemble = []
RMSE_ensemble = []

for i in range(10):
    l = y_pred_dist_std_ordered_ensemble[(i * 738):((i + 1) * 738)]
    RMV_i = math.sqrt(sum(map(lambda x: x * x, l)) / len(l))
    RMV_ensemble.append(RMV_i)

    p = y_pred_ensemble[y_pred_dist_std_ordered_idx_ensemble[(i * 738):((i + 1) * 738)]]
    t = y_test[y_pred_dist_std_ordered_idx_ensemble[(i * 738):((i + 1) * 738)]]
    RMSE_i = math.sqrt(sum((p - t) ** 2) / len(p))
    RMSE_ensemble.append(RMSE_i)

    # print(RMV, RMSE)

def ENCE(RMV, RMSE):
    return sum(abs(np.array(RMV) - np.array(RMSE)) / np.array(RMV)) / len(RMV)

ENCE(RMV_ensemble, RMSE_ensemble)


print(spearmanr(y_pred_dist_std_ensemble, abs(y_pred_ensemble - y_test)))
print(pearsonr(y_pred_dist_std_ensemble, abs(y_pred_ensemble - y_test)))
