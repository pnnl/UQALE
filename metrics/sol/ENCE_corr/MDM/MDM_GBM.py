
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
from scipy.stats.stats import pearsonr
from rdkit import Chem

import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import pickle
import json

import seaborn as sns

import math
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor

from keras.models import Model

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


def val_results(x_valx, y_valx, lc_name, modelx):
    model = modelx

    print("val scores")
    pred = model.predict(x_valx).reshape(-1, )
    print(r2_score(y_pred=pred, y_true=y_valx))
    print(mean_squared_error(y_pred=pred, y_true=y_valx) ** .5)
    print(spearmanr(pred, y_valx))

    #     plt.figure()
    #     plt.plot(np.arange(len(result.history["val_loss"])), result.history["val_loss"], '.-', label="val")
    #     plt.plot(np.arange(len(result.history["loss"])), result.history["loss"], '.-', label="train")
    #     plt.legend()
    plt.savefig(lc_name + "_lc.png")


def test_results(x_testx, y_testx, acc_plot_name, modelx):
    model = modelx

    print("test scores")
    pred = model.predict(x_testx).reshape(-1, )
    print(r2_score(y_pred=pred, y_true=y_testx))
    print(mean_squared_error(y_pred=pred, y_true=y_testx) ** .5)
    print(spearmanr(pred, y_testx))

    plt.figure()
    plt.plot(pred, y_testx, 'o')
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.plot([-10, 1], [-10, 1])
    plt.savefig(acc_plot_name + "_acc.png")
    return pred


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

    rmsprop = keras.optimizers.RMSprop(lr=10 ** -3)
    opt = rmsprop

    model.compile(loss='mean_squared_error', optimizer=opt)

    return model

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

model = create_model(x_train)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
#os.system("rm best_model.h5")
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

result = model.fit(x_train, y_train, batch_size = 64, epochs = 1000,
          verbose = 2, validation_data = (x_val,y_val), callbacks = [es,mc])

model.layers[-2].output
model_new = Model(model.input, model.layers[-2].output)
model_new.summary()

pred_new = model_new(x_test)
new_train = model_new(x_train)
new_train = np.array(new_train)

LOWER_ALPHA = 0.1
UPPER_ALPHA = 0.9
lower_model = GradientBoostingRegressor(loss="quantile",
                                        alpha=LOWER_ALPHA)
mid_model = GradientBoostingRegressor(loss="ls")
upper_model = GradientBoostingRegressor(loss="quantile",
                                        alpha=UPPER_ALPHA)

lower_model.fit(new_train, y_train)
mid_model.fit(new_train, y_train)
upper_model.fit(new_train, y_train)

predictions = pd.DataFrame(y_test)
predictions['lower'] = lower_model.predict(pred_new)
predictions['mid'] = mid_model.predict(pred_new)
predictions['upper'] = upper_model.predict(pred_new)


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
print(pearsonr(GB_unc, abs(predictions['mid'] - y_test)))
