
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

model = create_model(x_train)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
#os.system("rm best_model.h5")
#mc = ModelCheckpoint('mdm_best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

result = model.fit(x_train, y_train, batch_size = 64, epochs = 1000,
          verbose = 0, validation_data = (x_val,y_val), callbacks = [es])

from sklearn.ensemble import GradientBoostingRegressor
# Set lower and upper quantile
LOWER_ALPHA = 0.1
UPPER_ALPHA = 0.9
# Each model has to be separate
lower_model = GradientBoostingRegressor(loss="quantile",
                                        alpha=LOWER_ALPHA)
# The mid model will use the default loss
mid_model = GradientBoostingRegressor(loss="ls")
upper_model = GradientBoostingRegressor(loss="quantile",
                                        alpha=UPPER_ALPHA)

from keras.models import Model

model2 = Model(model.input, model.layers[-2].output)
model2.summary()
pred2 = model2(x_test)
new_data = model2(x_train)

lower_model.fit(new_data, y_train)
mid_model.fit(new_data, y_train)
upper_model.fit(new_data, y_train)

# Record actual values on test set
predictions = pd.DataFrame(y_test)
# Predict
predictions['lower'] = lower_model.predict(pred2)
predictions['mid'] = mid_model.predict(pred2)
predictions['upper'] = upper_model.predict(pred2)


GB_unc = np.array(np.sqrt(abs(predictions['upper'] - predictions['lower'])/2))
GB_unc_ordered_idx = np.argsort(GB_unc)

GB_unc_ordered = GB_unc.copy()
GB_unc_ordered.sort()

RMV = []
RMSE = []

for i in range(10):
    l = GB_unc_ordered[(i * 738):((i + 1) * 738)]
    RMV_i = math.sqrt(sum(map(lambda x: x * x, l)) / len(l))
    RMV.append(RMV_i)

    p = predictions['mid'][GB_unc_ordered_idx[(i * 738):((i + 1) * 738)]]
    t = y_test[GB_unc_ordered_idx[(i * 738):((i + 1) * 738)]]
    RMSE_i = math.sqrt(sum((p - t) ** 2) / len(p))
    RMSE.append(RMSE_i)

    print(RMV_i, RMSE_i)


def ENCE(RMV, RMSE):
    return sum(abs(np.array(RMV) - np.array(RMSE)) / np.array(RMV)) / len(RMV)

ENCE(RMV, RMSE)


print(spearmanr(GB_unc, abs(predictions['mid'] - y_test)))
print(pearsonr(GB_unc, abs(predictions['mid'] - y_test)))

