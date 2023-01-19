
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
from scipy import stats


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

import seaborn as sns

import math


def get_ENCE(uncertainty, num_bins, y_test_pred, y_test):
    uncertainty = np.array(uncertainty)
    uncertainty_ordered_idx = np.argsort(uncertainty)

    uncertainty_ordered = uncertainty.copy()
    uncertainty_ordered.sort()

    RMV = []
    RMSE = []

    bin_size = round(len(uncertainty) / num_bins)

    for i in range(num_bins):
        l = uncertainty_ordered[(i * bin_size):((i + 1) * bin_size)]
        RMV_i = math.sqrt(sum(map(lambda x: x * x, l)) / len(l))
        RMV.append(RMV_i)

        p = np.array(y_test_pred)[uncertainty_ordered_idx[(i * bin_size):((i + 1) * bin_size)]]
        t = y_test[uncertainty_ordered_idx[(i * bin_size):((i + 1) * bin_size)]]
        RMSE_i = math.sqrt(sum((p - t) ** 2) / len(p))
        RMSE.append(RMSE_i)

    return ENCE(RMV, RMSE)


def get_correlation(uncertainty, y_test_pred, y_test):
    spearman_corr = stats.spearmanr(uncertainty, abs(y_test_pred - y_test))
    pearson_corr = stats.pearsonr(uncertainty, abs(y_test_pred - y_test))
    return pearson_corr, spearman_corr

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



# MDM model
model = create_model(x_train)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
#os.system("rm best_model.h5")
#mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=0)

result = model.fit(x_train, y_train, batch_size = 64, epochs = 1000,
          verbose = 0, validation_data = (x_val,y_val), callbacks = [es])


# fingerprint similarity
similarity_all = pd.read_csv('D:\\ESMI\\df_similarity.csv')
similarity_train_fingerprint = similarity_all.iloc[0:14576, 1:14577]
similarity_train_fingerprint_np = np.array(similarity_train_fingerprint)
similarity_test_fingerprint = similarity_all.iloc[15862:, 1:14577]
similarity_test_fingerprint_np = np.array(similarity_test_fingerprint)

# embedding similarity
# sol_cos_mat = np.load('D:\\ESMI\\sol_cos_mat.npz')
# sol_cos_mat = sol_cos_mat['arr_0']
# sol_cos_mat_test_np = np.array(sol_cos_mat[15862:, 0:14576])



similarity_temp = similarity_test_fingerprint_np
num_NNs = 3

val_NN_mean_dist = []
for j in range(similarity_temp.shape[0]):
    temp_sim_array = np.array(similarity_temp[j, :])
    temp_NN_array = temp_sim_array[temp_sim_array.argsort()[-num_NNs:][::-1]]
    val_NN_mean_dist.append(np.mean(temp_NN_array))
val_NN_mean_dist = np.array(val_NN_mean_dist)

uncertainty = 1- val_NN_mean_dist

y_test_pred = model.predict(x_test)
y_test_pred = y_test_pred.reshape(1,1287)[0]

pearson_corr, spearman_corr = get_correlation(uncertainty, y_test_pred, y_test)
print(pearson_corr)
print(spearman_corr)





# GBM model
from sklearn.ensemble import GradientBoostingRegressor

mid_model = GradientBoostingRegressor(loss="ls")
mid_model.fit(x_train, y_train)

y_test_pred = np.array(mid_model.predict(x_test))

pearson_corr, spearman_corr = get_correlation(uncertainty, y_test_pred, y_test)
print(pearson_corr)
print(spearman_corr)




# GNN model
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

%matplotlib inline
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
from torch_geometric.data import DataLoader
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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def train_func(epoch):
    model.train()
    #     loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.reshape(64, )

        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()


def test_func(loader, model):
    model.eval()
    with torch.no_grad():
        target, predicted = [], []
        for data in loader:
            data = data.to(device)
            output = model(data)
            #         pred = output.reshape(64,)
            pred = output

            target += list(data.y.cpu().numpy().ravel())
            predicted += list(pred.cpu().numpy().ravel())

    return mean_squared_error(y_true=target, y_pred=predicted)


def test_func_plotting(loader, model):
    model.eval()

    with torch.no_grad():
        target, predicted = [], []
        for data in loader:
            data = data.to(device)
            output = model(data)
            #         pred = output.reshape(64,)
            pred = output

            target += list(data.y.cpu().numpy().ravel())
            predicted += list(pred.cpu().numpy().ravel())

    return np.array(target), np.array(predicted)


dataloader_dir = # GNN data path

import gzip, pickle
with gzip.open(dataloader_dir+"/train.pkl.gz", "rb") as f:
    train_X = pickle.load(f)

with gzip.open(dataloader_dir+"/val.pkl.gz", "rb") as f:
    val_X = pickle.load(f)

with gzip.open(dataloader_dir+"/test.pkl.gz", "rb") as f:
    test_X = pickle.load(f)

n_features = 65 # number of node features
act = {0: torch.nn.ReLU(), 1:torch.nn.SELU(), 2:torch.nn.Sigmoid()}
bs = 64

train_loader = DataLoader(train_X, batch_size=bs, shuffle=True, drop_last=True)
val_loader = DataLoader(val_X, batch_size=bs, shuffle=True, drop_last=True)
test_loader = DataLoader(test_X, batch_size=bs, shuffle=False, drop_last=False)

train_loader_no_shuffle = DataLoader(train_X, batch_size = bs, shuffle=False, drop_last=False)
val_loader_no_shuffle = DataLoader(val_X, batch_size = bs, shuffle=False, drop_last=False)

params = {'a1': 0, 'a2': 2, 'a3': 1, 'a4': 2, 'bs': 1, 'd1': 0.015105134306121593, 'd2': 0.3431295462686682, \
          'd3': 0.602688496976768, 'd4': 0.9532038077650021, 'e1': 256.0, 'eact1': 0, 'edo1': 0.4813038851902818,\
          'f1': 256.0, 'f2': 256.0, 'f3': 160.0, 'f4': 24.0, 'g1': 256.0, 'g2': 320.0, 'g21': 448.0,\
          'g22': 512.0, 'gact1': 2, 'gact2': 2, 'gact21': 2, 'gact22': 0, 'gact31': 2, 'gact32': 1, 'gact33': 1,\
          'gdo1': 0.9444250299450242, 'gdo2': 0.8341272742321129, 'gdo21': 0.7675340644596443,\
          'gdo22': 0.21498171859119775, 'gdo31': 0.8236003195596049, 'gdo32': 0.6040220843354102,\
          'gdo33': 0.21007469160431758, 'lr': 0, 'nfc': 0, 'ngl': 1, 'opt': 0}

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCNConv(n_features, int(params['g1']), cached=False)
        self.gcn2 = GCNConv( int(params['g1']), int(params['g2']), cached=False)
        self.gcn21 = GCNConv( int(params['g2']), int(params['g21']), cached=False)
        self.gcn22 = GCNConv( int(params['g21']), int(params['g22']), cached=False)

        self.gcn31 = GCNConv(int(params['g2']), int(params['e1']), cached=False)
        self.gcn32 = GCNConv(int(params['g21']), int(params['e1']), cached=False)
        self.gcn33 = GCNConv(int(params['g22']), int(params['e1']), cached=False)

        self.gdo1 = nn.Dropout(p = params['gdo1'] )
        self.gdo2 = nn.Dropout(p = params['gdo2'] )
        self.gdo31 = nn.Dropout(p = params['gdo31'] )
        self.gdo21 = nn.Dropout(p = params['gdo21'] )
        self.gdo32 = nn.Dropout(p = params['gdo32'] )
        self.gdo22 = nn.Dropout(p = params['gdo22'] )
        self.gdo33 = nn.Dropout(p = params['gdo33'] )

        self.gact1 = act[params['gact1'] ]
        self.gact2 = act[params['gact2'] ]
        self.gact31 = act[params['gact31']]
        self.gact21 = act[params['gact21'] ]
        self.gact32 = act[params['gact32'] ]
        self.gact22 = act[params['gact22'] ]
        self.gact33 = act[params['gact33'] ]

        self.ecn1 = EdgeConv(nn = nn.Sequential(nn.Linear(n_features*2, int(params['e1']) ),
                                          nn.ReLU(),
                                          nn.Linear( int(params['e1']) , int(params['f1'])  ),))

        self.edo1 = nn.Dropout(p = params['edo1'] )
        self.eact1 = act[params['eact1'] ]


        self.fc1 = Linear( int(params['e1'])+ int(params['f1']), int(params['f1']))
        self.dropout1 = nn.Dropout(p = params['d1'] )
        self.act1 = act[params['a1']]

        self.fc2 = Linear(int(params['f1']), int(params['f2']))
        self.dropout2 = nn.Dropout(p = params['d2'] )
        self.act2 = act[params['a2']]

        self.fc3 = Linear(int(params['f2']), int(params['f3']))
        self.dropout3 = nn.Dropout(p = params['d3'] )
        self.act3 = act[params['a3']]

        self.fc4 = Linear(int(params['f3']), int(params['f4']))
        self.dropout4 = nn.Dropout(p = params['d4'] )
        self.act4 = act[params['a4']]

        self.out2 = Linear(int(params['f2']), 1)
        self.out3 = Linear(int(params['f3']), 1)
        self.out4 = Linear(int(params['f4']), 1)


    def forward(self, data):
        node_x, edge_x, edge_index = data.x, data.edge_attr, data.edge_index

        x1 = self.gdo1(self.gact1( self.gcn1( node_x, edge_index ) ) )
#         if params['ngl'] == 2:
#             x1 = self.gdo2(self.gact2(self.gcn2(x1, edge_index)) )
#             x1 = self.gdo31(self.gact31(self.gcn31(x1, edge_index)) )
#         if params['ngl'] == 3:
        x1 = self.gdo2(self.gact2(self.gcn2(x1, edge_index)) )
        x1 = self.gdo21(self.gact21(self.gcn21(x1, edge_index)) )
        x1 = self.gdo32(self.gact32(self.gcn32(x1, edge_index)) )
#         if params['ngl'] == 4:
#         x1 = self.gdo2(self.gact2(self.gcn2(x1, edge_index)) )
#         x1 = self.gdo21(self.gact21(self.gcn21(x1, edge_index)) )
#         x1 = self.gdo22(self.gact22(self.gcn22(x1, edge_index)) )
#         x1 = self.gdo33(self.gact33(self.gcn33(x1, edge_index)) )


        x2 = self.edo1(self.eact1(self.ecn1(node_x, edge_index)) )
        x3 = torch.cat((x1,x2), 1)

        x3 = global_add_pool(x3, data.batch)
        x3 = self.dropout1(self.act1(self.fc1( x3 )))

#         if params['nfc']  == 2:
        x3 = self.dropout2(self.act2(self.fc2( x3 )))
        x3 = self.out2(x3)

#         if params['nfc']  == 3:
#         x3 = self.dropout2(self.act2(self.fc2( x3 )))
#         x3 = self.dropout3(self.act3(self.fc3( x3 )))
#         x3 = self.out3(x3)

#         if params['nfc']  == 4:
#             x3 = self.dropout2(self.act2(self.fc2( x3 )))
#             x3 = self.dropout3(self.act3(self.fc3( x3 )))
#             x3 = self.dropout4(self.act4(self.fc4( x3 )))
#             x3 = self.out4(x3)

#             x3 = self.out(x3)

        return x3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

adam = torch.optim.Adam(model.parameters(), lr = 1e-3 )
# rmsprop = torch.optim.RMSprop(model.parameters(), lr = 1e-3 )
# sgd = torch.optim.SGD(model.parameters(),lr = 1e-3 )

# opt_choice = params['opt']
# if opt_choice == 'adam':
optimizer = adam
# elif opt_choice == 'rmsprop':
# optimizer = rmsprop
# elif opt_choice == 'sgd':
#     optimizer = sgd

early_stopping = EarlyStopping(patience=25, verbose=True)
criterion = nn.MSELoss()

n_epochs = 1000

test_t, test_p = test_func_plotting(test_loader, model)

pearson_corr, spearman_corr = get_correlation(uncertainty, test_p, y_test)
print(pearson_corr)
print(spearman_corr)

