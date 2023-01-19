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
from scipy.stats.stats import pearsonr
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
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        #if self.verbose:
            #print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), 'checkpoint.pt')
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
            # output = model(data)
            #         pred = output.reshape(64,)
            gamma, v, alpha, beta = model(data)  # torch.chunk(output, 4, dim=1)
            pred = gamma

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

def test_func_pubchem(loader, model):
    model.eval()
    with torch.no_grad():
        predicted = []
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = output
            predicted += list(pred.cpu().numpy().ravel() )

    return predicted

wdir = # MDM data path # "../../common_data/pnnl_dataset/sets/"
# to_remove = [ 'cas', 'ref', 'temp','log_kow','inchi','melting_point','sol_energy',
#              'dip_mmnt', 'dip_mmnt_vol','quad_mmnt']

train = pd.read_csv(wdir+"train.csv")
val = pd.read_csv(wdir+"val.csv")
test = pd.read_csv(wdir+"test.csv")

dataloader_dir = # GNN data path # "../../common_data/pnnl_dataset/gnn_data"

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

retrain = True
if retrain:

    hist = {"train_rmse":[], "val_rmse":[]}
    for epoch in range(0, n_epochs):
    #     train_func(epoch)
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = output.reshape(-1,)

            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

    #     model.eval()
        train_rmse = test_func(train_loader, model)
        val_rmse = test_func(val_loader, model)
    #     train_rmse = test_func(train_loader)
    #     val_rmse = test_func(val_loader)

    #     if epoch >40:
        early_stopping(val_rmse, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    #     hist["loss"].append(train_loss)
        hist["train_rmse"].append(train_rmse)
        hist["val_rmse"].append(val_rmse)
        print(f'Epoch: {epoch}, Train_rmse: {train_rmse:.3}, Val_rmse: {val_rmse:.3}')


print(model)
model.out2 = nn.Identity()
print(model)

train_t, train_p = test_func_plotting(train_loader_no_shuffle, model)
train_p_res = train_p.reshape(14576, 256)


from sklearn.ensemble import GradientBoostingRegressor
LOWER_ALPHA = 0.1
UPPER_ALPHA = 0.9
lower_model = GradientBoostingRegressor(loss="quantile",
                                        alpha=LOWER_ALPHA)
mid_model = GradientBoostingRegressor(loss="ls")
upper_model = GradientBoostingRegressor(loss="quantile",
                                        alpha=UPPER_ALPHA)

y_train = train['log_sol']
lower_model.fit(train_p_res, y_train)
mid_model.fit(train_p_res, y_train)
upper_model.fit(train_p_res, y_train)


with gzip.open("", "rb") as f: # pubchem GNN data path
    train_pubchem = pickle.load(f)
pubchem_loader = DataLoader(train_pubchem, batch_size = bs, shuffle=False, drop_last=False)


train_p_pubchem = test_func_pubchem(pubchem_loader, model)
train_p_pubchem = np.array(train_p_pubchem)
train_p_pubchem_res = train_p_pubchem.reshape(2000, 256)

pubchem_predictions = pd.DataFrame()
pubchem_predictions['lower'] = lower_model.predict(train_p_pubchem_res)
pubchem_predictions['mid'] = mid_model.predict(train_p_pubchem_res)
pubchem_predictions['upper'] = upper_model.predict(train_p_pubchem_res)

GB_unc_pubchem = np.array(np.sqrt(abs(pubchem_predictions['upper'] - pubchem_predictions['lower'])/2))

pubchem_max = # pubchem MDM data path
pubchem_max_similarity = # pubchem similarity data path

pubchem_new_smiles = list(pubchem_max['orig_smiles'])
pubchem_orig_smiles = list(pubchem_max_similarity['smiles'])
indices_A = [pubchem_orig_smiles.index(x) for x in pubchem_new_smiles]
pubchem_orig_sim = pubchem_max_similarity['max_sim'][indices_A]
pubchem_all_max_sim = pubchem_max_similarity['max_sim']

print(spearmanr(GB_unc_pubchem, pubchem_orig_sim))
print(pearsonr(GB_unc_pubchem, pubchem_orig_sim))



