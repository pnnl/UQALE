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
from torch_geometric.loader import DataLoader
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



# from gnn_utils import create_data_list



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# from keras.models import load_model

import matplotlib.pyplot as plt
#import evidential_deep_learning as edl

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

# import seaborn as sns

import math

from scipy.stats.stats import pearsonr
import random


from sklearn.decomposition import PCA
import time

def get_transformed_data(train ,val ,test, to_drop, y):

    x_train = train.drop(to_drop, axis=1)
    x_val = val.drop(to_drop , axis=1)
    x_test = test.drop(to_drop , axis=1)

    y_train = train[y].values
    y_val = val[y].values
    y_test = test[y].values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train ,y_train, x_test, y_test, x_val, y_val, scaler

wdir = # MDM data path

to_remove = [ 'cas', 'ref', 'temp' ,'log_kow' ,'inchi' ,'melting_point' ,'sol_energy',
              'dip_mmnt', 'dip_mmnt_vol' ,'quad_mmnt', 'mass']


train = pd.read_csv(wdir +"train.csv")
val = pd.read_csv(wdir +"val.csv")
test = pd.read_csv(wdir +"test.csv")

# print(train.shape)
train = train.drop(to_remove, axis=1)
val = val.drop(to_remove, axis=1)
test = test.drop(to_remove, axis=1)
# print(train.shape)

trainx = train
valx = val
testx = test

to_drop = ['log_sol', 'smiles']

x_train ,y_train, x_test, y_test, x_val, y_val, sc = get_transformed_data(train   = trainx,
                                                                         val     = valx,
                                                                         test    = testx,
                                                                         to_drop = to_drop,
                                                                         y       = "log_sol")



pca = PCA(n_components=20)
pca.fit(x_train)
pca_space_train = pca.transform(x_train)
pca_space_val = pca.transform(x_val)
pca_space_test = pca.transform(x_test)


def get_bin_index(pc_train, pc_val, pc_test, num_bins):
    cut_val = pd.qcut(pc_train, q=num_bins, retbins=True)[1]  # equalObs(x_train[:, col_num], num_bins)

    val_in_bin_idx_ls = []
    val_out_bin_idx_ls = []
    test_in_bin_idx_ls = []
    train_in_bin_idx_ls = []
    train_out_bin_idx_ls = []

    for i in range(len(cut_val) - 1):
        val_in_bin_idx_temp = [i for i, x in enumerate((pc_val >= cut_val[i]) & (pc_val < cut_val[i + 1])) if x]
        val_out_bin_idx_temp = list(np.delete(range(len(pc_val)), val_in_bin_idx_temp, axis=0))
        test_in_bin_idx_temp = [i for i, x in enumerate((pc_test >= cut_val[i]) & (pc_test < cut_val[i + 1])) if x]
        train_in_bin_idx_temp = [i for i, x in enumerate((pc_train >= cut_val[i]) & (pc_train < cut_val[i + 1])) if x]
        train_out_bin_idx_temp = list(np.delete(range(len(pc_train)), train_in_bin_idx_temp, axis=0))

        val_in_bin_idx_ls.append(val_in_bin_idx_temp)
        val_out_bin_idx_ls.append(val_out_bin_idx_temp)
        test_in_bin_idx_ls.append(test_in_bin_idx_temp)
        train_in_bin_idx_ls.append(train_in_bin_idx_temp)
        train_out_bin_idx_ls.append(train_out_bin_idx_temp)

    return val_in_bin_idx_ls, val_out_bin_idx_ls, test_in_bin_idx_ls, train_in_bin_idx_ls, train_out_bin_idx_ls



############## GNN

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
        #torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss




def test_func(loader, model, device):
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


def test_func_plotting(loader, model, device):
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


n_features = 65  # number of node features
act = {0: torch.nn.ReLU(), 1: torch.nn.SELU(), 2: torch.nn.Sigmoid()}
bs = 64

train_loader = DataLoader(train_X, batch_size=bs, shuffle=True, drop_last=True)
val_loader = DataLoader(val_X, batch_size=bs, shuffle=True, drop_last=True)
# val_loader_no_shuffle = DataLoader(val_X, batch_size=bs, shuffle=False, drop_last=False)
test_loader_no_shuffle = DataLoader(test_X, batch_size=bs, shuffle=False, drop_last=False)

def get_RMSE_bin_revomal_results(train_X, val_X, train_loader, val_loader, test_loader_no_shuffle, y_test,
                                 val_in_bin_idx_ls, train_in_bin_idx_ls, test_in_bin_idx_ls, n_exp):
    test_rmse_in_bin_ls = []
    test_rmse_remain_ls = []
    test_rmse_in_bin_nr_ls = []
    test_rmse_remain_nr_ls = []
    for j in range(n_exp):
        test_rmse_in_bin = []
        test_rmse_remain = []
        for i in range(len(train_in_bin_idx_ls)):
            train_ind = train_in_bin_idx_ls[i]
            val_ind = val_in_bin_idx_ls[i]
            test_ind = test_in_bin_idx_ls[i]

            train_X_now = [mol for (i, mol) in enumerate(train_X) if i not in train_ind]
            train_loader_now = DataLoader(train_X_now, batch_size=bs, shuffle=True, drop_last=True)
            val_X_now = [mol for (i, mol) in enumerate(val_X) if i not in val_ind]
            val_loader_now = DataLoader(val_X_now, batch_size=bs, shuffle=True, drop_last=True)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = Net().to(device)
            adam = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer = adam
            early_stopping = EarlyStopping(patience=25, verbose=False)
            criterion = nn.MSELoss()
            n_epochs = 1000

            for epoch in range(0, n_epochs):
                model.train()
                loss_all = 0
                for data in train_loader_now:
                    data = data.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    output = output.reshape(-1, )

                    loss = criterion(output, data.y)
                    loss.backward()
                    optimizer.step()

                val_rmse = test_func(val_loader_now, model, device)
                early_stopping(val_rmse, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            test_t, test_p = test_func_plotting(test_loader_no_shuffle, model, device)
            test_rmse_in_bin.append(mean_squared_error(y_test[test_ind], test_p[test_ind], squared=False))
            test_rmse_remain.append(mean_squared_error(np.delete(y_test, test_ind, axis=0), np.delete(test_p, test_ind, axis=0), squared=False))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net().to(device)
        adam = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = adam
        early_stopping = EarlyStopping(patience=25, verbose=False)
        criterion = nn.MSELoss()
        n_epochs = 1000

        for epoch in range(0, n_epochs):
            model.train()
            loss_all = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                output = output.reshape(-1, )

                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()

            val_rmse = test_func(val_loader, model, device)
            early_stopping(val_rmse, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        test_t, test_p = test_func_plotting(test_loader_no_shuffle, model, device)
        test_rmse_in_bin_nr = []
        test_rmse_remain_nr = []
        for i in range(len(test_in_bin_idx_ls)):
            test_ind = test_in_bin_idx_ls[i]
            test_rmse_in_bin_nr.append(mean_squared_error(y_test[test_ind], test_p[test_ind], squared=False))
            test_rmse_remain_nr.append(mean_squared_error(np.delete(y_test, test_ind, axis=0), np.delete(test_p, test_ind, axis=0), squared=False))
        test_rmse_in_bin_ls.append(test_rmse_in_bin)
        test_rmse_remain_ls.append(test_rmse_remain)
        test_rmse_in_bin_nr_ls.append(test_rmse_in_bin_nr)
        test_rmse_remain_nr_ls.append(test_rmse_remain_nr)
    test_rmse_in_bin_df = pd.DataFrame(test_rmse_in_bin_ls)
    test_rmse_remain_df = pd.DataFrame(test_rmse_remain_ls)
    test_rmse_in_bin_nr_df = pd.DataFrame(test_rmse_in_bin_nr_ls)
    test_rmse_remain_nr_df = pd.DataFrame(test_rmse_remain_nr_ls)
    return test_rmse_in_bin_df, test_rmse_remain_df, test_rmse_in_bin_nr_df, test_rmse_remain_nr_df


def get_RMSE_result(train_X, val_X, train_loader, val_loader, test_loader_no_shuffle,
                    y_test, num_pc, num_bins, pca_ncomp, n_exp):
    pca = PCA(n_components=pca_ncomp)
    pca.fit(x_train)
    pca_space_train = pca.transform(x_train)
    pca_space_val = pca.transform(x_val)
    pca_space_test = pca.transform(x_test)
    err_perc_pc = []

    test_rmse_in_bin_df_ls = []
    test_rmse_remain_df_ls = []
    test_rmse_in_bin_nr_df_ls = []
    test_rmse_remain_nr_df_ls = []

    for k in range(num_pc):
        val_in_bin_idx_ls, val_out_bin_idx_ls, test_in_bin_idx_ls, train_in_bin_idx_ls, train_out_bin_idx_ls = \
            get_bin_index(pca_space_train[:, k], pca_space_val[:, k], pca_space_test[:, k], num_bins)
        test_rmse_in_bin_df, test_rmse_remain_df, test_rmse_in_bin_nr_df, test_rmse_remain_nr_df = get_RMSE_bin_revomal_results(
            train_X, val_X, train_loader, val_loader, test_loader_no_shuffle, y_test,
            val_in_bin_idx_ls, train_in_bin_idx_ls, test_in_bin_idx_ls, n_exp)
        err_perc_pc.append((test_rmse_in_bin_df.mean() - test_rmse_in_bin_nr_df.mean()) / test_rmse_in_bin_nr_df.mean())
        test_rmse_in_bin_df_ls.append(test_rmse_in_bin_df)
        test_rmse_remain_df_ls.append(test_rmse_remain_df)
        test_rmse_in_bin_nr_df_ls.append(test_rmse_in_bin_nr_df)
        test_rmse_remain_nr_df_ls.append(test_rmse_remain_nr_df)

    return np.array(err_perc_pc).reshape(1, num_pc * num_bins)[
               0], test_rmse_in_bin_df_ls, test_rmse_remain_df_ls, test_rmse_in_bin_nr_df_ls, test_rmse_remain_nr_df_ls

start = time.time()
err_perc_pc, test_rmse_in_bin_df_ls, test_rmse_remain_df_ls, test_rmse_in_bin_nr_df_ls, test_rmse_remain_nr_df_ls = \
    get_RMSE_result(train_X, val_X, train_loader, val_loader, test_loader_no_shuffle, y_test, 3, 5, 20, 1)
end = time.time()
print(end - start)


np.savez('sol_gnn_error.npz', err_perc_pc, test_rmse_in_bin_df_ls,
         test_rmse_remain_df_ls,
         test_rmse_in_bin_nr_df_ls,
         test_rmse_remain_nr_df_ls)











