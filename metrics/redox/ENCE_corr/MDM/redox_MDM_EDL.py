import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from keras.models import load_model

import matplotlib.pyplot as plt
#import evidential_deep_learning as edl

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

import seaborn as sns

import math

from scipy.stats.stats import pearsonr
import random
from numpy.random import choice
import time

from scipy import stats

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

def MSE(y, y_, reduce=True):
    ax = list(range(1, len(y.shape)))

    mse = tf.reduce_mean((y-y_)**2, axis=ax)
    return tf.reduce_mean(mse) if reduce else mse

def RMSE(y, y_):
    rmse = tf.sqrt(tf.reduce_mean((y-y_)**2))
    return rmse

def Gaussian_NLL(y, mu, sigma, reduce=True):
    ax = list(range(1, len(y.shape)))

    logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
    loss = tf.reduce_mean(-logprob, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss

def Gaussian_NLL_logvar(y, mu, logvar, reduce=True):
    ax = list(range(1, len(y.shape)))

    log_liklihood = 0.5 * (
        -tf.exp(-logvar)*(mu-y)**2 - tf.math.log(2*tf.constant(np.pi, dtype=logvar.dtype)) - logvar
    )
    loss = tf.reduce_mean(-log_liklihood, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*tf.math.log(np.pi/v)  \
        - alpha*tf.math.log(twoBlambda)  \
        + (alpha+0.5) * tf.math.log(v*(y-gamma)**2 + twoBlambda)  \
        + tf.math.lgamma(alpha)  \
        - tf.math.lgamma(alpha+0.5)

    return tf.reduce_mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*tf.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*tf.math.log(tf.abs(v2)/tf.abs(v1))  \
        - 0.5 + a2*tf.math.log(b1/b2)  \
        - (tf.math.lgamma(a1) - tf.math.lgamma(a2))  \
        + (a1 - a2)*tf.math.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = tf.abs(y-gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi

    return tf.reduce_mean(reg) if reduce else reg

def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg

from tensorflow.keras.layers import Layer


class DenseNormal(Layer):
    def __init__(self, units):
        super(DenseNormal, self).__init__()
        self.units = int(units)
        self.dense = Dense(2 * self.units)

    def call(self, x):
        output = self.dense(x)
        mu, logsigma = tf.split(output, 2, axis=-1)
        sigma = tf.nn.softplus(logsigma) + 1e-6
        return tf.concat([mu, sigma], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.units)

    def get_config(self):
        base_config = super(DenseNormal, self).get_config()
        base_config['units'] = self.units
        return base_config


class DenseNormalGamma(Layer):
    def __init__(self, units):
        super(DenseNormalGamma, self).__init__()
        self.units = int(units)
        self.dense = Dense(4 * self.units, activation=None)

    def evidence(self, x):
        # return tf.exp(x)
        return tf.nn.softplus(x)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4 * self.units)

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config['units'] = self.units
        return base_config


class DenseDirichlet(Layer):
    def __init__(self, units):
        super(DenseDirichlet, self).__init__()
        self.units = int(units)
        self.dense = Dense(int(units))

    def call(self, x):
        output = self.dense(x)
        evidence = tf.exp(output)
        alpha = evidence + 1
        prob = alpha / tf.reduce_sum(alpha, 1, keepdims=True)
        return tf.concat([alpha, prob], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.units)


class DenseSigmoid(Layer):
    def __init__(self, units):
        super(DenseSigmoid, self).__init__()
        self.units = int(units)
        self.dense = Dense(int(units))

    def call(self, x):
        logits = self.dense(x)
        prob = tf.nn.sigmoid(logits)
        return [logits, prob]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


# tpe hopt
act = {0:'relu', 1:'selu', 2:'sigmoid'}

args={'a1': 2, 'a2': 2, 'a3': 0, 'a4': 1, 'a5': 2, 'bs': 2, 'd1': 0.14172091923172192, 'd2': 0.04405799333743232, \
      'd3': 0.011693878279452341, 'd4': 0.4804983581922393, 'd5': 0.43108843234185323, 'h1': 256.0, 'h2': 128.0, \
      'h3': 384.0, 'h4': 128.0, 'h5': 576.0, 'lr': 0, 'nfc': 0, 'opt': 0}


# Define our model with an evidential output
model = tf.keras.Sequential([
    Dense(int(args['h1']), input_shape = (x_train.shape[1],)),
    Activation(act[args['a1']] ),
    Dropout(args['d1'] ),
    Dense(int(args['h2'])  ),
    Activation(act[args['a2']] ),
    Dropout(args['d2'] ),
    DenseNormalGamma(1),
])

# Custom loss function to handle the custom regularizer coefficient
def EvidentialRegressionLoss(true, pred):
    return EvidentialRegression(true, pred, coeff=0.003)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
# Compile and fit the model!
model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-4),
    loss=EvidentialRegressionLoss)
model.fit(x_train, y_train, batch_size=64, epochs=1000, validation_data = (x_val, y_val), verbose = 0, callbacks=[callback])


def get_EDL_unc(model, x_test):
    y_test_pred = model(x_test)
    data_uncertainty = y_test_pred[:, 3] / (y_test_pred[:, 2] - 1)
    model_uncertainty = y_test_pred[:, 3] / (y_test_pred[:, 1] * (y_test_pred[:, 2] - 1))
    total_uncertainty = data_uncertainty + model_uncertainty
    return y_test_pred, data_uncertainty, model_uncertainty, total_uncertainty


def ENCE(RMV, RMSE):
    return sum(abs(np.array(RMV) - np.array(RMSE)) / np.array(RMV)) / len(RMV)


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

y_test_pred, data_uncertainty, model_uncertainty, total_uncertainty = get_EDL_unc(model, x_test)
ENCE_EDL_MDM = get_ENCE(total_uncertainty, 10, y_test_pred[:, 0], y_test)
ENCE_EDL_MDM

pearson_corr_EDL, spearman_corr_EDL = get_correlation(total_uncertainty, y_test_pred[:, 0], y_test)
print(pearson_corr_EDL)
print(spearman_corr_EDL)











