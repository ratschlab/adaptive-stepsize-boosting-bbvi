from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.special import expit as sigmoid
from scipy.misc import logsumexp
from sklearn.metrics import roc_auc_score

from edward.models import Bernoulli, Normal, Empirical, MultivariateNormalDiag
from edward.models import Mixture, Categorical

import boosting_bbvi.core.relbo as relbo
import boosting_bbvi.core.utils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_enum('exp', 'chem', [
    'synthetic_linearly_sep', 'synthetic_not_linearly_sep', 'synthetic_1d',
    'synthetic_1d_bimodal', 'chem', 'wine', 'eicu_icu_mortality',
    'eicu_hospital_mortality'
], 'Dataset name')
tf.flags.DEFINE_string('datapath', 'data/chem', 'path containing data')

def build_1d(N=40, noise_std=0.1):
    D = 1
    X = np.linspace(-6, 6, num=N)
    y = np.tanh(X) + np.random.normal(0, noise_std, size=N)
    y[y < 0.5] = 0
    y[y >= 0.5] = 1
    X = (X - 4.0) / 4.0
    X = X.reshape((N, D))
    return X, y

def build_1d_bimodal(N=40, noise_std=0.1):
    D = 1
    X = np.linspace(-6, 60, num=N)
    y = np.tanh(X) + np.random.normal(0, noise_std, size=N)
    y[y < 0.5] = 0
    y[y >= 0.5] = 1
    X = (X - 4.0) / 4.0

    X = np.append(X, X[y == 0] + 20)
    y = np.append(y, np.zeros((y[y == 0]).shape))

    X = X.reshape((X.shape[0], D))

    plt.scatter(X,y)
    plt.show()

    return X, y

def build_linearly_separable():
    Os = np.array([
        np.array([3.0, 3.0]) + np.random.normal([0, 0], [2., 2.])
        for _ in range(100)
    ])
    Xs = np.array([
        np.array([-3.0, -3.0]) + np.random.normal([0, 0], [2., 2.])
        for _ in range(100)
    ])

    X = np.vstack((Os, Xs))
    y = np.concatenate([np.zeros(100), np.ones(100)])
    idxs = np.arange(200)
    np.random.shuffle(idxs)

    return X[idxs,:], y[idxs]

def build_xs_and_os():
    Os1 = np.array([
        np.array([3.0, 3.0]) + np.random.normal([0, 0], [1., 1.])
        for _ in range(100)
    ])
    Xs1 = np.array([
        np.array([3.0, -3.0]) + np.random.normal([0, 0], [1., 1.])
        for _ in range(100)
    ])
    Os2 = np.array([
        np.array([-3.0, -3.0]) + np.random.normal([0, 0], [1., 1.])
        for _ in range(100)
    ])
    Xs2 = np.array([
        np.array([-3.0, 3.0]) + np.random.normal([0, 0], [1., 1.])
        for _ in range(100)
    ])
    Os = np.vstack((Os1, Os2))
    Xs = np.vstack((Xs1, Xs2))
    X = np.vstack((Os, Xs))
    y = np.concatenate([np.zeros(200), np.ones(200)])
    idxs = np.arange(400)
    np.random.shuffle(idxs)

    return X[idxs,:], y[idxs]

def load_chem_data(path):
    dat = np.load(path)
    if 'X' in dat:
        X = dat['X']
    else:
        X = sp.csr_matrix((dat['data'], dat['indices'], dat['indptr']),
                          shape=dat['shape'])
    y = dat['y']
    y[y <0] = 0
    return X, y

def get_chem_data():
    traindatapath = os.path.join(FLAGS.datapath, 'ds1.100_train.npz')
    Xtrain, ytrain = load_chem_data(traindatapath)
    testdatapath = os.path.join(FLAGS.datapath, 'ds1.100_test.npz')
    Xtest, ytest = load_chem_data(testdatapath)
    return ((Xtrain, ytrain), (Xtest, ytest))

# FIXME: This is never used
def add_bias_column(X):
    N,D = X.shape
    ret = np.c_[X, np.ones(N)]
    return ret

def load_wine_data():
    basepath = os.path.expanduser("data/wine")
    filename = os.path.join(basepath, 'train_test_split.npz')
    data = np.load(filename)
    return (data['Xtrain'], data['ytrain']), (data['Xtest'], data['ytest'])

def load_eicu(task='icu_mortality'):
    basepath = os.path.join(FLAGS.datapath, "train_test_split.npz")
    data = np.load(basepath)

    # N.B. we only deal with classification,
    # no regression on length of icu stay.
    tasks = ['icu_mortality', 'hospital_mortality', 'length_of_icu_stay']
    col = tasks.index(task)

    return (data['xtrain'], data['ytrain'][:,col]), (data['xtest'], data['ytest'][:,col])

def load_eicu_bak(task='icu_mortality'):
    '''TODO. We need to release this data. Gideon <gideon@inf.ethz.ch>'''
    basepath = "data/eicu/train_test_split.npz"
    data = np.load(basepath)

    # N.B. we only deal with classification, no regression on length of icu stay.
    tasks = ['icu_mortality', 'hospital_mortality', 'length_of_icu_stay']
    col = tasks.index(task)

    return (data['xtrain'], data['ytrain'][:,col]), (data['xtest'], data['ytest'][:,col])

def get_data():
    if FLAGS.exp == 'synthetic_linearly_sep':
        Xtrain,ytrain = build_linearly_separable()
        Xtest,ytest = build_linearly_separable()
        #fig, ax = plt.subplots() # TODO dup below
        #ax.scatter(Xtrain[ytrain == 0,0], Xtrain[ytrain == 0,1], marker='o')
        #ax.scatter(Xtrain[ytrain == 1,0], Xtrain[ytrain == 1,1], marker='x')
        #plt.show()
    elif FLAGS.exp == 'synthetic_not_linearly_sep':
        Xtrain,ytrain = build_xs_and_os()
        Xtest,ytest = build_xs_and_os()
    elif FLAGS.exp == 'synthetic_1d':
        return build_1d(), build_1d()
    elif FLAGS.exp == 'synthetic_1d_bimodal':
        return build_1d_bimodal(), build_1d_bimodal()
    elif FLAGS.exp == 'chem':
        ((Xtrain, ytrain), (Xtest, ytest)) = get_chem_data()
    elif FLAGS.exp == 'wine':
        ((Xtrain, ytrain), (Xtest, ytest)) = load_wine_data()
    elif FLAGS.exp == 'eicu_icu_mortality':
        ((Xtrain, ytrain), (Xtest, ytest)) = load_eicu('icu_mortality')
    elif FLAGS.exp == 'eicu_hospital_mortality':
        ((Xtrain, ytrain), (Xtest, ytest)) = load_eicu('hospital_mortality')
    else:
        raise Exception("unknown experiment")
    return (Xtrain, ytrain), (Xtest, ytest)


class Joint:
    '''Wrapper to handle calculating the joint probability of data

    log p(y, w | X) = log [ p(y | X, w) * p(w) ]
    '''
    def __init__(self, X, y, sess, n_samples, logger=None):
        """Initialize the distribution.

            Constructs the graph for evaluation of joint probabilities
            of data X and weights (latent vars) w
        
            Args:
                X:  [N x D] data
                y:  [D] predicted target variable
                sess: tensorflow session
                n_samples: number of monte carlo samples to compute expectation
        """
        self.sess = sess
        self.n_samples = n_samples
        # (N, ) -> (N, n_samples)
        # np.tile(y[:, np.newaxis], (1, self.n_samples))
        y_matrix = np.repeat(y[:, np.newaxis], self.n_samples, axis=1)
        if logger is not None: self.logger = logger

        # Define the model graph
        N, D = X.shape
        self.X = tf.convert_to_tensor(X, dtype=tf.float32)
        self.Y = tf.convert_to_tensor(y_matrix, dtype=tf.float32)
        self.W = tf.get_variable('samples', (self.n_samples, D), tf.float32,
                initializer=tf.zeros_initializer())
        # (N, n_samples)
        self.py = Bernoulli(logits=tf.matmul(self.X, tf.transpose(self.W)))
        self.w_prior = Normal(
            loc=tf.zeros([self.n_samples, D], tf.float32),
            scale=tf.ones([self.n_samples, D], tf.float32))
        # to get prior log probability would be summed across the D features
        # [n_samples D] -> [n_samples]
        self.prior = tf.reduce_sum(self.w_prior.log_prob(self.W), axis=1)
        log_likelihoods = self.py.log_prob(self.Y) # (N, n_samples)
        self.ll = tf.reduce_sum(log_likelihoods, axis=0) # (n_samples, )
        self.joint = self.ll + self.prior

    def log_prob(self, samples):
        """Log probability of samples.
        
        Since X is already given. samples, like for target distribution, for
        base distributions on approximation, for individual atoms are all
        samples of w.

        Args:
            samples: [self.n_samples x D] tensor
        Returns:
            [self.n_samples, ] joint log probability of samples, X, y
        """
        assert samples.shape[0] == self.n_samples, 'Different number of samples'
        self.sess.run(self.W.assign(samples))
        return self.joint
