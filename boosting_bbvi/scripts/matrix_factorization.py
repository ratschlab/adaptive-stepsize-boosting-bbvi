"""Probabilistic matrix factorization using variational inference.

Visualizes the actual and the estimated rating matrices as heatmaps.

Follows the Edward example:-
https://github.com/blei-lab/edward/blob/master/examples/probabilistic_matrix_factorization.py
"""
import os, sys
#import matplotlib
#matplotlib.use('Agg')

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io as sio

from edward.models import (Normal, MultivariateNormalDiag, Mixture,
                           Categorical, ParamMixture)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import boosting_bbvi.core.relbo as relbo
from boosting_bbvi.core.utils import block_diagonal, eprint, debug, append_to_file
import boosting_bbvi.core.utils as coreutils
logger = coreutils.get_logger()

flags = tf.app.flags
FLAGS = tf.flags.FLAGS
flags.DEFINE_string('outdir', '/tmp', 'directory to log all the results etc.')
flags.DEFINE_integer("D", 3, "Number of latent factors.")
flags.DEFINE_integer("N", 50, "Number of users.")
flags.DEFINE_integer("M", 60, "Number of movies.")
flags.DEFINE_integer('n_fw_iter', 10, '')
flags.DEFINE_integer('LMO_iter', 1000, '')
flags.DEFINE_integer('seed', 0, 'The random seed to use for everything.')
flags.DEFINE_float('mask_ratio', 0.5, 'Test train indicator matrix mask ratio')
tf.flags.DEFINE_enum('exp', 'cbcl', ['synthetic', 'cbcl'], 'Dataset name')
tf.flags.DEFINE_enum(
    "base_dist", 'mvn',
    ['normal', 'laplace', 'mvnormal', 'mvlaplace', 'mvn', 'mvl'],
    'base distribution for variational approximation')
flags.DEFINE_enum('fw_variant', 'fixed', ['fixed'],
                  '[fixed (default)] The Frank-Wolfe variant to use.')
tf.flags.DEFINE_string('datapath', 'data/chem', 'path containing data')

ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)


def build_toy_dataset(U, V, N, M, noise_std=0.1):
    R = np.dot(np.transpose(U), V) + np.random.normal(
        0, noise_std, size=(N, M))
    return R


def get_indicators(N, M, prob_std=0.5):
    ind = np.random.binomial(1, prob_std, (N, M))
    return ind


def get_data():
    if FLAGS.exp == 'cbcl':
        tr = sio.loadmat(os.path.join(FLAGS.datapath, 'cbcl.mat'))['V']
        N,M = tr.shape
        I_train = get_indicators(N, M, FLAGS.mask_ratio)
        I_test = 1 - I_train
        return N, M, FLAGS.D, tr, I_train, I_test
    elif FLAGS.exp == 'synthetic':
        N, M, D = FLAGS.N, FLAGS.M, FLAGS.D
        # true latent factors
        U_true = np.random.randn(D, N)
        V_true = np.random.randn(D, M)
        R_true = build_toy_dataset(U_true, V_true, N, M)
        I_train = get_indicators(N, M, FLAGS.mask_ratio)
        I_test = 1 - I_train
        return N, M, D, R_true, I_train, I_test
    pass


def main(_):
    # setting up output directory
    outdir = os.path.expanduser(FLAGS.outdir)
    os.makedirs(outdir, exist_ok=True)

    N, M, D, R_true, I_train, I_test = get_data()

    # Solution components
    weights, qUVt_components = [], []

    # Metrics to log
    # TODO replace it with file logging
    mses, test_mses, test_lls = [], [], []

    for t in range(FLAGS.n_fw_iter):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(FLAGS.seed)
            sess = tf.InteractiveSession()
            with sess.as_default():
                # MODEL
                I = tf.placeholder(tf.float32, [N, M])

                scale_uv = tf.concat(
                    [tf.ones([D, N]),
                     tf.ones([D, M])], axis=1)
                mean_uv = tf.concat(
                    [tf.zeros([D, N]),
                     tf.zeros([D, M])], axis=1)

                UV = Normal(loc=mean_uv, scale=scale_uv)
                debug('UV', UV.event_shape, UV.batch_shape)
                R = Normal(
                    loc=tf.matmul(tf.transpose(UV[:, :N]), UV[:, N:]) * I,
                    scale=tf.ones([N, M]))

                # TODO build previous components and add here
                if t == 0:
                    fw_iterates = {}
                else:
                    # Current solution
                    prev_components = [
                        coreutils.base_loc_scale(
                            'mvnormal',
                            c['loc'],
                            c['scale'],
                            multivariate=False) for c in qUVt_components
                    ]
                    qUV_prev = coreutils.get_mixture(weights, prev_components)
                    fw_iterates = {UV: qUV_prev}

                # LMO (via relbo INFERENCE)
                mean_suv = tf.concat([
                    tf.get_variable("qU/loc", [D, N]),
                    tf.get_variable("qV/loc", [D, M])
                ],
                                     axis=1)
                scale_suv = tf.concat([
                    tf.nn.softplus(tf.get_variable("qU/scale", [D, N])),
                    tf.nn.softplus(tf.get_variable("qV/scale", [D, M]))
                ],
                                      axis=1)

                sUV = Normal(loc=mean_suv, scale=scale_suv)
                debug('sUV', sUV.event_shape, sUV.batch_shape)
                debug('t is ', t)

                inference = relbo.KLqp({UV: sUV}, data={R: R_true, I: I_train},
                                       fw_iterates=fw_iterates, fw_iter=t)
                inference.run(n_iter=FLAGS.LMO_iter)

                loc_s = sUV.mean().eval()
                scale_s = sUV.stddev().eval()

                # TODO move this to fixed step
                gamma = 2. / (t + 2.)
                weights = [(1. - gamma) * w for w in weights]
                weights.append(gamma)
                qUVt_components.append({'loc': loc_s, 'scale': scale_s})

                if t == 0:
                    gamma = 1.
                    new_components = [sUV]
                else:
                    new_components = [
                        coreutils.base_loc_scale(
                            'normal',
                            c['loc'],
                            c['scale'],
                            multivariate=False) for c in qUVt_components
                    ]

                qUV_new = coreutils.get_mixture(weights, new_components)

                qR = Normal(
                    loc=tf.matmul(
                        tf.transpose(qUV_new[:, :N]), qUV_new[:, N:]),
                    scale=tf.ones([N, M]))

                # CRITICISM
                test_mse = ed.evaluate(
                    'mean_squared_error',
                    data={
                        qR: R_true,
                        I: I_test.astype(bool)
                    })
                test_mses.append(test_mse)
                print('test mse', test_mse)

                test_ll = ed.evaluate(
                    'log_lik',
                    data={
                        qR: R_true.astype('float32'),
                        I: I_test.astype(bool)
                    })
                test_lls.append(test_ll)
                print('test_ll', test_ll)

                #np.savetxt(
                #    os.path.join(FLAGS.outdir, 'test_mse.csv'),
                #    test_mses,
                #    delimiter=',')
                #np.savetxt(
                #    os.path.join(FLAGS.outdir, 'test_ll.csv'),
                #    test_lls,
                #    delimiter=',')
                sess.close()


if __name__ == "__main__":
    tf.app.run(main)
