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

from edward.models import (Normal, MultivariateNormalDiag, Mixture,
                           Categorical, ParamMixture)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.utils import eprint, debug, append_to_file
import boosting_bbvi.core.utils as coreutils
from boosting_bbvi.scripts.bmf_utils import get_data, Joint, elbo
logger = coreutils.get_logger()

flags = tf.app.flags
FLAGS = tf.flags.FLAGS
flags.DEFINE_string('outdir', '/tmp', 'directory to log all the results etc.')
flags.DEFINE_integer("D", 3, "Number of latent factors.")
flags.DEFINE_integer("N", 50, "Number of users.")
flags.DEFINE_integer("M", 60, "Number of movies.")
flags.DEFINE_integer('VI_iter', 1000, '')
flags.DEFINE_integer('seed', 0, 'The random seed to use for everything.')
flags.DEFINE_float('mask_ratio', 0.5, 'Test train indicator matrix mask ratio')
tf.flags.DEFINE_enum(
    "base_dist", 'normal',
    ['normal', 'laplace', 'mvnormal', 'mvlaplace', 'mvn', 'mvl'],
    'base distribution for variational approximation')

ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)


def main(_):
    # setting up output directory
    outdir = os.path.expanduser(FLAGS.outdir)
    #os.makedirs(outdir, exist_ok=True)

    # DATA
    N, M, D, R_true, I_train, I_test = get_data()

    # MODEL
    I = tf.placeholder(tf.float32, [N, M])

    scale_uv = tf.concat(
        [tf.ones([D, N]),
            tf.ones([D, M])], axis=1)
    mean_uv = tf.concat(
        [tf.zeros([D, N]),
            tf.zeros([D, M])], axis=1)

    UV = Normal(loc=mean_uv, scale=scale_uv)
    #R = Normal(
    #    loc=tf.matmul(tf.transpose(UV[:, :N]), UV[:, N:]) * I,
    #    scale=tf.ones([N, M]))
    R = Normal(
        loc=tf.matmul(tf.transpose(UV[:, :N]), UV[:, N:]),
        scale=tf.ones([N, M]))  # generator dist. for matrix
    R_mask = R * I  # generated masked matrix

    sess = tf.InteractiveSession()
    p_joint = Joint(R_true, I_train, sess, D, N, M)

    # INFERENCE
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

    qUV = Normal(loc=mean_suv, scale=scale_suv)

    inference = ed.KLqp({UV: qUV}, data={R_mask: R_true, I: I_train})
    inference.run(n_iter=FLAGS.VI_iter)

    # CRITICISM
    cR = ed.copy(R_mask, {UV: qUV}) # reconstructed matrix
    test_mse = ed.evaluate('mean_squared_error',
                            data={
                                cR: R_true,
                                I: I_test.astype(bool)
                            })
    logger.info("iters %d ed test mse %.5f" % (FLAGS.VI_iter, test_mse))
    train_mse = ed.evaluate('mean_squared_error',
                            data={
                                cR: R_true,
                                I: I_train.astype(bool)
                            })
    logger.info("iters %d ed train mse %.5f" % (FLAGS.VI_iter, train_mse))

    elbo_t = elbo(qUV, p_joint)
    logger.info('iters %d elbo %.2f' % (FLAGS.VI_iter, elbo_t))


if __name__ == "__main__":
    tf.app.run(main)

