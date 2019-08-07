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
import boosting_bbvi.core.relbo as relbo
import boosting_bbvi.optim.bmf_step_size as opt
import boosting_bbvi.core.elbo as elboModel
from boosting_bbvi.core.utils import block_diagonal, eprint, debug, append_to_file
import boosting_bbvi.core.utils as coreutils
from boosting_bbvi.scripts.bmf_utils import get_data, Joint
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
tf.flags.DEFINE_enum(
    "base_dist", 'mvn',
    ['normal', 'laplace', 'mvnormal', 'mvlaplace', 'mvn', 'mvl'],
    'base distribution for variational approximation')
flags.DEFINE_enum('fw_variant', 'fixed', ['fixed', 'adafw'],
                  '[fixed (default)] The Frank-Wolfe variant to use.')

ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)


def main(_):
    # setting up output directory
    outdir = os.path.expanduser(FLAGS.outdir)
    os.makedirs(outdir, exist_ok=True)

    N, M, D, R_true, I_train, I_test = get_data()

    # Solution components
    weights, qUVt_components = [], []

    # Metrics to log
    # TODO replace it with file logging
    #mses, test_mses, test_lls = [], [], []
    mse_train_filename = os.path.join(outdir, 'mse_train.csv')
    open(mse_train_filename, 'w').close()

    mse_test_filename = os.path.join(outdir, 'mse_test.csv')
    open(mse_test_filename, 'w').close()

    ll_test_filename = os.path.join(outdir, 'll_test.csv')
    open(ll_test_filename, 'w').close()

    elbos_filename = os.path.join(outdir, 'elbos.csv')
    open(elbos_filename, 'w').close()

    # 'adafw', 'ada_afw', 'ada_pfw'
    if FLAGS.fw_variant.startswith('ada'):
        lipschitz_filename = os.path.join(outdir, 'lipschitz.csv')
        open(lipschitz_filename, 'w').close()

        iter_info_filename = os.path.join(outdir, 'iter_info.txt')
        open(iter_info_filename, 'w').close()

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
                R = Normal(
                    loc=tf.matmul(tf.transpose(UV[:, :N]), UV[:, N:]) * I,
                    scale=tf.ones([N, M]))

                p_joint = Joint(R_true, I_train, sess, D, N, M)

                # TODO build previous components and add here
                if t == 0:
                    fw_iterates = {}
                else:
                    # Current solution
                    prev_components = [
                        coreutils.base_loc_scale(
                            'mvn',
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

                inference = relbo.KLqp({UV: sUV}, data={R: R_true, I: I_train},
                                       fw_iterates=fw_iterates, fw_iter=t)
                inference.run(n_iter=FLAGS.LMO_iter)

                loc_s = sUV.mean().eval()
                scale_s = sUV.stddev().eval()

                data = {R: R_true, I: I_train}
                if t == 0:
                    gamma = 1.
                    lipschitz_estimate = opt.adafw_linit()
                    step_type = 'init'
                elif FLAGS.fw_variant == 'fixed':
                    step_result = opt.fixed(weights, qUVt_components, qUV_prev,
                                            loc_s, scale_s, sUV, p_joint, data, t)
                elif FLAGS.fw_variant == 'adafw':
                    step_result = opt.adaptive_fw(
                        weights, qUVt_components, qUV_prev, loc_s, scale_s, sUV,
                        UV, data, t, lipschitz_estimate)
                    step_type = step_result['step_type']
                    if step_type == 'adaptive':
                        lipschitz_estimate = step_result['l_estimate']

                if t == 0:
                    gamma = 1.
                    weights.append(gamma)
                    qUVt_components.append({'loc': loc_s, 'scale': scale_s})
                    #new_components = [sUV]
                    new_components = [coreutils.base_loc_scale(
                            'mvn',
                            loc_s,
                            scale_s,
                            multivariate=False)]
                else:
                    qUVt_components = step_result['params']
                    weights = step_result['weights']
                    gamma = step_result['gamma']
                    new_components = [
                        coreutils.base_loc_scale(
                            'mvn',
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
                if FLAGS.fw_variant.startswith('ada'):
                    append_to_file(lipschitz_filename, lipschitz_estimate)
                    append_to_file(iter_info_filename, step_type)
                    logger.info('lt = %.5f, iter_type = %s' %
                                (lipschitz_estimate, step_type))

                test_mse = ed.evaluate(
                    'mean_squared_error',
                    data={
                        qR: R_true,
                        I: I_test.astype(bool)
                    })
                logger.info("iter %d ed test mse %.5f" % (t, test_mse))
                append_to_file(mse_test_filename, test_mse)

                test_ll = ed.evaluate(
                    'log_lik',
                    data={
                        qR: R_true.astype('float32'),
                        I: I_test.astype(bool)
                    })
                logger.info("tier %d ed test ll %.5f" % (t, test_ll))
                append_to_file(ll_test_filename, test_ll)

                elbo_loss = elboModel.KLqp({UV: qUV_new}, data={R: R_true, I: I_train})
                res_update = elbo_loss.run()
                logger.info('iter %d -elbo loss %.2f' % (t, res_update['loss']))
                append_to_file(elbos_filename, -res_update['loss'])

                sess.close()


if __name__ == "__main__":
    tf.app.run(main)
