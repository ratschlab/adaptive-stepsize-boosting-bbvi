"""Probabilistic matrix factorization using variational inference.

Visualizes the actual and the estimated rating matrices as heatmaps.

Follows the Edward example:-
https://github.com/blei-lab/edward/blob/master/examples/probabilistic_matrix_factorization.py
"""
import os, sys
import time
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
from boosting_bbvi.scripts.bmf_utils import (get_data, Joint, elbo, log_likelihood, get_random_components)
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
flags.DEFINE_enum(
    "base_dist", 'mvn0',
    ['normal', 'laplace', 'mvnormal', 'mvlaplace', 'mvn', 'mvl', 'mvn0'],
    'base distribution for variational approximation')
flags.DEFINE_enum('fw_variant', 'fixed',
                  ['fixed', 'adafw', 'ada_pfw', 'ada_afw', 'line_search'],
                  '[fixed (default)] The Frank-Wolfe variant to use.')
flags.DEFINE_boolean('restore', False, 
        'is the algorithm starting from 0 or restoring a previous solution')

ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)


def main(_):
    # setting up output directory
    outdir = os.path.expanduser(FLAGS.outdir)
    os.makedirs(outdir, exist_ok=True)

    N, M, D, R_true, I_train, I_test = get_data()
    debug('N, M, D', N, M, D)

    # Solution components
    weights, qUVt_components = [], []

    # Files to log metrics
    times_filename = os.path.join(outdir, 'times.csv')
    mse_train_filename = os.path.join(outdir, 'mse_train.csv')
    mse_test_filename = os.path.join(outdir, 'mse_test.csv')
    ll_test_filename = os.path.join(outdir, 'll_test.csv')
    ll_train_filename = os.path.join(outdir, 'll_train.csv')
    elbos_filename = os.path.join(outdir, 'elbos.csv')
    gap_filename = os.path.join(outdir, 'gap.csv')
    step_filename = os.path.join(outdir, 'steps.csv')
    # 'adafw', 'ada_afw', 'ada_pfw'
    if FLAGS.fw_variant.startswith('ada'):
        lipschitz_filename = os.path.join(outdir, 'lipschitz.csv')
        iter_info_filename = os.path.join(outdir, 'iter_info.txt')

    start = 0
    if FLAGS.restore:
        #start = 50
        #qUVt_components = get_random_components(D, N, M, start)
        #weights = np.random.dirichlet([1.] * start).astype(np.float32)
        #lipschitz_estimate = opt.adafw_linit()
        parameters = np.load(os.path.join(outdir, 'qt_latest.npz'))
        weights = list(parameters['weights'])
        start = parameters['fw_iter']
        qUVt_components = list(parameters['comps'])
        assert len(weights) == len(qUVt_components), "Inconsistent storage"
        # get lipschitz estimate from the file, could've stored it
        # in params but that would mean different saved file for
        # adaptive variants
        if FLAGS.fw_variant.startswith('ada'):
            lipschitz_filename = os.path.join(outdir, 'lipschitz.csv')
            if not os.path.isfile(lipschitz_filename):
                raise ValueError("Inconsistent storage")
            with open(lipschitz_filename, 'r') as f:
                l = f.readlines()
                lipschitz_estimate = float(l[-1].strip())
    else:
        # empty the files present in the folder already
        open(times_filename, 'w').close()
        open(mse_train_filename, 'w').close()
        open(mse_test_filename, 'w').close()
        open(ll_test_filename, 'w').close()
        open(ll_train_filename, 'w').close()
        open(elbos_filename, 'w').close()
        open(gap_filename, 'w').close()
        open(step_filename, 'w').close()
        # 'adafw', 'ada_afw', 'ada_pfw'
        if FLAGS.fw_variant.startswith('ada'):
            open(lipschitz_filename, 'w').close()
            open(iter_info_filename, 'w').close()

    for t in range(start, start + FLAGS.n_fw_iter):
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
                    loc=tf.matmul(tf.transpose(UV[:, :N]), UV[:, N:]),
                    scale=tf.ones([N, M]))  # generator dist. for matrix
                R_mask = R * I  # generated masked matrix

                p_joint = Joint(R_true, I_train, sess, D, N, M)

                if t == 0:
                    fw_iterates = {}
                else:
                    # Current solution
                    prev_components = [
                        coreutils.base_loc_scale(
                            'mvn0',
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

                #inference = relbo.KLqp({UV: sUV}, data={R: R_true, I: I_train},
                inference = relbo.KLqp({UV: sUV}, data={R_mask: R_true, I: I_train},
                                       fw_iterates=fw_iterates, fw_iter=t)
                inference.run(n_iter=FLAGS.LMO_iter)

                loc_s = sUV.mean().eval()
                scale_s = sUV.stddev().eval()
                # sUV is batched distrbution, there are issues making
                # Mixture with batch distributions. mvn0
                # with event size (D, N + M) and batch size ()
                # NOTE log_prob(sample) still returns tensor
                # mvn and multivariatenormaldiag work for 1-D not 2-D shapes
                sUV_mv = coreutils.base_loc_scale(
                    'mvn0', loc_s, scale_s, multivariate=False)
                # TODO send sUV or sUV_mv as argument to step size? sample
                # works the same way. same with log_prob

                total_time = 0.
                data = {R: R_true, I: I_train}
                if t == 0:
                    gamma = 1.
                    lipschitz_estimate = opt.adafw_linit()
                    step_type = 'init'
                elif FLAGS.fw_variant == 'fixed':
                    start_step_time = time.time()
                    step_result = opt.fixed(weights, qUVt_components, qUV_prev,
                                            loc_s, scale_s, sUV, p_joint,
                                            data, t)
                    end_step_time = time.time()
                    total_time += float(end_step_time - start_step_time)
                elif FLAGS.fw_variant == 'line_search':
                    start_step_time = time.time()
                    step_result = opt.line_search_dkl(weights, qUVt_components,
                                                      qUV_prev, loc_s, scale_s,
                                                      sUV, p_joint, data, t)
                    end_step_time = time.time()
                    total_time += float(end_step_time - start_step_time)
                elif FLAGS.fw_variant == 'adafw':
                    start_step_time = time.time()
                    step_result = opt.adaptive_fw(
                        weights, qUVt_components, qUV_prev, loc_s, scale_s,
                        sUV, p_joint, data, t, lipschitz_estimate)
                    end_step_time = time.time()
                    total_time += float(end_step_time - start_step_time)

                    step_type = step_result['step_type']
                    if step_type == 'adaptive':
                        lipschitz_estimate = step_result['l_estimate']
                elif FLAGS.fw_variant == 'ada_pfw':
                    start_step_time = time.time()
                    step_result = opt.adaptive_pfw(
                        weights, qUVt_components, qUV_prev, loc_s, scale_s,
                        sUV, p_joint, data, t, lipschitz_estimate)
                    end_step_time = time.time()
                    total_time += float(end_step_time - start_step_time)

                    step_type = step_result['step_type']
                    if step_type in ['adaptive', 'drop']:
                        lipschitz_estimate = step_result['l_estimate']
                elif FLAGS.fw_variant == 'ada_afw':
                    start_step_time = time.time()
                    step_result = opt.adaptive_pfw(
                        weights, qUVt_components, qUV_prev, loc_s, scale_s,
                        sUV, p_joint, data, t, lipschitz_estimate)
                    end_step_time = time.time()
                    total_time += float(end_step_time - start_step_time)

                    step_type = step_result['step_type']
                    if step_type in ['adaptive', 'away', 'drop']:
                        lipschitz_estimate = step_result['l_estimate']

                if t == 0:
                    gamma = 1.
                    weights.append(gamma)
                    qUVt_components.append({'loc': loc_s, 'scale': scale_s})
                    new_components = [sUV_mv]
                else:
                    qUVt_components = step_result['params']
                    weights = step_result['weights']
                    gamma = step_result['gamma']
                    new_components = [
                        coreutils.base_loc_scale(
                            'mvn0',
                            c['loc'],
                            c['scale'],
                            multivariate=False) for c in qUVt_components
                    ]

                qUV_new = coreutils.get_mixture(weights, new_components)

                #qR = Normal(
                #    loc=tf.matmul(
                #        tf.transpose(qUV_new[:, :N]), qUV_new[:, N:]),
                #    scale=tf.ones([N, M]))
                qR = ed.copy(R, {UV: qUV_new})
                cR = ed.copy(R_mask, {UV: qUV_new}) # reconstructed matrix

                # Log metrics for current iteration
                logger.info('total time %f' % total_time)
                append_to_file(times_filename, total_time)

                logger.info('iter %d, gamma %.4f' % (t, gamma))
                append_to_file(step_filename, gamma)

                if t > 0:
                    gap_t = step_result['gap']
                    logger.info('iter %d, gap %.4f' % (t, gap_t))
                    append_to_file(gap_filename, gap_t)


                # CRITICISM
                if FLAGS.fw_variant.startswith('ada'):
                    append_to_file(lipschitz_filename, lipschitz_estimate)
                    append_to_file(iter_info_filename, step_type)
                    logger.info('lt = %.5f, iter_type = %s' %
                                (lipschitz_estimate, step_type))

                test_mse = ed.evaluate(
                    'mean_squared_error',
                    data={
                        cR: R_true,
                        I: I_test
                    })
                logger.info("iter %d ed test mse %.5f" % (t, test_mse))
                append_to_file(mse_test_filename, test_mse)

                train_mse = ed.evaluate(
                    'mean_squared_error',
                    data={
                        cR: R_true,
                        I: I_train
                    })
                logger.info("iter %d ed train mse %.5f" % (t, train_mse))
                append_to_file(mse_train_filename, train_mse)

                # very slow
                #train_ll = log_likelihood(qUV_new, R_true, I_train, sess, D, N,
                #                          M)
                train_ll = ed.evaluate('log_lik', data={qR: R_true.astype(np.float32), I: I_train})
                logger.info("iter %d train log lik %.5f" % (t, train_ll))
                append_to_file(ll_train_filename, train_ll)

                #test_ll = log_likelihood(qUV_new, R_true, I_test, sess, D, N, M)
                test_ll = ed.evaluate('log_lik', data={qR: R_true.astype(np.float32), I: I_test})
                logger.info("iter %d test log lik %.5f" % (t, test_ll))
                append_to_file(ll_test_filename, test_ll)

                # elbo_loss might be meaningless
                elbo_loss = elboModel.KLqp({UV: qUV_new}, data={R: R_true, I: I_train})
                elbo_t = elbo(qUV_new, p_joint)
                res_update = elbo_loss.run()
                logger.info('iter %d -elbo loss %.2f or %.2f' % (t, res_update['loss'], elbo_t))
                append_to_file(elbos_filename, "%f,%f" % (elbo_t, res_update['loss']))

                # serialize the current iterate
                np.savez(os.path.join(outdir, 'qt_latest.npz'), weights=weights,
                        comps=qUVt_components, fw_iter=t+1)

                sess.close()
        tf.reset_default_graph()


if __name__ == "__main__":
    tf.app.run(main)
