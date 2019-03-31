"""Run Bayesian Linear Regression.

Example usage:

"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import edward as ed
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.distributions import kl_divergence
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture, Bernoulli)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import boosting_bbvi.core.relbo as relbo
import boosting_bbvi.core.utils as coreutils
import boosting_bbvi.optim.blr_step_size as opt
from boosting_bbvi.optim.utils import elbo
import boosting_bbvi.scripts.blr_utils as blr_utils
from boosting_bbvi.core.utils import eprint, debug, append_to_file
logger = coreutils.get_logger()


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp',
                    'directory to store all the results, models, plots, etc.')
flags.DEFINE_integer('seed', 0, 'The random seed to use for everything.')
tf.flags.DEFINE_enum(
    "base_dist", 'normal',
    ['normal', 'laplace', 'mvnormal', 'mvlaplace', 'mvn', 'mvl'],
    'base distribution for variational approximation')
flags.DEFINE_integer('n_fw_iter', 100, '')
flags.DEFINE_integer('LMO_iter', 1000, '')
flags.DEFINE_enum(
    'fw_variant', 'fixed',
    ['fixed', 'line_search', 'fc', 'adafw', 'ada_afw', 'ada_pfw'],
    '[fixed (default), line_search, fc] The Frank-Wolfe variant to use.')
flags.DEFINE_enum('iter0', 'vi', ['vi', 'random'],
                  '1st component a random distribution or from vi')
ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)


def main(_):
    ed.set_seed(FLAGS.seed)
    # setting up output directory
    outdir = FLAGS.outdir
    if '~' in outdir: outdir = os.path.expanduser(outdir)
    os.makedirs(outdir, exist_ok=True)

    is_vector = FLAGS.base_dist in ['mvnormal', 'mvlaplace']

    ((Xtrain, ytrain), (Xtest, ytest)) = blr_utils.get_data()
    N,D = Xtrain.shape
    N_test,D_test = Xtest.shape
    assert D_test == D, 'Test dimension %d different than train %d' % (D_test,
                                                                       D)
    logger.info('D = %d, Ntrain = %d, Ntest = %d' % (D, N, N_test))

    # Solution components
    weights, q_params = [], []
    # L-continous gradient estimate
    lipschitz_estimate = None

    # Metrics to log
    times_filename = os.path.join(outdir, 'times.csv')
    open(times_filename, 'w').close()

    # (mean, +- std)
    elbos_filename = os.path.join(outdir, 'elbos.csv')
    logger.info('saving elbos to, %s' % elbos_filename)
    open(elbos_filename, 'w').close()

    rocs_filename = os.path.join(outdir, 'roc.csv')
    logger.info('saving rocs to, %s' % rocs_filename)
    open(rocs_filename, 'w').close()

    gap_filename = os.path.join(outdir, 'gap.csv')
    open(gap_filename, 'w').close()

    step_filename = os.path.join(outdir, 'steps.csv')
    open(step_filename, 'w').close()

    # (mean, std)
    ll_train_filename = os.path.join(outdir, 'll_train.csv')
    open(ll_train_filename, 'w').close()
    ll_test_filename = os.path.join(outdir, 'll_test.csv')
    open(ll_test_filename, 'w').close()

    # (bin_ac_train, bin_ac_test)
    bin_ac_filename = os.path.join(outdir, 'bin_ac.csv')
    open(bin_ac_filename, 'w').close()

    # 'adafw', 'ada_afw', 'ada_pfw'
    if FLAGS.fw_variant.startswith('ada'):
        lipschitz_filename = os.path.join(outdir, 'lipschitz.csv')
        open(lipschitz_filename, 'w').close()

        iter_info_filename = os.path.join(outdir, 'iter_info.txt')
        open(iter_info_filename, 'w').close()

    for t in range(FLAGS.n_fw_iter):
        g = tf.Graph()
        with g.as_default():
            sess = tf.InteractiveSession()
            with sess.as_default():
                tf.set_random_seed(FLAGS.seed)

                # Build Model
                w = Normal(
                    loc=tf.zeros(D, tf.float32), scale=tf.ones(D, tf.float32))

                X = tf.placeholder(tf.float32, [None, D])
                y = Bernoulli(logits=ed.dot(X, w))

                p_joint = blr_utils.Joint(Xtrain, ytrain, sess,
                                          FLAGS.n_monte_carlo_samples, logger)

                # vectorized Model evaluations
                n_test_samples = 100
                W = tf.placeholder(tf.float32, [n_test_samples, D])
                y_data = tf.placeholder(tf.float32, [None]) # N -> (N, n_test)
                y_data_matrix = tf.tile(
                    tf.expand_dims(y_data, 1), (1, n_test_samples))
                pred_logits = tf.matmul(X, tf.transpose(W)) # (N, n_test)
                # FIXME(sauravshekhar) first sigmoid then mean
                # or first mean then sigmoid, prev implementation is
                # mean logit -> sigmoid
                ypred = tf.sigmoid(tf.reduce_mean(pred_logits, axis=1))
                pY = Bernoulli(logits=pred_logits) # (N, n_test)
                log_likelihoods = pY.log_prob(y_data_matrix) # (N, n_test)
                log_likelihood_expectation = tf.reduce_mean(
                    log_likelihoods, axis=1) # (N, )
                ll_mean, ll_std = tf.nn.moments(
                    log_likelihood_expectation, axes=[0])

                if t == 0:
                    fw_iterates = {}
                else:
                    # Current solution
                    prev_components = [
                        coreutils.base_loc_scale(
                            FLAGS.base_dist,
                            c['loc'],
                            c['scale'],
                            multivariate=is_vector) for c in q_params
                    ]
                    qtw_prev = coreutils.get_mixture(weights, prev_components)
                    fw_iterates = {w: qtw_prev}

                # s is the solution to LMO, random initialization
                s = coreutils.construct_base(
                    FLAGS.base_dist, [D], t, 's', multivariate=is_vector)

                sess.run(tf.global_variables_initializer())

                total_time = 0.
                inference_time_start = time.time()
                # Run relbo to solve LMO problem
                inference = relbo.KLqp({w: s},
                                       fw_iterates=fw_iterates,
                                       data={
                                           X: Xtrain,
                                           y: ytrain
                                       },
                                       fw_iter=t)
                inference.run(n_iter=FLAGS.LMO_iter)
                inference_time_end = time.time()
                total_time += float(inference_time_end - inference_time_start)

                loc_s = s.mean().eval()
                scale_s = s.stddev().eval()

                # Evaluate the next step
                step_result = {}
                if t == 0:
                    # Initialization, q_0
                    q_params.append({'loc': loc_s, 'scale': scale_s})
                    weights.append(1.)
                    if FLAGS.fw_variant.startswith('ada'):
                        lipschitz_estimate = opt.adafw_linit(s, p_joint)
                    step_type = 'init'
                elif FLAGS.fw_variant == 'fixed':
                    start_step_time = time.time()
                    step_result = opt.fixed(weights, q_params, qtw_prev,
                                            loc_s, scale_s, s, p_joint, t)
                    end_step_time = time.time()
                    total_time += float(end_step_time - start_step_time)
                elif FLAGS.fw_variant == 'adafw':
                    start_step_time = time.time()
                    step_result = opt.adaptive_fw(weights, q_params, qtw_prev,
                                                  loc_s, scale_s, s, p_joint,
                                                  t, lipschitz_estimate)
                    end_step_time = time.time()
                    total_time += float(end_step_time - start_step_time)
                    step_type = step_result['step_type']
                    if step_type == 'adaptive':
                        lipschitz_estimate = step_result['l_estimate']
                else:
                    raise NotImplementedError(
                        'Step size variant %s not implemented' %
                        FLAGS.fw_variant)

                if t == 0:
                    gamma = 1.
                    qtw_new = s
                else:
                    q_params = step_result['params']
                    weights = step_result['weights']
                    gamma = step_result['gamma']
                    new_components = [
                        coreutils.base_loc_scale(
                            FLAGS.base_dist,
                            c['loc'],
                            c['scale'],
                            multivariate=is_vector) for c in q_params
                    ]
                    qtw_new = coreutils.get_mixture(weights, new_components)

                # Log metrics for current iteration
                logger.info('total time %f' % total_time)
                append_to_file(times_filename, total_time)

                elbo_t = elbo(qtw_new, p_joint, return_std=False)
                logger.info("iter, %d, elbo, %.2f " % (t, elbo_t))
                append_to_file(elbos_filename, "%f" % (elbo_t))

                logger.info('iter %d, gamma %.4f' % (t, gamma))
                append_to_file(step_filename, gamma)

                if t > 0:
                    gap_t = step_result['gap']
                    logger.info('iter %d, gap %.4f' % (t, gap_t))
                    append_to_file(gap_filename, gap_t)

                if FLAGS.fw_variant.startswith('ada'):
                    append_to_file(lipschitz_filename, lipschitz_estimate)
                    append_to_file(iter_info_filename, step_type)
                    logger.info('lt = %.5f, iter_type = %s' %
                                (lipschitz_estimate, step_type))

                # get weight samples to evaluate expectations
                w_samples = qtw_new.sample([n_test_samples]).eval()
                ll_train_mean, ll_train_std = sess.run([ll_mean, ll_std],
                                                       feed_dict={
                                                           W: w_samples,
                                                           X: Xtrain,
                                                           y_data: ytrain
                                                       })
                logger.info("iter, %d, train ll, %.2f +/- %.2f" %
                            (t, ll_train_mean, ll_train_std))
                append_to_file(ll_train_filename,
                                "%f,%f" % (ll_train_mean, ll_train_std))

                ll_test_mean, ll_test_std, y_test_pred = sess.run(
                    [ll_mean, ll_std, ypred],
                    feed_dict={
                        W: w_samples,
                        X: Xtest,
                        y_data: ytest
                    })
                logger.info("iter, %d, test ll, %.2f +/- %.2f" %
                            (t, ll_test_mean, ll_test_std))
                append_to_file(ll_test_filename,
                                "%f,%f" % (ll_test_mean, ll_test_std))

                roc_score = roc_auc_score(ytest, y_test_pred)
                logger.info("iter %d, roc %.4f" % (t, roc_score))
                append_to_file(rocs_filename, roc_score)

                y_post = ed.copy(y, {w: qtw_new})
                # eq. to y = Bernoulli(logits=ed.dot(X, qtw_new))

                ed_train_ll = ed.evaluate(
                    'log_likelihood', data={
                        X: Xtrain,
                        y_post: ytrain,
                    })
                ed_test_ll = ed.evaluate(
                    'log_likelihood', data={
                        X: Xtest,
                        y_post: ytest,
                    })
                logger.info("edward train ll %.2f test ll %.2f" %
                            (ed_train_ll, ed_test_ll))

                bin_ac_train = ed.evaluate(
                    'binary_accuracy', data={
                        X: Xtrain,
                        y_post: ytrain,
                    })
                bin_ac_test = ed.evaluate(
                    'binary_accuracy', data={
                        X: Xtest,
                        y_post: ytest,
                    })
                append_to_file(bin_ac_filename,
                               "%f,%f" % (bin_ac_train, bin_ac_test))
                logger.info("edward binary accuracy train ll %.2f test ll %.2f"
                            % (bin_ac_train, bin_ac_test))

                mse_test = ed.evaluate(
                    'mean_squared_error', data={
                        X: Xtest,
                        y_post: ytest,
                    })
                logger.info("edward mse test ll %.2f" % (mse_test))
        tf.reset_default_graph()


if __name__ == "__main__":
    tf.app.run()
