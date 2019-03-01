"""Frank-Wolfe Optimizer for Boosted Black Box Variational Inference.

Logs  elbos, relbos, objectives, times, iteration wide info and other
relevant metrics.
"""
# NOTE: Currently the code assumed the target distribution is given and the
# objective function is kl divergence.
# TODO: For using on real world datasets, joint model p(z, x) will be given
# and target will be p(z | x). Optimization objective would be ELBO (with
# corresponding gradient) and not KL divergence

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import kl_divergence
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture)
import edward as ed

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import boosting_bbvi.core.relbo as relbo
import boosting_bbvi.core.utils as coreutils
from boosting_bbvi.optim.utils import elbo
import boosting_bbvi.optim.fw_step_size as opt
from boosting_bbvi.core.utils import eprint, debug, append_to_file
logger = coreutils.get_logger()


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n_fw_iter', 100, '')
flags.DEFINE_integer('LMO_iter', 1000, '')
flags.DEFINE_enum(
    'fw_variant', 'fixed',
    ['fixed', 'line_search', 'fc', 'adafw', 'ada_afw', 'ada_pfw'],
    '[fixed (default), line_search, fc] The Frank-Wolfe variant to use.')
flags.DEFINE_enum('iter0', 'vi', ['vi', 'random'],
                  '1st component a random distribution or from vi')
# flags.DEFINE_string('decay', 'log',
# '[linear, log (default), squared] The decay rate to use for Lambda.')

class FWOptimizer(object):

    """Class for running Frank-Wolfe for Boosted BBVI. """

    def __init__(self):
        """Construct a FWOptimizer object """


    def run(self, outdir, pi, mus, stds, n_features):
        """Run Boosted BBVI.

        Args:
            outdir: output directory
            pi: weights of target mixture
            mus: means of target mixture
            stds: scale of target mixture
            n_features: dimensionality

        Returns:
            runs FLAGS.n_fw_iter of frank-wolfe and logs
            relevant metrics
        """

        # comps: component atoms of boosting (contains a dict of params)
        # weights: weights given to every atom over comps
        # Together S = {weights, comps} make the active set
        weights, comps = [], []
        # L-continuous gradient estimate
        lipschitz_estimate = None

        # Metrics to log
        times_filename = os.path.join(outdir, 'times.csv')
        open(times_filename, 'w').close() # truncate the file if exists

        elbos_filename = os.path.join(outdir, 'elbos.csv')
        logger.info("saving elbos to, %s" % elbos_filename)
        open(elbos_filename, 'w').close()

        relbos_filename = os.path.join(outdir, 'relbos.csv')
        logger.info('saving relbos to, %s' % relbos_filename)
        open(relbos_filename, 'w').close()

        objective_filename = os.path.join(outdir, 'kl.csv')
        logger.info("saving kl divergence to, %s" % objective_filename)
        open(objective_filename, 'w').close()

        step_filename = os.path.join(outdir, 'steps.csv')
        logger.info("saving gamma values to, %s" % step_filename)
        open(step_filename, 'w').close()

        # 'adafw', 'ada_afw', 'ada_pfw'
        if FLAGS.fw_variant.startswith('ada'):
            lipschitz_filename = os.path.join(outdir, 'lipschitz.csv')
            open(lipschitz_filename, 'w').close()

            gap_filename = os.path.join(outdir, 'gap.csv')
            open(gap_filename, 'w').close()

            iter_info_filename = os.path.join(outdir, 'iter_info.txt')
            open(iter_info_filename, 'w').close()
        elif FLAGS.fw_variant == 'line_search':
            goutdir = os.path.join(outdir, 'gradients')
            os.makedirs(goutdir, exist_ok=True)

        for t in range(FLAGS.n_fw_iter):
            # NOTE: First iteration (t = 0) is initialization
            g = tf.Graph()
            with g.as_default():
                tf.set_random_seed(FLAGS.seed)
                sess = tf.InteractiveSession()
                with sess.as_default():
                    # build target distribution
                    pcomps = [
                        MultivariateNormalDiag(
                            loc=tf.convert_to_tensor(mus[i], dtype=tf.float32),
                            scale_diag=tf.convert_to_tensor(
                                stds[i], dtype=tf.float32))
                        for i in range(len(mus))
                    ]
                    p = Mixture(
                        cat=Categorical(probs=tf.convert_to_tensor(pi[0])),
                        components=pcomps)

                    if t == 0:
                        fw_iterates = {}
                    else:
                        # current iterate (solution until now)
                        qtx = Mixture(
                            cat=Categorical(
                                probs=tf.convert_to_tensor(weights)),
                            components=[
                                MultivariateNormalDiag(**c) for c in comps
                            ])
                        fw_iterates = {p: qtx}

                    # s is the solution to LMO. It is initialized randomly
                    s = coreutils.construct_normal([n_features], t, 's')

                    sess.run(tf.global_variables_initializer())

                    total_time = 0
                    start_inference_time = time.time()
                    # Run inference on relbo to solve LMO problem
                    # If initilization of mixture is random, then the
                    # first component will be random distribution, in
                    # that case no inference is needed.
                    # NOTE: KLqp has a side effect, it is modifying s
                    if FLAGS.iter0 == 'vi' or t > 0:
                        inference = relbo.KLqp(
                            {
                                p: s
                            }, fw_iterates=fw_iterates, fw_iter=t)
                        inference.run(n_iter=FLAGS.LMO_iter)
                    # s now contains solution to LMO
                    """
                    ..takes me one step closer to the edge...
                    """
                    end_inference_time = time.time()

                    mu_s = s.loc.eval() # s.mean().eval()
                    cov_s = s.scale.eval() # s.stddev().eval()

                    total_time += end_inference_time - start_inference_time

                    # compute step size to update the next iterate
                    step_result = {}
                    """
                    Step by step, heart to heart, left right left...
                    """
                    if t == 0:
                        gamma = 1.
                        if FLAGS.fw_variant.startswith('ada'):
                            lipschitz_estimate = opt.adafw_linit(s, p)
                    elif FLAGS.fw_variant == 'fixed':
                        gamma = 2. / (t + 2.)
                    elif FLAGS.fw_variant == 'line_search':
                        start_line_search_time = time.time()
                        step_result = opt.line_search_dkl(
                            weights, [c['loc'] for c in comps],
                            [c['scale_diag'] for c in comps], qtx, mu_s, cov_s,
                            s, p, t)
                        end_line_search_time = time.time()
                        total_time += (
                            end_line_search_time - start_line_search_time)
                        gamma = step_result['gamma']
                    elif FLAGS.fw_variant == 'fc':
                        # Add a fixed component. Correct later
                        gamma = 2. / (t + 2.)
                    elif FLAGS.fw_variant == 'adafw':
                        start_adafw_time = time.time()
                        step_result = opt.adaptive_fw(
                            weights, [c['loc'] for c in comps],
                            [c['scale_diag'] for c in comps], qtx, mu_s, cov_s,
                            s, p, t, lipschitz_estimate)
                        end_adafw_time = time.time()
                        total_time += end_adafw_time - start_adafw_time
                        gamma = step_result['gamma']
                    elif FLAGS.fw_variant == 'ada_afw':
                        start_adaafw_time = time.time()
                        step_result = opt.adaptive_afw(
                            weights, comps, [c['loc'] for c in comps],
                            [c['scale_diag'] for c in comps], qtx, mu_s, cov_s,
                            s, p, t, lipschitz_estimate)
                        end_adaafw_time = time.time()
                        total_time += end_adaafw_time - start_adaafw_time
                        gamma = step_result['gamma'] # just for logging
                    elif FLAGS.fw_variant == 'ada_pfw':
                        start_adapfw_time = time.time()
                        step_result = opt.adaptive_pfw(
                            weights, comps, [c['loc'] for c in comps],
                            [c['scale_diag'] for c in comps], qtx, mu_s, cov_s,
                            s, p, t, lipschitz_estimate)
                        end_adapfw_time = time.time()
                        total_time += end_adapfw_time - start_adapfw_time
                        gamma = step_result['gamma'] # just for logging

                    # TODO(sauravshekhar): In more complex algorithms like
                    # Away-Steps and fully corrective, it is not necessary
                    # that a new component will always be added. So instead of
                    # getting gamma, pass comps and weights to the step
                    # size selection methods so they can change them
                    if ((FLAGS.fw_variant == 'ada_afw'
                         or FLAGS.fw_variant == 'ada_pfw') and t > 0):
                        comps = step_result['comps']
                        weights = step_result['weights']
                    else:
                        comps.append({'loc': mu_s, 'scale_diag': cov_s})
                        weights = coreutils.update_weights(weights, gamma, t)

                    # TODO: Move this to fw_step_size.py
                    if FLAGS.fw_variant == "fc":
                        q_latest = Mixture(
                            cat=Categorical(
                                probs=tf.convert_to_tensor(weights)),
                            components=[
                                MultivariateNormalDiag(**c) for c in comps
                            ])
                        # Correction
                        start_fc_time = time.time()
                        weights = opt.fully_corrective(q_latest, p)
                        weights = list(weights)
                        for i in reversed(range(len(weights))):
                            # Remove components whose weight is 0
                            w = weights[i]
                            if w == 0:
                                del weights[i]
                                del comps[i]
                        weights = np.array(weights)
                        end_fc_time = time.time()
                        total_time += end_fc_time - start_fc_time

                    q_latest = Mixture(
                        cat=Categorical(probs=tf.convert_to_tensor(weights)),
                        components=[
                            MultivariateNormalDiag(**c) for c in comps
                        ])

                    # Log metrics for current iteration
                    time_t = float(total_time)
                    logger.info('total time %f' % (time_t))
                    append_to_file(times_filename, time_t)

                    elbo_t = elbo(q_latest, p)
                    logger.info("iter, %d, elbo, %.2f +/- %.2f" %
                                (t, elbo_t[0], elbo_t[1]))
                    append_to_file(elbos_filename,
                                   "%f,%f" % (elbo_t[0], elbo_t[1]))

                    logger.info('iter %d, gamma %.4f' % (t, gamma))
                    append_to_file(step_filename, gamma)

                    if t > 0:
                        relbo_t = -coreutils.compute_relbo(
                            s, fw_iterates[p], p, np.log(t + 1))
                        append_to_file(relbos_filename, relbo_t)

                    objective_t = kl_divergence(q_latest, p).eval()
                    logger.info("iter, %d, kl, %.2f" % (t, objective_t))
                    append_to_file(objective_filename, objective_t)

                    if FLAGS.fw_variant.startswith('ada'):
                        if t > 0:
                            lipschitz_estimate = step_result['l_estimate']
                            append_to_file(gap_filename, step_result['gap'])
                            append_to_file(iter_info_filename,
                                        step_result['step_type'])
                            logger.info(
                                'gap = %.3f, lt = %.5f, iter_type = %s' %
                                (step_result['gap'], step_result['l_estimate'],
                                step_result['step_type']))
                        # l_estimate for iter 0 is the intial value
                        append_to_file(lipschitz_filename, lipschitz_estimate)
                    elif FLAGS.fw_variant == 'line_search' and t > 0:
                        n_line_search_samples = step_result['n_samples']
                        grad_t = step_result['grad_gamma']
                        g_outfile = os.path.join(
                            goutdir, 'line_search_samples_%d.npy.%d' %
                            (n_line_search_samples, t))
                        logger.info(
                            'saving line search data to, %s' % g_outfile)
                        np.save(open(g_outfile, 'wb'), grad_t)

                    for_serialization = {
                        'locs': np.array([c['loc'] for c in comps]),
                        'scale_diags':
                        np.array([c['scale_diag'] for c in comps])
                    }
                    qt_outfile = os.path.join(outdir, 'qt_iter%d.npz' % t)
                    np.savez(qt_outfile, weights=weights, **for_serialization)
                    np.savez(
                        os.path.join(outdir, 'qt_latest.npz'),
                        weights=weights,
                        **for_serialization)
                    logger.info("saving qt to, %s" % qt_outfile)
            tf.reset_default_graph()
