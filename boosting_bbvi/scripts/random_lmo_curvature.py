"""Run random LMO experiment on synthetic data.

Example usage:

python scripts/toy_random_lmo.py \
        --relbo_reg 1.0 \
        --relbo_anneal linear \
        --fw_variant fixed \
        --dimension 2 \
        --outdir=${TD}/2d \
        --n_fw_iter=10 \
        --LMO_iter=20 \
        --seed 2019 \
        --norestore
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import edward as ed
from tensorflow.contrib.distributions import kl_divergence
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture, Exponential, VectorExponentialDiag,
                           VectorLaplaceDiag)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.utils import eprint, debug
import boosting_bbvi.core.relbo as relbo
import boosting_bbvi.core.utils as coreutils
from boosting_bbvi.core.utils import eprint, debug, softplus
from boosting_bbvi.optim.utils import elbo
import boosting_bbvi.optim.fw_step_size_curvature as opt
from boosting_bbvi.core.utils import eprint, debug, append_to_file
logger = coreutils.get_logger()


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp',
                    'directory to store all the results etc.')
flags.DEFINE_integer('seed', 0, 'The random seed to use for everything.')
flags.DEFINE_integer('dimension', 1,
                    'number of dimensions to run the experiment for')
flags.DEFINE_integer('n_target_components', 2,
                     'number of components in the target distribution')
flags.DEFINE_integer('n_fw_iter', 100, '')
flags.DEFINE_enum(
    'fw_variant', 'fixed',
    ['fixed', 'line_search', 'adafw'],
    '[fixed (default), line_search, adafw] The Frank-Wolfe variant to use.')
flags.DEFINE_enum('LMO', 'relbo', ['relbo', 'random'],
                  'get lmo from relbo or random')
flags.DEFINE_enum('init', 'good', ['good', 'random'],
                  'initialization good or random')
flags.DEFINE_integer('LMO_iter', 1000,
                     'number of LMO iterations, not used if random lmo')
flags.DEFINE_enum('dist', 'normal', ['normal', 'laplace'],
                  'target/base distribution')

ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)


D, K = FLAGS.dimension, FLAGS.n_target_components


def create_target_dist():
    """Create and return target distribution."""
    if FLAGS.dist != 'normal':
        raise NotImplementedError

    pi = np.random.dirichlet([1.] * K)
    #pi = pi[np.newaxis, :].astype(np.float32)

    #mus = 2.*np.random.rand(K, D).astype(np.float32) - 1.
    #stds = np.random.rand(K, D).astype(np.float32)

    mus = np.random.randn(K, D).astype(np.float32)
    stds = softplus(np.random.randn(K, D).astype(np.float32))

    pcomps = [
        MultivariateNormalDiag(
            loc=tf.convert_to_tensor(mus[i], dtype=tf.float32),
            scale_diag=tf.convert_to_tensor(stds[i], dtype=tf.float32))
        for i in range(K)
    ]
    p = Mixture(
        cat=Categorical(probs=tf.convert_to_tensor(pi, dtype=tf.float32)),
        components=pcomps)
    #q = VectorLaplaceDiag(loc=mus[0], scale_diag=stds[0])
    return p, mus, stds


def estimate_global_curvature(comps, qt):

    def f(gamma):
        weights = [(1 - gamma), gamma]
        q_l = Mixture(
            cat=Categorical(probs=tf.convert_to_tensor(weights)),
            components=[MultivariateNormalDiag(**c) for c in comps])
        return kl_divergence(q_l, qt).eval()

    def test():
        assert abs(f(0.) - 0.) < 1e-6

    test()
    DELTA = 0.05
    return max([(2. * f(gamma)) / (gamma * gamma)
                for gamma in np.arange(0. + DELTA, 1., DELTA)])


def main(argv):
    del argv

    outdir = FLAGS.outdir
    if '~' in outdir: outdir = os.path.expanduser(outdir)
    os.makedirs(outdir, exist_ok=True)

    # Files to log metrics
    times_filename = os.path.join(outdir, 'times.csv')
    elbos_filename = os.path.join(outdir, 'elbos.csv')
    objective_filename = os.path.join(outdir, 'kl.csv')
    reference_filename = os.path.join(outdir, 'ref_kl.csv')
    step_filename = os.path.join(outdir, 'steps.csv')
    # 'adafw', 'ada_afw', 'ada_pfw'
    if FLAGS.fw_variant.startswith('ada'):
        curvature_filename = os.path.join(outdir, 'curvature.csv')
        gap_filename = os.path.join(outdir, 'gap.csv')
        iter_info_filename = os.path.join(outdir, 'iter_info.txt')
    elif FLAGS.fw_variant == 'line_search':
        goutdir = os.path.join(outdir, 'gradients')

    # empty the files present in the folder already
    open(times_filename, 'w').close()
    open(elbos_filename, 'w').close()
    open(objective_filename, 'w').close()
    open(reference_filename, 'w').close()
    open(step_filename, 'w').close()
    # 'adafw', 'ada_afw', 'ada_pfw'
    if FLAGS.fw_variant.startswith('ada'):
        open(curvature_filename, 'w').close()
        append_to_file(curvature_filename, "c_local,c_global")
        open(gap_filename, 'w').close()
        open(iter_info_filename, 'w').close()
    elif FLAGS.fw_variant == 'line_search':
        os.makedirs(goutdir, exist_ok=True)

    for i in range(FLAGS.n_fw_iter):
        # NOTE: First iteration (t = 0) is initialization
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(FLAGS.seed)
            sess = tf.InteractiveSession()
            with sess.as_default():
                p, mus, stds = create_target_dist()

                # current iterate (solution until now)
                if FLAGS.init == 'random':
                    muq = np.random.randn(D).astype(np.float32)
                    stdq = softplus(np.random.randn(D).astype(np.float32))
                    raise ValueError
                else:
                    muq = mus[0]
                    stdq = stds[0]

                # 1 correct LMO
                t = 1
                comps = [{'loc': muq, 'scale_diag': stdq}]
                weights = [1.0]
                curvature_estimate = opt.adafw_linit()

                qtx = MultivariateNormalDiag(
                    loc=tf.convert_to_tensor(muq, dtype=tf.float32),
                    scale_diag=tf.convert_to_tensor(stdq, dtype=tf.float32))
                fw_iterates = {p: qtx}

                # calculate kl-div with 1 component
                objective_old = kl_divergence(qtx, p).eval()
                logger.info("kl with init %.4f" % (objective_old))
                append_to_file(reference_filename, objective_old)

                # s is the solution to LMO. It is initialized randomly
                # mu ~ N(0, 1), std ~ softplus(N(0, 1))
                s = coreutils.construct_multivariatenormaldiag([D], t, 's')

                sess.run(tf.global_variables_initializer())

                total_time = 0
                start_inference_time = time.time()
                if FLAGS.LMO == 'vi':
                    # we have to iterate over parameter space
                    raise ValueError
                    inference = relbo.KLqp(
                        {
                            p: s
                        }, fw_iterates=fw_iterates, fw_iter=t)
                    inference.run(n_iter=FLAGS.LMO_iter)
                # s now contains solution to LMO
                end_inference_time = time.time()

                mu_s = s.mean().eval()
                cov_s = s.stddev().eval()

                # NOTE: keep only step size time
                #total_time += end_inference_time - start_inference_time

                # compute step size to update the next iterate
                step_result = {}
                if FLAGS.fw_variant == 'fixed':
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
                elif FLAGS.fw_variant == 'adafw':
                    start_adafw_time = time.time()
                    step_result = opt.adaptive_fw(
                        weights, [c['loc'] for c in comps],
                        [c['scale_diag'] for c in comps], qtx, mu_s, cov_s,
                        s, p, t, curvature_estimate)
                    end_adafw_time = time.time()
                    total_time += end_adafw_time - start_adafw_time
                    gamma = step_result['gamma']
                else:
                    raise NotImplementedError

                comps.append({'loc': mu_s, 'scale_diag': cov_s})
                weights = [(1. - gamma), gamma]

                c_global = estimate_global_curvature(comps, qtx)

                q_latest = Mixture(
                    cat=Categorical(probs=tf.convert_to_tensor(weights)),
                    components=[
                        MultivariateNormalDiag(**c) for c in comps
                    ])

                # Log metrics for current iteration
                time_t = float(total_time)
                logger.info('total time %f' % (time_t))
                append_to_file(times_filename, time_t)

                elbo_t = elbo(q_latest, p, n_samples=1000)
                logger.info("iter, %d, elbo, %.2f +/- %.2f" %
                            (t, elbo_t[0], elbo_t[1]))
                append_to_file(elbos_filename,
                                "%f,%f" % (elbo_t[0], elbo_t[1]))

                logger.info('iter %d, gamma %.4f' % (t, gamma))
                append_to_file(step_filename, gamma)

                objective_t = kl_divergence(q_latest, p).eval()
                logger.info("run %d, kl %.4f" % (i, objective_t))
                append_to_file(objective_filename, objective_t)

                if FLAGS.fw_variant.startswith('ada'):
                    curvature_estimate = step_result['c_estimate']
                    append_to_file(gap_filename, step_result['gap'])
                    append_to_file(iter_info_filename, step_result['step_type'])
                    logger.info(
                        'gap = %.3f, ct = %.5f, iter_type = %s' %
                        (step_result['gap'], step_result['c_estimate'],
                        step_result['step_type']))
                    append_to_file(curvature_filename, '%f,%f' % (curvature_estimate, c_global))
                elif FLAGS.fw_variant == 'line_search':
                    n_line_search_samples = step_result['n_samples']
                    grad_t = step_result['grad_gamma']
                    g_outfile = os.path.join(
                        goutdir, 'line_search_samples_%d.npy.%d' %
                        (n_line_search_samples, t))
                    logger.info(
                        'saving line search data to, %s' % g_outfile)
                    np.save(open(g_outfile, 'wb'), grad_t)

            sess.close()

        tf.reset_default_graph()



if __name__ == "__main__":
    tf.app.run()
