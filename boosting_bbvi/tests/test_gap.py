"""Show duality gap for a simple run of FW.

Usage:
    python test_gap.py \
            --n_fw_iter 10 \
            --LMO_iter 1000 \
            --n_monte_carlo_samples 1000
"""

import os
import sys
import numpy as np
import time
import tensorflow as tf
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture)
import edward as ed
from tensorflow.contrib.distributions import kl_divergence
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import boosting_bbvi.core.relbo as relbo
import boosting_bbvi.core.utils as coreutils
from boosting_bbvi.optim.utils import elbo, setup_outdir, grad_kl
import boosting_bbvi.optim.fw_step_size as opt
import boosting_bbvi.optim.fw as optim
import boosting_bbvi.scripts.mixture_model_relbo as mm
from boosting_bbvi.core.utils import eprint, debug, construct_normal
logger = coreutils.get_logger()


flags = tf.app.flags
FLAGS = flags.FLAGS


def run_gap(pi, mus, stds):
    weights, comps = [], []
    elbos = []
    relbo_vals = []
    for t in range(FLAGS.n_fw_iter):
        logger.info('Frank Wolfe Iteration %d' % t)
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(FLAGS.seed)
            sess = tf.InteractiveSession()
            with sess.as_default():
                # target distribution components
                pcomps = [
                    MultivariateNormalDiag(
                        loc=tf.convert_to_tensor(mus[i], dtype=tf.float32),
                        scale_diag=tf.convert_to_tensor(
                            stds[i], dtype=tf.float32))
                    for i in range(len(mus))
                ]
                # target distribution
                p = Mixture(
                    cat=Categorical(probs=tf.convert_to_tensor(pi)),
                    components=pcomps)

                # LMO appoximation
                s = construct_normal([1], t, 's')
                fw_iterates = {}
                if t > 0:
                    qtx = Mixture(
                        cat=Categorical(probs=tf.convert_to_tensor(weights)),
                        components=[
                            MultivariateNormalDiag(**c) for c in comps
                        ])
                    fw_iterates = {p: qtx}
                sess.run(tf.global_variables_initializer())
                # Run inference on relbo to solve LMO problem
                # NOTE: KLqp has a side effect, it is modifying s
                inference = relbo.KLqp(
                    {
                        p: s
                    }, fw_iterates=fw_iterates, fw_iter=t)
                inference.run(n_iter=FLAGS.LMO_iter)
                # s now contains solution to LMO

                if t > 0:
                    sample_s = s.sample([FLAGS.n_monte_carlo_samples])
                    sample_q = qtx.sample([FLAGS.n_monte_carlo_samples])
                    step_s = tf.reduce_mean(grad_kl(qtx, p, sample_s)).eval()
                    step_q = tf.reduce_mean(grad_kl(qtx, p, sample_q)).eval()
                    gap = step_q - step_s
                    logger.info('Frank-Wolfe gap at iter %d is %.5f' % (t, gap))
                    if gap < 0:
                        eprint('Frank-Wolfe gab becoming negative!')
                    # f(q*) = f(p) = 0
                    logger.info('Objective value (actual gap) is %.5f' %
                                kl_divergence(qtx, p).eval())

                gamma = 2. / (t + 2.)
                comps.append({
                    'loc': s.mean().eval(),
                    'scale_diag': s.stddev().eval()
                })
                weights = coreutils.update_weights(weights, gamma, t)

        tf.reset_default_graph()


def main(argv):
    # target distribution parameters
    pi = np.array([0.4, 0.6]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.6], [.6]]
    run_gap(pi, mus, stds)

if __name__ == "__main__":
    tf.app.run()
