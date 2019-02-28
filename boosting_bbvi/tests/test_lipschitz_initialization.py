"""Smoke test for lipschitz initialization.

Usage:
    python test_lipschitz_initialization \
            --linit lipschitz_v2 \
            --linit_fixed 0.001 \
            --adafw_MAXITER 6 \
            --damping_adafw 0.2 \
            --exp_adafw 2.0
"""

import os
import sys
import numpy as np
import math
import time
import tensorflow as tf
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture)
import edward as ed
from tensorflow.contrib.distributions import kl_divergence
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import boosting_bbvi.core.utils as coreutils
import boosting_bbvi.optim.fw_step_size as opt
import boosting_bbvi.scripts.mixture_model_relbo as mm
from boosting_bbvi.core.utils import eprint, debug, construct_normal
logger = coreutils.get_logger()


flags = tf.app.flags
FLAGS = flags.FLAGS

def test_lipschitz_init(pi, mus, stds):
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(FLAGS.seed)
        sess = tf.InteractiveSession()
        with sess.as_default():
            s = construct_normal([1], 0, 's')
            sess.run(tf.global_variables_initializer())
            logger.info('mean of s = %.3f, std = %.3f' % (s.mean().eval(),
                                                          s.stddev().eval()))
            # build target distribution
            pcomps = [
                MultivariateNormalDiag(
                    loc=tf.convert_to_tensor(mus[i], dtype=tf.float32),
                    scale_diag=tf.convert_to_tensor(
                        stds[i], dtype=tf.float32))
                for i in range(len(mus))
            ]
            p = Mixture(
                cat=Categorical(probs=tf.convert_to_tensor(pi)),
                components=pcomps)
            lipschitz_init_estimate = opt.adafw_linit(s, p)
            logger.info('L estimate is %.5f' % lipschitz_init_estimate)

def main(argv):
    pi = np.array([0.4, 0.6]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.6], [.6]]
    for linit_fixed in [0.001, 0.01, 0.1, 1.0, 10.0]:
        FLAGS.linit_fixed = linit_fixed
        test_lipschitz_init(pi, mus, stds)

if __name__ == "__main__":
    tf.app.run()
