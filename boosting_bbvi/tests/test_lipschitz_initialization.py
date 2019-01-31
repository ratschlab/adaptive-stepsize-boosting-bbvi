"""Smoke test for lipschitz initialization"""

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
import boosting_bbvi.core.utils as utils
import boosting_bbvi.core.opt as opt
from boosting_bbvi.core.utils import eprint, debug
from boosting_bbvi.scripts.mixture_model_relbo import construct_normal
logger = utils.get_logger()


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
            lipschitz_init_estimate = opt.adafw_linit(s, p, 'lipschitz_v2')
            logger.info('L estimate is %.5f' % lipschitz_init_estimate)

def main(argv):
    pi = np.array([0.4, 0.6]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.6], [.6]]
    test_lipschitz_init(pi, mus, stds)

if __name__ == "__main__":
    tf.app.run()
