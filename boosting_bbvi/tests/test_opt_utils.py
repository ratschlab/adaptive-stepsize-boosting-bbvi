"""Tests for optimization utilities.

Prints accuracy of the monte carlo KL divergence estimate for different
pairs of distrbutions and sample sizes.

Usage:
    python test_opt_utils.py
"""
import tensorflow as tf

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import boosting_bbvi.optim.fw_step_size as opt
import boosting_bbvi.optim.utils as opt_utils
from boosting_bbvi.tests.test_step_size import print_err
from edward.models import Normal
from tensorflow.python.ops.distributions import normal
import boosting_bbvi.core.utils as coreutils
logger = coreutils.get_logger()


def test_kl_monte_carlo():
    # Normal distribution
    mus = [0.1, 10.]
    stds = [2., 35.]
    n_samples = [100, 1000]

    with tf.Session() as sess:
        for mu_a in mus:
            for std_a in stds:
                a = Normal(loc=mu_a, scale=std_a)
                for mu_b in mus:
                    for std_b in stds:
                        b = Normal(loc=mu_b, scale=std_b)
                        true_kl = normal._kl_normal_normal(a, b).eval()
                        for n in n_samples:
                            comp_kl = opt_utils._kl_monte_carlo(a, b, n).eval()
                            logger.info('N(%.2f, %.2f) N(%.2f, %.2f) n = %d' %
                                        (mu_a, std_a, mu_b, std_b, n))
                            print_err(true_kl, comp_kl)

if __name__ == "__main__":
    test_kl_monte_carlo()
