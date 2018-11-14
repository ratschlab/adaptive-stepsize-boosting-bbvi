"""Test line search variant of FW.

Usage:
    python test_line_search.py \
            --exp mixture \
            --n_line_search_samples 10 \
            --n_line_search_iter 10 \
            --outdir=out_path/test \
            --init_k 0 \
"""
from edward.models import Categorical, Normal, Mixture, MultivariateNormalDiag
import tensorflow as tf
import scipy.stats
import random
import numpy as np

from colorama import Fore, Style
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import boosting_bbvi.scripts.mixture_model_relbo as mixture_model_relbo

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('init_k', 0., 'Used to Initialize (simulating fw iteration)')

def print_err(true, calculated):
    if isinstance(true, np.ndarray):
        err = np.sum((true - calculated) ** 2)
        norm_const = np.sum(true**2) + 1e-16
    else:
        err = np.abs(true - calculated)
        norm_const = true + 1e-16
    rerr = err / norm_const
    if rerr < 0.01:
        etype = Fore.GREEN
    elif rerr < 0.10:
        etype = Fore.CYAN
    elif rerr < 0.25:
        etype = Fore.YELLOW
    else:
        etype = Fore.RED
    print(etype)
    print('true  %.5f, calculated  %.5f, Error %.5f' % (true, calculated,
                                                           rerr))
    print(Style.RESET_ALL)


def test_exact_gamma():
    pi = mixture_model_relbo.pi
    mus = mixture_model_relbo.mus
    stds = mixture_model_relbo.stds
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(FLAGS.seed)
        sess = tf.InteractiveSession()
        with sess.as_default():
            # Build p = pi[0] * N(mu[0], std[0]) + pi[1] * N(mu[1], std[1])
            # thus, gamma = pi[1] (=0.6), q_t = N(mu[0], std[0])
            # s = N(mu[1], std[1])
            pcomps = [
                Normal(
                    loc=tf.convert_to_tensor(mus[i], dtype=tf.float32),
                    scale=tf.convert_to_tensor(stds[i], dtype=tf.float32))
                for i in range(len(mus))
            ]
            p = Mixture(
                cat=Categorical(probs=tf.convert_to_tensor(pi)),
                components=pcomps)
            # build q_t
            weights = [1.]
            locs = [mus[0]]
            diags = [stds[0]]
            gamma = mixture_model_relbo.line_search_dkl(
                weights, locs, diags, mus[1], stds[1], p, FLAGS.init_k)
    print_err(pi[0][1], gamma)

def main(argv):
    test_exact_gamma()


if __name__ == "__main__":
    tf.app.run()
