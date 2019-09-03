"""Test step size variants of FW.

Usage:
    python test_step_size.py \
            --exp mixture \
            --n_monte_carlo_samples 100 \
            --n_line_search_iter 10 \
            --fw_variant=line_search \
            --linit_fixed 0.001 \
            --damping_adafw 0.5 \
            --exp_adafw 2.0 \
            --adafw_MAXITER 10 \
            --init_k 0 \
"""
from edward.models import Categorical, Normal, Mixture, MultivariateNormalDiag
from tensorflow.contrib.distributions import kl_divergence
import tensorflow as tf
import scipy.stats
import random
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import seaborn as sns
plt.style.use('ggplot')

from colorama import Fore, Style
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import boosting_bbvi.optim.fw_step_size as opt
import boosting_bbvi.scripts.mixture_model_relbo as mixture_model_relbo
import boosting_bbvi.core.utils as coreutils
from boosting_bbvi.core.utils import eprint, debug, append_to_file
logger = coreutils.get_logger()

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('init_k', 0.,
                   'Used to Initialize (simulating fw iteration)')


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
    return rerr

def plot_objective():
    weights_q = [0.6, 0.4]
    # weights_s = gamma is what we iterate on
    gammas = np.arange(0., 1., 0.02)
    # for exact gamma
    mus = [2., -1., 0.]
    stds = [.6, .4, 0.5]

    # for inexact approx
    mus2 = [-1., 1., 0., 2.0]
    stds2 = [3.3, 0.9, 0.5, 0.4]

    g = tf.Graph()
    with g.as_default():
        sess = tf.InteractiveSession()
        with sess.as_default():
            comps = [
                Normal(loc=tf.convert_to_tensor(mus[i], dtype=tf.float32),
                    scale=tf.convert_to_tensor(
                        stds[i], dtype=tf.float32))
                for i in range(len(mus))
            ]
            comps2 = [
                Normal(loc=tf.convert_to_tensor(mus2[i], dtype=tf.float32),
                    scale=tf.convert_to_tensor(
                        stds2[i], dtype=tf.float32))
                for i in range(len(mus2))
            ]
            # p = pi[0] * N(mus[0], stds[0]) + ... + pi[2] * N(mus[2], stds[2])
            weight_s = 0.5
            logger.info('true gamma for exact mixture %.2f' % (weight_s))
            final_weights = [(1 - weight_s) * w for w in weights_q]
            final_weights.append(weight_s)
            p = Mixture(
                cat=Categorical(probs=tf.convert_to_tensor(final_weights)),
                components=comps)

            objective_exact = []
            objective_inexact = []
            for gamma in gammas:
                new_weights = [(1 - gamma) * w for w in weights_q]
                new_weights.append(gamma)
                q = Mixture(
                    cat=Categorical(probs=tf.convert_to_tensor(new_weights)),
                    components=comps)
                objective = kl_divergence(q, p, allow_nan_stats=False).eval()
                objective_exact.append(objective)

                new_weights2 = [(1 - gamma) * w for w in final_weights]
                new_weights2.append(gamma)
                q2 = Mixture(
                    cat=Categorical(probs=tf.convert_to_tensor(new_weights2)),
                    components=comps2)
                objective2 = kl_divergence(q2, p, allow_nan_stats=False).eval()
                objective_inexact.append(objective2)

                logger.info(
                    'gamma = %.2f, D_kl_exact = %.5f, D_kl_inexact = %.5f' %
                    (gamma, objective, objective2))
    plt.plot(
        gammas,
        objective_exact,
        '-',
        color='r',
        linewidth=2.0,
        label='exact mixture')
    plt.plot(
        gammas,
        objective_inexact,
        '-',
        color='b',
        linewidth=2.0,
        label='inexact mixture')
    plt.legend()
    plt.xlabel('gamma')
    plt.ylabel('kl divergence of mixture')
    plt.show()


def test_adaptive_gamma():
    pi = np.array([0.2, 0.5, 0.3]).astype(np.float32)
    mus = [[2.], [-1.], [0.]]
    stds = [[.6], [.4], [0.5]]
    outfile = os.path.join(FLAGS.outdir, 'gamma.csv')
    g = tf.Graph()
    with g.as_default():
        sess = tf.InteractiveSession()
        with sess.as_default():
            # p = pi[0] * N(mus[0], stds[0]) + ... + pi[2] * N(mus[2], stds[2])
            p = Mixture(
                cat=Categorical(probs=tf.convert_to_tensor(pi)),
                components=[
                    #Normal(loc=tf.convert_to_tensor(mus[i], dtype=tf.float32),
                    #    scale=tf.convert_to_tensor(
                    #        stds[i], dtype=tf.float32)),
                    MultivariateNormalDiag(
                        loc=tf.convert_to_tensor(mus[i], dtype=tf.float32),
                        scale_diag=tf.convert_to_tensor(
                            stds[i], dtype=tf.float32))
                    for i in range(len(mus))
                ])
            qt = Mixture(
                cat=Categorical(probs=tf.convert_to_tensor(pi[:2])),
                components=[
                    MultivariateNormalDiag(
                        loc=tf.convert_to_tensor(mus[i], dtype=tf.float32),
                        scale_diag=tf.convert_to_tensor(
                            stds[i], dtype=tf.float32))
                    for i in range(len(mus[:2]))
                ])
            st = MultivariateNormalDiag(
                loc=tf.convert_to_tensor(mus[2], dtype=tf.float32),
                scale_diag=tf.convert_to_tensor(stds[2], dtype=tf.float32))

            if FLAGS.fw_variant == "line_search":
                gamma = opt.line_search_dkl(pi[:2], mus[:2], stds[:2], qt, mus[2],
                                            stds[2], st, p, FLAGS.init_k,
                                            return_gamma=True)
                # seed, n_line_search_iter, n_monte_carlo_samples, b, gamma
                append_to_file(outfile, "%d,%d,%d,%d,%f" %
                               (FLAGS.seed, FLAGS.n_line_search_iter,
                                FLAGS.n_monte_carlo_samples, 1, gamma))
            elif FLAGS.fw_variant == "adafw":
                gamma = opt.adaptive_fw(
                    weights=pi[:2],
                    locs=mus[:2],
                    diags=stds[:2],
                    q_t=qt,
                    mu_s=mus[2],
                    cov_s=stds[2],
                    s_t=st,
                    p=p,
                    k=FLAGS.init_k,
                    l_prev=opt.adafw_linit(qt, p),
                    return_gamma=True)
                # seed, n_monte_carlo_samples, eta, tau, linit, gamma
                append_to_file(outfile, "%d,%d,%f,%f,%f,%f" %
                               (FLAGS.seed, FLAGS.n_monte_carlo_samples,
                                FLAGS.damping_adafw, FLAGS.exp_adafw,
                                FLAGS.linit_fixed, gamma))
    print_err(pi[2], gamma)


def test_exact_gamma():
    pi = mixture_model_relbo.pi
    mus = mixture_model_relbo.mus
    stds = mixture_model_relbo.stds
    outfile = os.path.join(FLAGS.outdir, 'gamma.csv')
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(FLAGS.seed)
        sess = tf.InteractiveSession()
        with sess.as_default():
            # Build p = pi[0] * N(mu[0], std[0]) + pi[1] * N(mu[1], std[1])
            # thus, gamma = pi[1] (=0.6), q_t = N(mu[0], std[0])
            # s = N(mu[1], std[1])
            pcomps = [
                MultivariateNormalDiag(
                    loc=tf.convert_to_tensor(mus[i], dtype=tf.float32),
                    scale_diag=tf.convert_to_tensor(stds[i], dtype=tf.float32))
                for i in range(len(mus))
            ]
            p = Mixture(
                cat=Categorical(probs=tf.convert_to_tensor(pi[0])),
                components=pcomps)
            # build q_t
            weights = [1.]
            locs = [mus[0]]
            diags = [stds[0]]
            # Create current iter $q_t$
            qt = Mixture(
                cat=Categorical(probs=tf.convert_to_tensor(weights)),
                components=[
                    MultivariateNormalDiag(loc=loc, scale_diag=diag)
                    for loc, diag in zip(locs, diags)
                ])
            s = MultivariateNormalDiag(loc=mus[1], scale_diag=stds[1])
            if FLAGS.fw_variant == "line_search":
                gamma = opt.line_search_dkl(weights, locs, diags, qt, mus[1],
                                            stds[1], s, p, FLAGS.init_k,
                                            return_gamma=True)
                # seed, n_line_search_iter, n_monte_carlo_samples, b, gamma
                append_to_file(outfile, "%d,%d,%d,%d,%f" %
                               (FLAGS.seed, FLAGS.n_line_search_iter,
                                FLAGS.n_monte_carlo_samples, 1, gamma))
            elif FLAGS.fw_variant == "adafw":
                gamma = opt.adaptive_fw(
                    weights=weights,
                    locs=locs,
                    diags=diags,
                    q_t=qt,
                    mu_s=mus[1],
                    cov_s=stds[1],
                    s_t=s,
                    p=p,
                    k=FLAGS.init_k,
                    l_prev=1.,
                    return_gamma=True)
                # seed, n_monte_carlo_samples, eta, tau, linit, gamma
                append_to_file(outfile, "%d,%d,%f,%f,%f,%f" %
                               (FLAGS.seed, FLAGS.n_monte_carlo_samples,
                                FLAGS.damping_adafw, FLAGS.exp_adafw,
                                FLAGS.linit_fixed, gamma))
            else:
                raise NotImplementedError('other variants not tested yet.')
    print_err(pi[0][1], gamma)

def main(argv):
    #test_exact_gamma()
    test_adaptive_gamma()
    #plot_objective()


if __name__ == "__main__":
    tf.app.run()
