"""Test if Mixture from Edward and scipy.InfiniteMixtureScipy are equal.
"""

import os
import sys
import numpy as np

import tensorflow as tf

from edward.models import (Categorical, Normal, MultivariateNormalDiag, Mixture)

import edward as ed
import scipy.stats as stats

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.tests.test_step_size import print_err
from boosting_bbvi.core.utils import eprint, debug
from boosting_bbvi.scripts.mixture_model_relbo import construct_normal
from boosting_bbvi.core.infinite_mixture import InfiniteMixtureScipy
import boosting_bbvi.core.mvn as mvn


N_SAMPLES = 1000


def get_mixture_scipy(locs, diags, weights):
    q = InfiniteMixtureScipy(stats.multivariate_normal)
    q.weights = weights
    q.params = list(zip([[l] for l in locs], [[np.dot(d, d)] for d in diags]))
    return q


def get_tf_mixture(locs, diags, weights):
    q_comps = [
        MultivariateNormalDiag(
            loc=loc,
            scale_diag=scale_diag)
        for loc, scale_diag in zip(locs, diags)
    ]
    cat = Categorical(probs=tf.convert_to_tensor(weights))
    return Mixture(cat=cat, components=q_comps)

def test_mvn_same_as_edward_log_prob():
    loc = np.zeros(5)
    scale = np.ones(5)

    A = mvn.mvn(loc=loc, scale=scale)
    B = MultivariateNormalDiag(loc=loc, scale_diag=scale)
    samples = np.random.rand(5,5)
    tf.InteractiveSession()

    print('Log probability of Multivariate Normal Scipy vs Edward')
    print_err(
        tf.reduce_sum(A.log_prob(samples)).eval(),
        tf.reduce_sum(B.log_prob(samples)).eval())


def test_mixture_same_mean_variance():
    # distribution parameters
    pi = np.array([0.4, 0.6]).astype(np.float32)
    mus = np.array([[1.], [-1.]]).astype(np.float32)
    stds = np.array([[.6], [.6]]).astype(np.float32)

    mixture_mean = pi[0] * mus[0][0] + pi[1] * mus[1][0]
    # use moments to find closed form expression for variance of mixture
    # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
    mixture_var = (
        pi[0] * stds[0][0] * stds[0][0] + pi[1] * stds[1][0] * stds[1][0] +
        pi[0] * mus[0][0] * mus[0][0] + pi[1] * mus[1][0] * mus[1][0] -
        ((pi[0] * mus[0][0] + pi[1] * mus[1][0])**2))

    # test mixtures
    q_scipy = get_mixture_scipy(mus, stds, pi)
    q_edward = get_tf_mixture(mus, stds, pi)

    samples_scipy = q_scipy.sample_n(N_SAMPLES)
    mean_scipy = np.mean(samples_scipy)
    with tf.Session() as sess:
        samples_tf = q_edward.sample([N_SAMPLES])
        mean_tf, var_tf = tf.nn.moments(tf.reshape(samples_tf, [-1]), [0])
        print('for Edward.Models mean, variance...')
        print_err(mixture_mean, mean_tf.eval())
    test_mvn_same_as_edward_log_prob()
