"""Utilities over probability distributions for Optimization.

Contains different divergence measures, metrics etc.

"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import kl_divergence
from tensorflow.python.ops.distributions import kullback_leibler

from edward.models.random_variable import RandomVariable
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture)

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.utils import eprint, debug

def setup_outdir(outdir):
    outdir = os.path.expanduser(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def softmax(v):
    return np.log(1 + np.exp(v))


def elbo(q, p, n_samples=1000):
    samples = q.sample(n_samples)
    elbo_samples = p.log_prob(samples) - q.log_prob(samples)
    elbo_samples = elbo_samples.eval()

    avg = np.mean(elbo_samples)
    std = np.std(elbo_samples)
    return avg, std

def divergence(q, p, metric='kl'):
    """Compute divergence measure between probability distributions.
    
    Args:
        q,p: probability distributions
        metric: only kl supported for now
    """
    if metric == 'kl':
        return kl_divergence(q, p)
    else:
        raise NotImplementedError('Metric not supported %s' % metric)


def is_distribution(p):
    """Check if p is one of pre-supported probability distributions.
    """
    DISTRIBUTION_LIST = [
        Normal, MultivariateNormalDiag, Mixture, Categorical, Empirical
    ]
    for dist in DISTRIBUTION_LIST:
        if isinstance(p, dist):
            return True
    return False

def grad_kl(q, p, theta):
    """Compute gradient of KL Divergence.
    
    Args:
        q, p: probability distributions
        theta: samples to compute
    Returns:
        float
    """
    # Functional Gradient w.r.t q $\nabla KL(q || p) = \log q - \log p$
    return q.log_prob(theta) - p.log_prob(theta)

# FIXME: samples can be from a different distribution
def lmo(y, p, n_samples=1000):
    """Approximate the lmo problem.
    
    Args:
        q: current approximation
        p: target distribution
        n_samples: monte carlo samples to approximate gradients
    """
    # $f = \mathcal{D}^{KL}(y || p)$
    # $\langle \nabla f , y \rangle = \mathbb{E}_{y}{\nabla f}$
    sample_y = y.sample([n_samples])
    return tf.reduce_mean(grad_kl(y, p, sample_y))

@kullback_leibler.RegisterKL(RandomVariable, RandomVariable)
def _kl_monte_carlo(q, p, n_samples=1000, name=None):
    """Compute Monte Carlo Estimate of KL divergence."""
    if not is_distribution(q) or not is_distribution(p):
        raise NotImplementedError(
            "type %s and type %s not supported. If they have a sample() and"
            "log_prob() method add them" % (type(distribution_a).__name__,
                                            type(distribution_b).__name__))
    samples = q.sample([n_samples])
    expectation_q = tf.reduce_mean(q.log_prob(samples))
    expectation_p = tf.reduce_mean(p.log_prob(samples))
    return expectation_q - expectation_p