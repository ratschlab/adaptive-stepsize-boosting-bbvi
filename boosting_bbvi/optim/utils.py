"""Utilities over probability distributions for Optimization.

Contains different divergence measures, metrics etc.

"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import kl_divergence
from tensorflow.python.ops.distributions import kullback_leibler

from edward.models.random_variable import RandomVariable
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture, Laplace, VectorLaplaceDiag)

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.utils import eprint, debug

def setup_outdir(outdir):
    outdir = os.path.expanduser(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def softmax(v):
    return np.log(1 + np.exp(v))


def elbo(q, joint, n_samples=1000, return_std=True):
    """Computes ELBO
    
    Args:
        q: Probability distribution
        joint: p(z, y | X)
        n_samples: samples to compute
        return_std: return standard deviation of ELBO
    Returns:
        mean, std of elbo if return_std is True, only ELBO ow
    """
    # TODO remove the return_std option
    # it is only for the mm example and is not very relevant
    samples = q.sample([n_samples])
    q_log_prob = q.log_prob(samples)
    p_log_prob = joint.log_prob(samples)
    elbo_samples = p_log_prob - q_log_prob
    avg, std = tf.nn.moments(elbo_samples, axes=[0])
    if return_std:
        return avg.eval(), std.eval()
    return avg.eval()


def grad_elbo(q, p_joint, theta):
    u"""Gradient of -ELBO w.r.t q
    
    Since KL(q || p(z|x)) = <q, log q - log p(z|x)>
    => ∇q(KL) = log q - log p(z|x)
    Now, ELBO(q) = <q, log p(z, x) - log q>
    => ∇q(-ELBO) = log q - log p(z, x)

    So, as gradient of ELBO only replaces the posterior by joint
    in gradient of KL term
    """
    return grad_kl(q, p_joint, theta)


def argmax_grad_dotp(p, q, candidates, n_samples=1000):
    u"""Find the candidate most aligned with ascent of objective function.

    (Or most unaligned with descent direction -∇f) Ascent direction is ∇f
    Objective function here is KL(q||p) where p is the target distribution.
    Objective function -ELBO(q) also works if p is joint p(z, q)
    
    Args:
        p, q, candidates
    Returns:
        Index of optimum candidate, and value
    """
    max_i, max_step = None, None
    for i, s in enumerate(candidates):
        sample_s = s.sample([n_samples])
        # NOTE: grad_kl will work for both KL and -ELBO
        step_s = tf.reduce_mean(grad_kl(q, p, sample_s)).eval()
        if i == 0 or max_step < step_s:
            max_step = step_s
            max_i = i
    return max_i, max_step


def divergence(q, p, metric='kl', n_monte_carlo_samples=1000):
    """Compute divergence measure between probability distributions.
    
    Args:
        q,p: probability distributions
        metric: divergence metric
        n_monte_carlo_samples: number of monte carlo samples for estimate
    """
    if metric == 'kl':
        return kl_divergence(q, p, allow_nan_stats=False)
    elif metric == 'dotproduct':
        samples_q = q.sample([n_monte_carlo_samples])
        distance_wrt_q = tf.reduce_mean(q.prob(samples_q) - p.prob(samples_q))
        samples_p = p.sample([n_monte_carlo_samples])
        distance_wrt_p = tf.reduce_mean(q.prob(samples_p) - p.prob(samples_p))
        return (distance_wrt_q - distance_wrt_p)
    elif metric == 'gradkl':
        raise NotImplementedError('Metric not supported %s' % metric)
    else:
        raise NotImplementedError('Metric not supported %s' % metric)


def is_distribution(p):
    """Check if p is one of pre-supported probability distributions.
    """
    DISTRIBUTION_LIST = [
        Normal, MultivariateNormalDiag, Mixture, Categorical, Empirical,
        Laplace, VectorLaplaceDiag
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

@kullback_leibler.RegisterKL(RandomVariable, RandomVariable)
def _kl_monte_carlo(q, p, n_samples=1000, name=None):
    """Compute Monte Carlo Estimate of KL divergence."""
    if not is_distribution(q) or not is_distribution(p):
        raise NotImplementedError(
            "type %s and type %s not supported. If they have a sample() and"
            "log_prob() method add them" % (type(q).__name__,
                                            type(p).__name__))
    samples = q.sample([n_samples])
    expectation_q = tf.reduce_mean(q.log_prob(samples))
    expectation_p = tf.reduce_mean(p.log_prob(samples))
    return expectation_q - expectation_p
