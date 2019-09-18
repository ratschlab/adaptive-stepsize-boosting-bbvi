"""Utilities for Bayesian Matrix Factorization. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture)
import scipy.io as sio
flags = tf.app.flags
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_enum('exp', 'cbcl', ['synthetic', 'cbcl'], 'Dataset name')
tf.flags.DEFINE_string('datapath', 'data/chem', 'path containing data')
flags.DEFINE_integer('n_monte_carlo_samples', 1000,
                     'Number of samples for approximating gradient')


def build_toy_dataset(U, V, N, M, noise_std=0.1):
    R = np.dot(np.transpose(U), V) + np.random.normal(
        0, noise_std, size=(N, M))
    return R


def get_indicators(N, M, prob_std=0.5):
    ind = np.random.binomial(1, prob_std, (N, M))
    return ind

def softplus(x):
    # log1p(exp(x))
    return np.log(np.exp(x) + 1)


def get_random_components(D, N, M, n_components=10):
    "Get random LMO solutions."
    ret = []
    for i in range(n_components):
        loc_i = np.random.rand(D, (N + M)).astype(np.float32)
        scale_i = softplus(np.random.rand(D, (N + M))).astype(np.float32)
        ret.append({'loc': loc_i, 'scale': scale_i})
    return ret


def get_data():
    if FLAGS.exp == 'cbcl':
        tr = sio.loadmat(os.path.join(FLAGS.datapath, 'cbcl.mat'))['V']
        N,M = tr.shape
        I_train = get_indicators(N, M, FLAGS.mask_ratio)
        I_test = 1 - I_train
        return N, M, FLAGS.D, tr, I_train, I_test
    elif FLAGS.exp == 'synthetic':
        N, M, D = FLAGS.N, FLAGS.M, FLAGS.D
        # true latent factors
        U_true = np.random.randn(D, N)
        V_true = np.random.randn(D, M)
        R_true = build_toy_dataset(U_true, V_true, N, M)
        I_train = get_indicators(N, M, FLAGS.mask_ratio)
        I_test = 1 - I_train
        return N, M, D, R_true, I_train, I_test
    pass


class Joint:
    '''Wrapper to handle joint probability p(UV, R_train)
    
        log p(UV, R_train) = log [ p(R_train | UV) * p(UV) ]
    '''

    def __init__(self, R_true, I_train, sess, D, N, M):
        """
        Args:
            R_true: full matrix
            I_train: training mask
        """
        self.n_samples = FLAGS.n_monte_carlo_samples
        self.R = tf.constant(R_true, dtype=tf.float32)
        self.I = tf.constant(I_train, dtype=tf.float32)
        self.D = D
        self.N = N
        self.M = M
        scale_uv = tf.concat(
            [tf.ones([D, N]),
                tf.ones([D, M])], axis=1)
        mean_uv = tf.concat(
            [tf.zeros([D, N]),
                tf.zeros([D, M])], axis=1)

        self.prior_UV = Normal(loc=mean_uv, scale=scale_uv) # (D, N + M)

    def log_lik(self, sample_uv):
        """
        Args:
            sample_uv: single (D, (N + M)) samples from qUV
        Returns:
            tensor scalar of log likelihood
        """
        # constructed matrix dist. R ~ N(U'V, 1)
        pR = Normal(
            loc=tf.matmul(tf.transpose(sample_uv[:, :self.N]), sample_uv[:, self.N:]),
            scale=tf.ones([self.N, self.M]))  # dist (N, M)
        full_log_likelihood = pR.log_prob(self.R)  # (N, M)
        full_log_likelihood_t = full_log_likelihood.eval()
        train_log_likelihood = full_log_likelihood * self.I # (N, M)
        log_lik = tf.reduce_sum(train_log_likelihood) # ()
        return log_lik


    def log_prob(self, sample_uv):
        """
        Args:
            sample_uv: single (D, (N + M)) samples from qUV
        Returns:
            tensor scalar of log_prob
        """
        prior_batch = self.prior_UV.log_prob(sample_uv) # (D, N + M)
        prior = tf.reduce_sum(prior_batch) 
        ll = self.log_lik(sample_uv)
        #print('DEBUG values', prior.eval(), ll.eval())
        p_joint = prior + ll
        #return prior
        return p_joint

    def log_prob_batch(self, samples):
        """
            samples: (n_samples, D, N + M) tensor
        """
        raise NotImplementedError('what to do here? just run in a loop?')


def log_likelihood(qR, R, I, sess, D, N, M):
    """
        qR: posterior distribution
        R: Data matrix
        I: mask for train/test
        sess: tf session
        D, N, M: dimensions
    """
    p_joint = Joint(R, I, sess, D, N, M)
    theta = qR.sample()
    log_lik_evals = [
        p_joint.log_lik(theta).eval() for _ in range(FLAGS.n_monte_carlo_samples)
    ]
    ll = np.mean(elbo_evals)
    return ll

def grad_kl_dotp(q, p, p_theta):
    """Compute dot product of gradient of KL-divergence/-ELBO w.r.t p_theta.

    It means evaluate log q/p  on samples from p_theta
    
    Args:
        q, p: probability distributions
        p_theta: distribution to compute expectation w.r.t
    Returns:
        float
    """
    theta = p_theta.sample() # single sample
    grad_kl_tensor = tf.reduce_sum(q.log_prob(theta)) - p.log_prob(theta)
    grad_kl_evals = [
        grad_kl_tensor.eval() for _ in range(FLAGS.n_monte_carlo_samples)
    ]
    grad_kl = np.mean(grad_kl_evals)
    return grad_kl


def argmax_grad_dotp(p, q, candidates):
    u"""Find the component most aligned with ∇f

    finds s ∈ candidates argmax <log q/p, s>
    
    Args:
        p, q, candidates
    Returns:
        Index of optimum candidate, and value
    """
    maxi, max_step = None, None
    for i, s in enumerate(candidates):
        step_s = grad_kl_dotp(q, p, s)
        if i == 0 or max_step < step_s:
            max_step = step_s
            max_i = i
    return max_i, max_step


def elbo(q, joint):
    """Return ELBO 
    
    Args:
        q: distribution q
        joint: p(z, x)
    Returns:
        float
    """
    theta = q.sample()
    p_log_prob = joint.log_prob(theta)
    q_log_prob = tf.reduce_sum(q.log_prob(theta))
    elbo_tensor = p_log_prob - q_log_prob
    elbo_evals = [
        elbo_tensor.eval() for _ in range(FLAGS.n_monte_carlo_samples)
    ]
    elbo = np.mean(elbo_evals)
    return elbo


def divergence(q, p, metric='kl'):
    """Compute divergence measure between probability distributions.
    
    Args:
        q,p: probability distributions
        metric: divergence metric
    Returns:
        float
    """
    if metric == 'kl':
        theta = q.sample()  # (D, N + M)
        expectation_tensor = tf.reduce_sum(q.log_prob(theta)) - tf.reduce_sum(
            p.log_prob(theta))
        exp_evals = [
            expectation_tensor.eval()
            for _ in range(FLAGS.n_monte_carlo_samples)
        ]
        return np.mean(exp_evals)
    elif metric == 'dotproduct':
        raise NotImplementedError('Metric not supported %s' % metric)
    elif metric == 'gradkl':
        raise NotImplementedError('Metric not supported %s' % metric)
    else:
        raise NotImplementedError('Metric not supported %s' % metric)
