"""FW Step size selection methods for Bayesian Logistic Regression."""
import sys, os
import numpy as np
import copy
import tensorflow as tf
from tensorflow.contrib.distributions import kl_divergence
import edward as ed
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture)
from scipy.misc import logsumexp as logsumexp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.optim.utils import divergence, grad_elbo, argmax_grad_dotp
from boosting_bbvi.core.utils import eprint, debug
import boosting_bbvi.core.utils as coreutils
logger = coreutils.get_logger()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'n_monte_carlo_samples', 1000,
    'Number of samples for approximating gradient')
flags.DEFINE_integer(
    'n_line_search_iter', 10,
    'Number of iterations line search gradient descent')
flags.DEFINE_enum('linit', 'fixed',
                  ['fixed', 'lipschitz_v2', 'lipschitz_v1'],
                  'Initialization methods versions for lipschitz constant')
flags.DEFINE_float('linit_fixed', 0.001,
                   'Fixed initial estimate of Lipschitz constant')
flags.DEFINE_integer('adafw_MAXITER', 32,
                     'Maximum iterations of adaptive fw L estimation')
flags.DEFINE_float('damping_adafw', 0.99, 'Damping constant')
flags.DEFINE_float('exp_adafw', 2.0, 'Multiplicative factor in L')
flags.DEFINE_enum(
    'distance_metric', 'dotproduct', ['dotproduct', 'kl', 'constant'],
    'Metric to use for distance norm between probability distrbutions')

def fixed(weights, comps, q_t, mu_s, cov_s, s_t, p, k):
    """Fixed step size.
    
    Args:
        weights: [k] weights of the mixture components of q_t
        comps: dictionary containing mixture parameters ('loc', 'scale')
        q_t: current solution iterate
        mu_s: [dim], mean of LMO solution s
        cov_s: [dim], cov matrix for LMO solution s
        s_t: Current atom & LMO Solution s
        p: Joint distribution p(z, x)
        k: iteration number of Frank-Wolfe
    Returns:
        dictionary containing gamma, new weights and new components
        and duality gap of current iterate
    """
    gamma = 2. / (k + 2.)
    new_weights = [(1. - gamma) * w for w in weights]
    new_weights.append(gamma)
    new_comps = copy.copy(comps)
    new_comps.append({'loc': mu_s, 'scale': cov_s})

    # Frank-Wolfe gap
    N_samples = FLAGS.n_monte_carlo_samples
    sample_q = q_t.sample([N_samples])
    sample_s = s_t.sample([N_samples])
    step_s = tf.reduce_mean(grad_elbo(q_t, p, sample_s)).eval()
    step_q = tf.reduce_mean(grad_elbo(q_t, p, sample_q)).eval()
    gap = step_q - step_s
    if gap < 0: logger.warning("Frank-Wolfe duality gap is negative")

    return {
        'gamma': gamma,
        'weights': new_weights,
        'comps': new_comps,
        'gap': gap,
    }
