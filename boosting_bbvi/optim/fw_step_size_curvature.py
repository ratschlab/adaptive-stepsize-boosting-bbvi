"""Frank-Wolfe step size selection methods.
"""
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
from boosting_bbvi.optim.utils import divergence, grad_kl, argmax_grad_dotp
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
flags.DEFINE_float('linit_fixed', 1.0,
                   'Fixed initial estimate of curvature constant')
flags.DEFINE_integer('adafw_MAXITER', 32,
                     'Maximum iterations of adaptive fw L estimation')
flags.DEFINE_float('damping_adafw', 0.99, 'Damping constant')
flags.DEFINE_float('exp_adafw', 2.0, 'Multiplicative factor in L')
flags.DEFINE_enum(
    'distance_metric', 'dotproduct', ['dotproduct', 'kl', 'constant'],
    'Metric to use for distance norm between probability distrbutions')

MIN_GAMMA = 0.001

def adafw_linit():
    return FLAGS.linit_fixed

def adaptive_fw(weights, locs, diags, q_t, mu_s, cov_s, s_t, p, k, c_prev,
                return_gamma=False):
    """Adaptive Frank-Wolfe algorithm.
    
    Sets step size as suggested in Algorithm 1 of
    https://arxiv.org/pdf/1806.05123.pdf

    Args:
        weights: [k], weights of the mixture components of q_t
        locs: [k x dim], means of mixture components of q_t
        diags: [k x dim], std deviations of mixture components of q_t
        q_t: current mixture iterate q_t
        mu_s: [dim], mean for LMO solution s
        cov_s: [dim], cov matrix for LMO solution s
        s_t: Current atom & LMO Solution s
        p: edward.model, target distribution p
        k: iteration number of Frank-Wolfe
        c_prev: previous curvature estimate
        return_gamma: only return the value of gamma
    Returns:
        If return_gamma is True, only the computed value of gamma
        is returned. Else returns a dictionary containing gamma, 
        curvature_estimate estimate, duality gap and step information
    """

    # Set $q_{t+1}$'s params
    new_locs = copy.copy(locs)
    new_diags = copy.copy(diags)
    new_locs.append(mu_s)
    new_diags.append(cov_s)

    N_samples = FLAGS.n_monte_carlo_samples
    # create and sample from $s_t, q_t$
    sample_q = q_t.sample([N_samples])
    sample_s = s_t.sample([N_samples])
    step_s = tf.reduce_mean(grad_kl(q_t, p, sample_s)).eval()
    step_q = tf.reduce_mean(grad_kl(q_t, p, sample_q)).eval()
    gap = step_q - step_s
    logger.info('duality gap %.5f' % gap)
    if gap < 0: logger.warning("Duality gap is negative returning 0 step")

    #gamma = 2. / (k + 2.)
    gamma = 0.
    tau = FLAGS.exp_adafw
    eta = FLAGS.damping_adafw
    # did the adaptive loop suceed or not
    step_type = "fixed"
    # NOTE: this is from v1 of the paper, new version
    # replaces multiplicative tau with divisor eta
    pow_tau = 1.0
    i, c_t = 0, c_prev
    f_t =  kl_divergence(q_t, p, allow_nan_stats=False).eval()
    debug('f(q_t) = %.5f' % (f_t))
    # return intial estimate if gap is -ve
    while gap >= 0:
        # compute $L_t$ and $\gamma_t$
        c_t = pow_tau * eta * c_prev
        gamma = min(gap / c_t , 1.0)
        d_1 = - gamma * gap
        d_2 = gamma * gamma * c_t / 2.
        debug('linear d1 = %.5f, quad d2 = %.5f' % (d_1, d_2))
        quad_bound_rhs = f_t  + d_1 + d_2

        # $w_{t + 1} = [(1 - \gamma)w_t, \gamma]$
        new_weights = copy.copy(weights)
        new_weights = [(1. - gamma) * w for w in new_weights]
        new_weights.append(gamma)
        qt_new = Mixture(
            cat=Categorical(probs=tf.convert_to_tensor(new_weights)),
            components=[
                MultivariateNormalDiag(loc=loc, scale_diag=diag)
                for loc, diag in zip(new_locs, new_diags)
            ])
        quad_bound_lhs = kl_divergence(qt_new, p, allow_nan_stats=False).eval()
        logger.info('ct = %.5f, gamma = %.3f, f_(qt_new) = %.5f, '
                    'linear extrapolated = %.5f' % (c_t, gamma, quad_bound_lhs,
                                                    quad_bound_rhs))
        if quad_bound_lhs <= quad_bound_rhs:
            step_type = "adaptive"
            break
        pow_tau *= tau
        i += 1
        #if i > FLAGS.adafw_MAXITER or gamma < MIN_GAMMA:
        if i > FLAGS.adafw_MAXITER:
            # estimate not good
            #gamma = 2. / (k + 2.)
            gamma = 0.
            c_t = c_prev
            step_type = "fixed_adaptive_MAXITER"
            break

    if return_gamma: return gamma
    return {
        'gamma': gamma,
        'c_estimate': c_t,
        'gap': gap,
        'step_type': step_type
    }


def line_search_dkl(weights, locs, diags, q_t, mu_s, cov_s, s_t, p, k,
                    return_gamma=False):
    """Performs line search for the best step size gamma.
    
    Uses gradient ascent to find gamma that minimizes
    KL(q_t + gamma (s - q_t) || p)
    
    Args:
        weights: [k], weights of mixture components of q_t
        locs: [k x dim], means of mixture components of q_t
        diags: [k x dim], deviations of mixture components of q_t
        q_t: current mixture iterate q_t
        mu_s: [dim], mean for LMO Solution s
        cov_s: [dim], cov matrix for LMO solution s
        s_t: Current atom & LMO Solution s
        p: edward.model, target distribution p
        k: iteration number of Frank-Wolfe
        return_gamma: only return the value of gamma
    Returns:
        If return_gamma is True, only the computed value of
        gamma is returned. Else along with gradient data
        is returned in a dict
    """
    N_samples = FLAGS.n_monte_carlo_samples
    # sample from $q_t$ and s
    sample_q = q_t.sample([N_samples])
    sample_s = s_t.sample([N_samples])
    # set $q_{t+1}$'s parameters
    new_locs = copy.copy(locs)
    new_diags = copy.copy(diags)
    new_locs.append(mu_s)
    new_diags.append(cov_s)
    # initialize $\gamma$
    gamma = 2. / (k + 2.)
    n_steps = FLAGS.n_line_search_iter
    prog_bar = ed.util.Progbar(n_steps)
    # storing gradients for analysis
    grad_gamma = []
    for it in range(n_steps):
        print("line_search iter %d, %.5f" % (it, gamma))
        new_weights = copy.copy(weights)
        new_weights = [(1. - gamma) * w for w in new_weights]
        new_weights.append(gamma)
        qt_new = Mixture(
            cat=Categorical(probs=tf.convert_to_tensor(new_weights)),
            components=[
                MultivariateNormalDiag(loc=loc, scale_diag=diag)
                for loc, diag in zip(new_locs, new_diags)
            ])
        rez_s = grad_kl(qt_new, p, sample_s).eval()
        rez_q = grad_kl(qt_new, p, sample_q).eval()
        grad_gamma.append({'E_s': rez_s, 'E_q': rez_q, 'gamma': gamma})
        # Gradient descent step size decreasing as $\frac{1}{it + 1}$
        gamma_prime = gamma - 0.1 * (np.mean(rez_s) - np.mean(rez_q)) / (it + 1.)
        # Projecting it back to [0, 1]
        if gamma_prime >= 1 or gamma_prime <= 0:
            gamma_prime = max(min(gamma_prime, 1.), 0.)

        if np.abs(gamma - gamma_prime) < 1e-6:
            gamma = gamma_prime
            break

        gamma = gamma_prime

    if return_gamma: return gamma
    return {'gamma': gamma, 'n_samples': N_samples, 'grad_gamma': grad_gamma}


def fully_corrective(q, p):
    """Fully corrective FW strategy"""
    comps = q.components
    n_comps = len(comps)
    # randomly initialize, rather than take q.cat as the initialization
    weights = np.random.random(n_comps).astype(np.float32)
    weights /= np.sum(weights)
    n_samples = 1000
    samples = [comp.sample(n_samples) for comp in comps]
    S = []
    for j in range(n_comps):
        for i in range(n_comps):
            comp_log_prob = tf.squeeze(comps[i].log_prob(samples[j]))
            S.append(comp_log_prob)
    S = tf.transpose(tf.reshape(tf.stack(S), [n_comps, n_comps, n_samples]))
    S = S.eval()

    p_log_probs = [tf.squeeze(p.log_prob(i)).eval() for i in samples]

    T = 1000000
    for t in range(T):
        grad = np.zeros(n_comps)
        for i in range(n_comps):
            part = np.zeros([n_samples, n_comps])
            for j in range(n_comps):
                probs = S[:, j, i]
                part[:, j] = np.log(weights[j] + 1e-10) + probs
            part = logsumexp(part, axis=1)
            diff = part - p_log_probs[i]
            grad[i] = np.mean(diff, axis=0)
        min_i = np.argmin(grad)
        corner = np.zeros(weights.shape)
        corner[min_i] = 1
        if t % 1000 == 0:
            print("grad", grad)
        duality_gap = -np.dot(grad, (corner - weights))
        if t % 1000 == 0:
            print("duality_gap", duality_gap)
        if duality_gap < 1e-6:
            return weights
        weights += 2. / (t + 2.) * (corner - weights)
        if t % 1000 == 0:
            print("weights", weights, t)
    print("weights", weights, t)
    return weights
